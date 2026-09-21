"""Aggregated results of one run group (run-groups protocol, single_setup).

The observations of a window-size block are the temporal splits of that window size; the observations of the
``overall`` block are the window sizes (each method's mean over the splits of a window size). Methods are named
``<family>_<method>``. Each block holds, per subset size: the ranking (paired Wilcoxon tests with Holm correction
against Random and against the strongest baseline, dominance ranks; a test needs at least six non-zero differences),
the error statistics of the compared methods, the probability of superiority over the strongest baseline, the
win/tie/loss record against the per-seed baseline draws, the error by prediction horizon, and the statistics of
every evaluated method. Two methods observed on different numbers of units are compared on the first units of the
longer series. The compared methods are the three reference baselines plus the ``n_top`` methods with the lowest mean
RMSE (restricted, where possible, to methods that beat the median draw of the strongest baseline in more than half of
the seeds); every ranking block lists them under ``methods_covered``.
"""
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

from selection.determinism import exact_binomial_interval

from .statistics import describe, holm_adjust, paired_cliff_delta, probability_of_superiority

FAMILIES = ("baselines", "embedding_strata", "pure_embedding", "clustering", "shortlist")
TIE_TOLERANCE = 0.01
ALPHA = 0.05
MIN_DIFFERENCES = 6   # smallest number of non-zero differences for a Wilcoxon test
RANDOM = "baselines_random"
REFERENCE_BASELINES = (RANDOM, "baselines_difficulty_stratified", "baselines_consistency_stratified")
LABELS = {"random": "Random", "difficulty_stratified": "Diff-Strat", "consistency_stratified": "Consist-Strat",
          "repo_stratified": "Repo-Strat", "repo_difficulty_stratified": "Repo*Diff", "repo_consistency_stratified": "Repo*Consist",
          "fl": "FL", "medoid": "Medoid", "centroid": "Centroid", "ks": "KS", "core_edge": "Core/Edge", "kmeans": "K-Means",
          "hdbscan": "HDBSCAN", "pooled": "Pooled", "ts": "TS", "strata": "+Strata", "nostrata": "NoStrata", "cal": "+Cal",
          "features": "Feat", "features_top5": "Feat5"}
FAMILY_TAGS = {"embedding_strata": "[ES]", "pure_embedding": "[PE]", "clustering": "[CL]", "shortlist": "[SL]"}


# Names and inputs

def family_of(name):
    """Family of a method name "<family>_<method>", "unknown" without a known prefix."""
    return next((f for f in FAMILIES if name.startswith(f + "_")), "unknown")


def local_name(name):
    """Method name without its family prefix."""
    family = family_of(name)
    return name[len(family) + 1:] if family != "unknown" else name


def label_of(name):
    """Display label of a method, e.g. "embedding_strata_fl_pooled" -> "FL Pooled [ES]"."""
    family, local = family_of(name), local_name(name)
    if family == "baselines":
        return LABELS.get(local, local.replace("_", " ").title())
    parts, words, i = local.split("_"), [], 0
    while i < len(parts):
        if i + 1 < len(parts) and f"{parts[i]}_{parts[i + 1]}" in LABELS:
            words.append(LABELS[f"{parts[i]}_{parts[i + 1]}"])
            i += 2
            continue
        part = parts[i]
        words.append(LABELS[part] if part in LABELS else part.replace("x", "*") if part[:-1].isdigit() and part.endswith("x") else part.title())
        i += 1
    return " ".join(words) + (f" {FAMILY_TAGS[family]}" if family in FAMILY_TAGS else "")


def result_file(folder, stem):
    """``<stem>.json`` or, for a run over some window sizes only, ``<stem>_w<W>-<W>.json``; None when absent."""
    matches = sorted(p for p in Path(folder).glob(f"{stem}*.json") if p.name == f"{stem}.json" or p.name.startswith(f"{stem}_w"))
    matches = [p for p in matches if not p.name.endswith(".runlog.json")]
    if len(matches) > 1:
        raise SystemExit(f"More than one result file for {stem} in {folder}: {[m.name for m in matches]}")
    return matches[0] if matches else None


def load_run_group(folder, n_runs):
    """{window: table} of one run group from its family result files ``<family>_<N>runs.json``.

    A table holds, per subset size and method, the split-level RMSE, MaxErr and P95 (``dist_means``,
    ``dist_maxerr``, ``dist_p95``: one value per temporal split).
    """
    tables, families = {}, {}
    for family in FAMILIES:
        path = result_file(folder, f"{family}_{n_runs}runs")
        if path is None:
            continue
        stored = json.loads(path.read_text())
        families[family] = list(stored["methods"])
        for window, cells in stored["windows"].items():
            table = tables.setdefault(int(window), {"window": int(window), "n_runs": n_runs, "methods": [], "results": {}})
            for key, methods in cells.items():
                for method, entry in methods.items():
                    name = method if method.startswith(family) else f"{family}_{method}"
                    n_splits = len(entry["split_errors"])
                    maxerr = entry["split_maxerr"] if len(entry.get("split_maxerr", [])) == n_splits else [entry.get("maxerr", 0.0)] * n_splits
                    p95 = entry["split_p95"] if len(entry.get("split_p95", [])) == n_splits else [entry.get("p95", 0.0)] * n_splits
                    table["results"].setdefault(key, {})[name] = {"dist_means": entry["split_errors"], "dist_maxerr": maxerr, "dist_p95": p95}
    if not tables:
        raise SystemExit(f"No result files of the {n_runs}-run group in {folder}")
    names = sorted({name for t in tables.values() for cell in t["results"].values() for name in cell})
    for table in tables.values():
        table["methods"] = names
        table["n_observations"] = max((len(e["dist_means"]) for cell in table["results"].values() for e in cell.values()), default=0)
    return dict(sorted(tables.items())), families


def overall_table(tables):
    """Table whose observations are the window sizes: each method's mean over the splits of every window size it has."""
    windows = sorted(tables)
    names = list(dict.fromkeys(name for w in windows for cell in tables[w]["results"].values() for name in cell))
    results = {}
    for key in tables[windows[0]]["results"]:
        results[key] = {}
        for name in names:
            present = [tables[w]["results"][key][name] for w in windows if name in tables[w]["results"].get(key, {})]
            if present:
                results[key][name] = {field: [float(np.mean(entry[field])) for entry in present if entry[field]]
                                      for field in ("dist_means", "dist_maxerr", "dist_p95")}
    return {"window": "overall", "n_runs": tables[windows[0]]["n_runs"], "methods": sorted(names), "results": results,
            "n_observations": len(windows)}


def load_comparisons(folder, n_runs, windows):
    """{window: {"comparisons", "baselines"}} from ``determinism_<N>runs.json`` (empty when the file is absent)."""
    path = result_file(folder, f"determinism_{n_runs}runs")
    if path is None:
        return {}
    stored = json.loads(path.read_text()).get("windows", {})
    loaded = {}
    for window in windows:
        if str(window) in stored:
            entry = stored[str(window)]
            loaded[window] = {"baselines": entry.get("baselines", []),
                              "comparisons": {key: {m: {b: {"win_rate": v.get("win_rate", 50), "gap_pct": v.get("gap_pct", 0)} for b, v in per.items()}
                                                    for m, per in methods.items()} for key, methods in entry.get("comparisons", {}).items()}}
    return loaded


def overall_comparisons(comparisons, keys):
    """Win rates and gaps averaged over the window sizes."""
    if not comparisons:
        return None
    merged = {"comparisons": {}, "baselines": []}
    for key in keys:
        baselines = sorted({b for c in comparisons.values() for b in c.get("baselines", [])})
        methods = list(dict.fromkeys(m for c in comparisons.values() for m in c["comparisons"].get(key, {})))
        merged["comparisons"][key] = {}
        for method in methods:
            merged["comparisons"][key][method] = {}
            for baseline in baselines:
                entries = [c["comparisons"][key][method][baseline] for c in comparisons.values()
                           if baseline in c["comparisons"].get(key, {}).get(method, {})]
                if entries:
                    merged["comparisons"][key][method][baseline] = {"win_rate": float(np.mean([e["win_rate"] for e in entries])),
                                                                    "gap_pct": float(np.mean([e["gap_pct"] for e in entries]))}
        merged["baselines"] = merged["baselines"] or baselines
    return merged


def load_draws(folder, n_runs, window):
    """{baseline: {subset key: RMSE[splits, seeds]}} of one window size, or None."""
    path = Path(folder) / f"determinism_{n_runs}runs_w{window}.npz"
    if not path.exists():
        return None
    draws = {}
    with np.load(path) as stored:
        for name in stored.files:
            parts = name.split("__")
            if len(parts) == 2:
                draws.setdefault(parts[1], {})[parts[0]] = stored[name]
    return draws


# Compared methods

def strongest_baseline(window, methods):
    """Diff-Strat for W = 1, Consist-Strat otherwise; Random when only baselines were evaluated."""
    if methods and all(m.startswith("baselines_") for m in methods):
        return RANDOM
    return "baselines_consistency_stratified" if window == "overall" or window >= 2 else "baselines_difficulty_stratified"


def top_methods(table, key, n_top, comparisons):
    """The ``n_top`` methods with the lowest mean RMSE plus the reference baselines (marked) when not among them."""
    results = table["results"].get(key, {})
    reference = strongest_baseline(table["window"], table["methods"])
    compared = comparisons["comparisons"][key] if comparisons is not None and key in comparisons.get("comparisons", {}) else None
    candidates = []
    for name, entry in results.items():
        rmse, maxerr = np.array(entry["dist_means"], dtype=float), np.array(entry["dist_maxerr"], dtype=float)
        if len(rmse) == 0:
            continue
        candidate = {"name": name, "label": label_of(name), "mean_rmse": float(np.mean(rmse)),
                     "mean_maxerr": float(np.mean(maxerr)) if len(maxerr) else 0, "dist_means": rmse, "dist_maxerr": maxerr, "win_rate": None}
        if compared is not None:
            record = compared.get(name) or compared.get(local_name(name))
            against = (record.get(reference) or record.get(local_name(reference))) if record else None
            if against:
                candidate["win_rate"] = against.get("win_rate", 0)
        candidates.append(candidate)
    kept = candidates
    if compared is not None:
        kept = [c for c in candidates if c["win_rate"] is not None and c["win_rate"] > 50]
        if len(kept) < n_top:
            kept = candidates
    kept = sorted(kept, key=lambda c: (c["mean_rmse"], c["mean_maxerr"]))
    top = kept[:n_top]
    for baseline in (reference, RANDOM):
        if baseline not in {c["name"] for c in top}:
            extra = next((c for c in candidates if c["name"] == baseline), None)
            if extra:
                extra["reference_only"] = True
                top.append(extra)
    return top


def compared_methods(table, key, n_top, comparisons):
    """{name: values} of the reference baselines followed by the top methods."""
    results, methods = table["results"].get(key, {}), {}
    for name in REFERENCE_BASELINES:
        if name in results and len(results[name]["dist_means"]):
            methods[name] = {"dist_means": np.array(results[name]["dist_means"], dtype=float),
                             "dist_maxerr": np.array(results[name]["dist_maxerr"], dtype=float),
                             "label": label_of(name), "group": "baselines", "is_bl": True}
    baselines = set(methods)
    for candidate in top_methods(table, key, n_top, comparisons):
        name = candidate["name"]
        if name in baselines or candidate.get("reference_only") or name not in results or len(results[name]["dist_means"]) == 0:
            continue
        methods[name] = {"dist_means": candidate["dist_means"], "dist_maxerr": candidate["dist_maxerr"], "label": candidate["label"],
                         "group": family_of(name), "is_bl": False}
    return methods


# Blocks

def wilcoxon_p(differences):
    """Two-sided Wilcoxon p-value of the non-zero differences (1.0 with fewer than six of them)."""
    nonzero = differences[differences != 0]
    if len(nonzero) < MIN_DIFFERENCES:
        return 1.0
    try:
        return stats.wilcoxon(nonzero, alternative="two-sided")[1]
    except Exception:   # undefined for degenerate inputs
        return 1.0


def dominance_ranks(methods):
    """{name: number of compared methods that beat it significantly} (pairwise Wilcoxon, Holm over all pairs)."""
    names = sorted(methods, key=lambda n: np.median(methods[n]["dist_means"]))
    medians = {n: float(np.median(methods[n]["dist_means"])) for n in names}
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    p_values = []
    for i, j in pairs:
        a, b = methods[names[i]]["dist_means"], methods[names[j]]["dist_means"]
        shared = min(len(a), len(b))
        p_values.append(wilcoxon_p(b[:shared] - a[:shared]))
    beaten_by = {n: 0 for n in names}
    for (i, j), p in zip(pairs, holm_adjust(p_values)):
        if p < ALPHA and medians[names[i]] != medians[names[j]]:
            beaten_by[names[j] if medians[names[i]] < medians[names[j]] else names[i]] += 1
    return beaten_by


def comparison_rows(methods, reference):
    """Paired comparison of every compared method with the reference; Holm over these comparisons."""
    ref_rmse, ref_maxerr = methods[reference]["dist_means"], methods[reference]["dist_maxerr"]
    rows = []
    for name, m in methods.items():
        if name == reference:
            continue
        shared = min(len(ref_rmse), len(m["dist_means"]))
        ref, own = ref_rmse[:shared], m["dist_means"][:shared]
        delta, magnitude = paired_cliff_delta(ref, own)
        mean_ref, mean_own = np.mean(ref), np.mean(own)
        shared_maxerr = min(len(ref_maxerr), len(m["dist_maxerr"]))
        mean_ref_maxerr = np.mean(ref_maxerr[:shared_maxerr]) if shared_maxerr else 0
        mean_own_maxerr = np.mean(m["dist_maxerr"][:shared_maxerr]) if shared_maxerr else 0
        tolerance = TIE_TOLERANCE * ref
        wins, losses = int(np.sum(own < ref - tolerance)), int(np.sum(own > ref + tolerance))
        rows.append({"name": name, "label": m["label"], "group": m["group"], "is_bl": m["is_bl"],
                     "median": float(np.median(m["dist_means"])), "mean": float(np.mean(m["dist_means"])),
                     "wilcox_p": wilcoxon_p(ref - own), "cliff_d": delta, "cliff_mag": magnitude,
                     "imp_rmse": (mean_ref - mean_own) / mean_ref * 100 if mean_ref > 0 else 0,
                     "imp_maxerr": (mean_ref_maxerr - mean_own_maxerr) / mean_ref_maxerr * 100 if mean_ref_maxerr > 0 else 0,
                     "wins": wins, "ties": shared - wins - losses, "losses": losses})
    for row, adjusted in zip(rows, holm_adjust([r["wilcox_p"] for r in rows])):
        row["holm_p"] = float(adjusted)
    return rows


def ranking_block(table, key, n_top, comparisons):
    """Ranking of the compared methods for one subset size."""
    methods = compared_methods(table, key, n_top, comparisons)
    if len(methods) < 3 or RANDOM not in methods:
        return {}
    reference = strongest_baseline(table["window"], table["methods"])
    reference = reference if reference in methods else RANDOM
    ranks = dominance_ranks(methods)
    return {"dominance_rank": {methods[name]["label"]: rank for name, rank in ranks.items()},
            "vs_random": comparison_rows(methods, RANDOM), "vs_best_baseline": comparison_rows(methods, reference),
            "best_baseline_used": reference, "random_baseline": RANDOM, "methods_covered": list(methods)}


def distributions_block(table, key, n_top, comparisons):
    """Error statistics of the reference baselines and the top methods."""
    results, entries = table["results"].get(key, {}), []
    for name in REFERENCE_BASELINES:
        if name in results:
            entries.append({"name": name, "label": label_of(name), "is_baseline": True,
                            "dist_means": np.array(results[name]["dist_means"], dtype=float),
                            "dist_maxerr": np.array(results[name]["dist_maxerr"], dtype=float)})
    listed = {e["name"] for e in entries}
    for candidate in top_methods(table, key, n_top, comparisons):
        if candidate["name"] not in listed and not candidate.get("reference_only"):
            entries.append({"name": candidate["name"], "label": candidate["label"], "is_baseline": False,
                            "dist_means": candidate["dist_means"], "dist_maxerr": candidate["dist_maxerr"]})
    entries = entries[:n_top + len(REFERENCE_BASELINES)]
    if len(entries) < 2:
        return {}
    return {e["label"]: {"name": e["name"], "is_baseline": e["is_baseline"], "rmse": describe(e["dist_means"]), "maxerr": describe(e["dist_maxerr"])}
            for e in entries}


def superiority_block(table, key, n_top, comparisons):
    """Probability of superiority of every compared method over the strongest baseline."""
    methods = compared_methods(table, key, n_top, comparisons)
    if len(methods) < 2 or RANDOM not in methods:
        return {}
    reference = strongest_baseline(table["window"], table["methods"])
    base = methods[reference if reference in methods else RANDOM]["dist_means"]
    block = {"methods": {}}
    for m in methods.values():
        shared = min(len(m["dist_means"]), len(base))
        ps, magnitude = probability_of_superiority(m["dist_means"][:shared], base[:shared])
        block["methods"][m["label"]] = {"ps": ps, "ps_mag": magnitude}
    return block


def determinism_block(table, key, n_top, comparisons, draws):
    """Win/tie/loss of the compared methods against every baseline's per-seed draws.

    Window-size blocks count over all (split, seed) pairs with a tie tolerance of 1% of the draw; the ``overall`` block
    uses the win rates of the comparison file.
    """
    if comparisons is None:
        return {}
    compared, baselines = comparisons.get("comparisons", {}).get(key, {}), comparisons.get("baselines", [])
    if not compared or not baselines:
        return {}
    n_units = table["n_observations"]
    top = top_methods(table, key, n_top, comparisons)
    top_names = {n for c in top if not c.get("reference_only") for n in (c["name"], local_name(c["name"]))}
    results = table["results"].get(key, {})
    rows = []
    for name, record in compared.items():
        if not (name in top_names or any(name in c["name"] for c in top)) and len(compared) > n_top:
            continue
        match = next((c for c in top if name in c["name"] or c["name"].endswith("_" + name)), None)
        source = match["name"] if match and match["name"] in results else name if name in results else None
        rmse = np.array(results[source]["dist_means"], dtype=float) if source else None
        against = {}
        for baseline in baselines:
            if baseline not in record:
                continue
            seeds = draws[baseline][key] if draws and baseline in draws and key in draws[baseline] and rmse is not None else None
            if seeds is not None and seeds.shape[0] == len(rmse):
                tolerance, column = TIE_TOLERANCE * seeds, rmse[:, None]
                wins_per_seed, losses_per_seed = np.sum(column < seeds - tolerance, axis=0), np.sum(column > seeds + tolerance, axis=0)
                total, wins, losses = seeds.size, int(np.sum(wins_per_seed)), int(np.sum(losses_per_seed))
                low, high = exact_binomial_interval(wins, total)
                try:
                    p_value = float(stats.wilcoxon(wins_per_seed / seeds.shape[0] - 0.5, alternative="two-sided")[1])
                except Exception:   # undefined when all seeds give the same win rate
                    p_value = None
                against[baseline] = {"win": wins, "tie": total - wins - losses, "loss": losses, "n": total, "win_frac": float(wins / total if total else 0),
                                     "ci_lo": low, "ci_hi": high, "wilcoxon_p": p_value, "gap_pct": float(record[baseline].get("gap_pct", 0))}
                continue
            rate = float(record[baseline].get("win_rate", 50)) / 100
            total = max(n_units, 1)
            wins = int(round(rate * total))
            low, high = exact_binomial_interval(wins, total)
            against[baseline] = {"win": wins, "tie": 0, "loss": total - wins, "n": total, "win_frac": float(rate), "ci_lo": low, "ci_hi": high,
                                 "wilcoxon_p": None, "gap_pct": float(record[baseline].get("gap_pct", 0))}
        if against:
            rows.append({"name": name, "label": match["label"] if match else name.replace("_", " ").title()[:30], "baselines": against,
                         "avg_win_rate": np.mean([s["win_frac"] for s in against.values()])})
    rows.sort(key=lambda r: -r["avg_win_rate"])
    return {r["label"]: {"name": r["name"], "avg_win_rate": r["avg_win_rate"], "baselines": r["baselines"]} for r in rows}


def horizon_block(table, key, n_top, comparisons):
    """Median split RMSE by number of test runs of the split, with Spearman's rho."""
    window, n_runs = table["window"], table["n_runs"]
    by_horizon, position = defaultdict(list), 0
    for train in combinations(range(n_runs), window):
        if max(train) + 1 < n_runs:
            by_horizon[n_runs - max(train) - 1].append(position)
            position += 1
    horizons = sorted(by_horizon)
    if position == 0 or len(horizons) < 2:
        return {}
    results, block = table["results"].get(key, {}), {}
    for name, m in compared_methods(table, key, n_top, comparisons).items():
        errors = results.get(name, {}).get("dist_means", [])
        if not errors or len(errors) != position:
            continue
        medians = [float(np.median(np.array([errors[s] for s in by_horizon[h]]))) for h in horizons]
        rho = stats.spearmanr(horizons, medians)[0] if len(horizons) >= 3 else np.nan
        block[m["label"]] = {"horizons": {str(h): v for h, v in zip(horizons, medians)},
                             "spearman_rho": None if np.isnan(rho) else float(rho), "is_baseline": m["is_bl"]}
    return block


def sensitivity(tables, keys):
    """Method with the lowest mean RMSE per (window size, subset size), and how often each method is best."""
    cells, frequency = [], defaultdict(int)
    for window, table in sorted(tables.items()):
        for key, methods in table["results"].items():
            scores = sorted(((name, float(np.mean(entry["dist_means"]))) for name, entry in methods.items() if entry["dist_means"]),
                            key=lambda item: item[1])
            if scores:
                cells.append({"window": window, "bucket": "all", "pct_key": key, "best_method": scores[0][0], "best_value": scores[0][1],
                              "runner_up": scores[1][0] if len(scores) > 1 else None, "runner_up_value": scores[1][1] if len(scores) > 1 else None})
    if not cells:
        return {}
    for cell in sorted(cells, key=lambda c: (c["window"], c["pct_key"])):
        if cell["pct_key"] in keys:
            frequency[cell["best_method"]] += 1
    return {"win_freq": {label_of(m): count for m, count in frequency.items()}, "cells": [c for c in cells if c["pct_key"] in keys]}


def all_methods_block(table, keys):
    """Statistics and per-unit values of every evaluated method."""
    block = {}
    for key in keys:
        block[key] = {}
        for name, entry in table["results"].get(key, {}).items():
            rmse, maxerr = np.array(entry["dist_means"], dtype=float), np.array(entry["dist_maxerr"], dtype=float)
            if len(rmse):
                block[key][name] = {"label": label_of(name), "group": family_of(name), "rmse": describe(rmse),
                                    "maxerr": describe(maxerr) if len(maxerr) else {}, "dist_means": rmse.tolist(),
                                    "dist_maxerr": maxerr.tolist() if len(maxerr) else []}
    return block


def analysis_block(table, keys, n_top, comparisons, draws):
    """One analysis block (a window size, or all window sizes)."""
    baseline_only = all(m.startswith("baselines_") for m in table["methods"])
    block = {"window": str(table["window"]), "n_observations": table["n_observations"], "n_methods": len(table["methods"]), "subset_sizes": {}}
    for key in keys:
        cell = {}
        if not baseline_only:
            cell["ranking"] = ranking_block(table, key, n_top, comparisons)
        cell["distributions"] = distributions_block(table, key, n_top, comparisons)
        cell["cdf_tail"] = superiority_block(table, key, n_top, comparisons)
        if not baseline_only:
            cell["determinism"] = determinism_block(table, key, n_top, comparisons, draws)
        if table["window"] != "overall":
            cell["temporal_gap"] = horizon_block(table, key, n_top, comparisons)
        block["subset_sizes"][key] = cell
    block["all_methods"] = all_methods_block(table, keys)
    return block


def evaluate_run_group(folder, n_runs, keys, n_top):
    """Aggregated results of one run group and its family coverage."""
    tables, families = load_run_group(folder, n_runs)
    comparisons = load_comparisons(folder, n_runs, list(tables))
    result = {"_meta": {"version": "3.0", "tie_tolerance": TIE_TOLERANCE, "effect_size": "Cliff delta paired (Romano et al. 2006)",
                        "significance": "Wilcoxon + Holm-Bonferroni a=0.05", "ci": "Clopper-Pearson exact", "n_top": n_top}, "analyses": {}}
    result["analyses"]["overall"] = analysis_block(overall_table(tables), keys, n_top, overall_comparisons(comparisons, keys), None)
    for window, table in tables.items():
        result["analyses"][f"w{window}"] = analysis_block(table, keys, n_top, comparisons.get(window), load_draws(folder, n_runs, window))
    result["sensitivity"] = sensitivity(tables, keys)
    return result, families
