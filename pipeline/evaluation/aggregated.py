"""Aggregated results of a dataset: one analysis block per window size and one over all window sizes.

Each block holds, per subset size: the ranking (paired Wilcoxon tests with Holm correction against Random and
against the strongest baseline, dominance ranks), the error distributions of the compared methods, tail
statistics, the win/tie/loss record against the per-seed baseline draws, the error by prediction horizon, and the
statistics of every evaluated method. The compared methods are the three reference baselines plus the ``n_top``
methods with the lowest mean RMSE (restricted, where possible, to methods that beat the strongest baseline's
per-seed draws significantly); every Holm correction runs over exactly these methods, which each ranking block
lists under ``methods_covered``.
"""
from collections import defaultdict
from itertools import combinations
from math import comb

import numpy as np
from scipy import stats

from selection.determinism import exact_binomial_interval

from .inputs import family_of, local_name
from .statistics import describe, holm_adjust, paired_cliff_delta, paired_wilcoxon, probability_of_superiority

TIE_TOLERANCE = 0.01   # relative to the reference RMSE
ALPHA = 0.05
RANDOM = "baselines:random"
REFERENCE_BASELINES = (RANDOM, "baselines:difficulty_stratified", "baselines:consistency_stratified")
DRAWN_BASELINES = ("random", "difficulty_stratified", "consistency_stratified", "repo_stratified",
                   "repo_difficulty_stratified", "repo_consistency_stratified")
LABELS = {"random": "Random", "difficulty_stratified": "Diff-Strat", "consistency_stratified": "Consist-Strat",
          "repo_stratified": "Repo-Strat", "repo_difficulty_stratified": "Repo×Diff", "repo_consistency_stratified": "Repo×Consist",
          "fl": "FL", "medoid": "Medoid", "centroid": "Centroid", "ks": "KS", "core_edge": "Core/Edge", "kmeans": "K-Means",
          "hdbscan": "HDBSCAN", "pooled": "Pooled", "ts": "TS", "strata": "+Strata", "nostrata": "NoStrata", "cal": "+Cal",
          "features": "Feat", "features_top5": "Feat5"}
FAMILY_TAGS = {"embedding_strata": "[ES]", "pure_embedding": "[PE]", "clustering": "[CL]", "shortlist": "[SL]"}


# Names

def label_of(name):
    """Display label of a method, e.g. "embedding_strata:fl_pooled" -> "FL Pooled [ES]"."""
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
        words.append(LABELS[part] if part in LABELS else part.replace("x", "×") if part[:-1].isdigit() and part.endswith("x") else part.title())
        i += 1
    return " ".join(words) + (f" {FAMILY_TAGS[family]}" if family in FAMILY_TAGS else "")


def strongest_baseline(window, methods):
    """Reference baseline of a block: Diff-Strat for W = 1, Consist-Strat otherwise; Random when only baselines were evaluated."""
    if methods and all(m.startswith("baselines:") for m in methods):
        return RANDOM
    if window == "overall":
        return "baselines:consistency_stratified"
    return "baselines:consistency_stratified" if window >= 2 else "baselines:difficulty_stratified"


def only_baselines(table):
    """True when the table holds no method outside the baseline family."""
    return all(m.startswith("baselines:") for m in table["methods"])


# Aggregation over window sizes

def aggregate_windows(tables):
    """Per-distribution values averaged over the window sizes in which a method exists (equal weight per window)."""
    windows = sorted(tables)
    template = tables[windows[0]]
    methods = [m for m in dict.fromkeys(m for w in windows for m in tables[w]["methods"])]
    results = {}
    for key in template["results"]:
        results[key] = {}
        for method in methods:
            present = [tables[w]["results"][key][method] for w in windows if method in tables[w]["results"].get(key, {})]
            if present:
                results[key][method] = {field: np.mean([entry[field] for entry in present], axis=0) for field in ("dist_means", "dist_maxerr", "dist_p95")}
    return {"window": "overall", "methods": sorted(methods), "n_distributions": template["n_distributions"], "levels": template["levels"],
            "results": results, "source_windows": windows}


def aggregate_comparisons(comparisons_by_window, key):
    """Win/tie/loss counts summed over window sizes; Wilcoxon p-values combined with Fisher's method."""
    baselines = sorted({b for c in comparisons_by_window.values() for b in c.get("baselines", [])})
    methods = list(dict.fromkeys(m for c in comparisons_by_window.values() for m in c.get("comparisons", {}).get(key, {})))
    combined = {}
    for method in methods:
        combined[method] = {}
        for baseline in baselines:
            entries = [c["comparisons"][key][method][baseline] for c in comparisons_by_window.values()
                       if baseline in c.get("comparisons", {}).get(key, {}).get(method, {})]
            total = sum(int(e["dist_wtl"]["n"]) for e in entries)
            if total == 0:
                continue
            wins = sum(int(e["dist_wtl"]["win"]) for e in entries)
            p_values = [e["per_dist"]["wilcoxon_pval"] for e in entries if e["per_dist"]["wilcoxon_pval"] is not None]
            p_combined = float(stats.combine_pvalues(p_values, method="fisher")[1]) if len(p_values) >= 2 else p_values[0] if p_values else None
            combined[method][baseline] = {"win_rate": wins / total * 100, "gap_pct": float(np.mean([e["gap_pct"] for e in entries])),
                                          "dist_wtl": {"win": wins, "tie": sum(int(e["dist_wtl"]["tie"]) for e in entries),
                                                       "loss": sum(abs(int(e["dist_wtl"]["loss"])) for e in entries), "n": total},
                                          "per_dist": {"wilcoxon_pval": p_combined}}
    return combined, baselines


def overall_comparisons(comparisons_by_window, keys):
    """Determinism comparisons over all window sizes, in the layout of a single window."""
    if not comparisons_by_window:
        return None
    overall = {"comparisons": {}, "baselines": [], "n_distributions": 0}
    for key in keys:
        overall["comparisons"][key], baselines = aggregate_comparisons(comparisons_by_window, key)
        overall["baselines"] = overall["baselines"] or baselines
        overall["n_distributions"] += sum(c.get("n_distributions", 0) for c in comparisons_by_window.values())
    return overall


# Compared methods

def top_methods(table, key, n_top, comparisons):
    """The ``n_top`` methods with the lowest mean RMSE, plus the reference baselines (marked) when not among them.

    With determinism comparisons available, candidates are first restricted to methods whose win rate against the
    strongest baseline's per-seed draws is above 50% and significant; if fewer than ``n_top`` remain, to win rate
    above 50%; if still fewer, all methods are candidates.
    """
    results = table["results"].get(key, {})
    reference = strongest_baseline(table["window"], table["methods"])
    compared = comparisons["comparisons"][key] if comparisons is not None and key in comparisons.get("comparisons", {}) else None
    candidates = []
    for name, entry in results.items():
        if len(entry["dist_means"]) == 0:
            continue
        candidate = {"name": name, "label": label_of(name), "mean_rmse": float(np.mean(entry["dist_means"])),
                     "mean_maxerr": float(np.mean(entry["dist_maxerr"])) if len(entry["dist_maxerr"]) else 0,
                     "dist_means": entry["dist_means"], "dist_maxerr": entry["dist_maxerr"], "win": None}
        if compared is not None:
            record = compared.get(name) or compared.get(local_name(name))
            against = (record[reference] if reference in record else record.get(local_name(reference))) if record else None
            if against:
                p_value = against.get("per_dist", {}).get("wilcoxon_pval")
                candidate["win"] = {"rate": against.get("win_rate", 0), "significant": p_value is not None and p_value < ALPHA}
        candidates.append(candidate)
    kept = candidates
    if compared is not None:
        kept = [c for c in candidates if c["win"] and c["win"]["significant"] and c["win"]["rate"] > 50]
        if len(kept) < n_top:
            kept = [c for c in candidates if c["win"] and c["win"]["rate"] > 50]
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
    results = table["results"].get(key, {})
    references = (RANDOM,) if only_baselines(table) else REFERENCE_BASELINES
    methods = {}
    for name in references:
        if name in results and len(results[name]["dist_means"]):
            methods[name] = {"dist_means": results[name]["dist_means"], "dist_maxerr": results[name]["dist_maxerr"],
                             "label": label_of(name), "group": "baselines", "is_bl": True}
    baselines = set(methods)
    for candidate in top_methods(table, key, n_top, comparisons):
        name = candidate["name"]
        if name in baselines or candidate.get("reference_only") or name not in results or len(results[name]["dist_means"]) == 0:
            continue
        methods[name] = {"dist_means": results[name]["dist_means"], "dist_maxerr": results[name]["dist_maxerr"],
                         "label": candidate["label"], "group": family_of(name), "is_bl": False}
    return methods


# Ranking

def dominance_ranks(methods):
    """{name: number of compared methods that beat it significantly} (pairwise Wilcoxon, Holm over all pairs)."""
    names = sorted(methods, key=lambda n: np.median(methods[n]["dist_means"]))
    medians = {n: float(np.median(methods[n]["dist_means"])) for n in names}
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    adjusted = holm_adjust([paired_wilcoxon(methods[names[j]]["dist_means"] - methods[names[i]]["dist_means"]) for i, j in pairs])
    beaten_by = {n: 0 for n in names}
    for (i, j), p in zip(pairs, adjusted):
        if p < ALPHA and medians[names[i]] != medians[names[j]]:
            beaten_by[names[j] if medians[names[i]] < medians[names[j]] else names[i]] += 1
    return beaten_by


def comparison_rows(methods, reference, n_distributions):
    """Paired comparison of every compared method with the reference; Holm over these comparisons (reference excluded)."""
    ref_rmse, ref_maxerr = methods[reference]["dist_means"], methods[reference]["dist_maxerr"]
    rows = []
    for name, m in methods.items():
        if name == reference:
            continue
        rmse = m["dist_means"]
        delta, magnitude = paired_cliff_delta(ref_rmse, rmse)
        tolerance = TIE_TOLERANCE * ref_rmse
        wins, losses = int(np.sum(rmse < ref_rmse - tolerance)), int(np.sum(rmse > ref_rmse + tolerance))
        mean_ref, mean_ref_maxerr = np.mean(ref_rmse), np.mean(ref_maxerr)
        rows.append({"name": name, "label": m["label"], "group": m["group"], "is_bl": m["is_bl"],
                     "median": float(np.median(rmse)), "mean": float(np.mean(rmse)), "wilcox_p": paired_wilcoxon(ref_rmse - rmse),
                     "cliff_d": delta, "cliff_mag": magnitude,
                     "imp_rmse": (mean_ref - np.mean(rmse)) / mean_ref * 100 if mean_ref > 0 else 0,
                     "imp_maxerr": (mean_ref_maxerr - np.mean(m["dist_maxerr"])) / mean_ref_maxerr * 100 if mean_ref_maxerr > 0 else 0,
                     "wins": wins, "ties": n_distributions - wins - losses, "losses": losses})
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
            "vs_random": comparison_rows(methods, RANDOM, table["n_distributions"]),
            "vs_best_baseline": comparison_rows(methods, reference, table["n_distributions"]),
            "best_baseline_used": reference, "random_baseline": RANDOM, "methods_covered": list(methods)}


# Error distributions

def baseline_draw_values(draws_by_window, table, key):
    """{baseline name: per-seed values} for the blocks' baselines; over all windows the draw matrices are averaged."""
    results, values = table["results"].get(key, {}), {}
    if table["window"] == "overall":
        for baseline in DRAWN_BASELINES:
            rmse, maxerr, complete = [], [], True
            for window in table["source_windows"]:
                draws = draws_by_window.get(window)
                if draws and baseline in draws and key in draws[baseline]:
                    rmse.append(draws[baseline][key])
                    if f"{key}__maxerr" in draws[baseline]:
                        maxerr.append(draws[baseline][f"{key}__maxerr"])
                    else:
                        complete = False
            if rmse:
                mean_rmse = np.mean(rmse, axis=0)
                if complete and len(maxerr) == len(rmse):
                    values[f"baselines:{baseline}"] = {"dist_means": mean_rmse.flatten(), "dist_maxerr": np.mean(maxerr, axis=0).flatten()}
                else:
                    values[f"baselines:{baseline}"] = {"dist_means": mean_rmse.flatten(),
                                                        "dist_maxerr": results.get(f"baselines:{baseline}", {}).get("dist_maxerr", np.array([]))}
        return values or None
    draws = draws_by_window.get(table["window"])
    if draws is None:
        return None
    for baseline in DRAWN_BASELINES:
        if baseline in draws and key in draws[baseline]:
            maxerr = draws[baseline].get(f"{key}__maxerr")
            values[f"baselines:{baseline}"] = {"dist_means": draws[baseline][key].flatten(),
                                                "dist_maxerr": maxerr.flatten() if maxerr is not None
                                                else results.get(f"baselines:{baseline}", {}).get("dist_maxerr", np.array([]))}
    return values or None


def distributions_block(table, key, n_top, comparisons, draws_by_window):
    """Error statistics of the compared methods; reference baselines are described over their per-seed draws."""
    results = table["results"].get(key, {})
    references = (RANDOM,) if only_baselines(table) else REFERENCE_BASELINES
    drawn = baseline_draw_values(draws_by_window, table, key)
    entries = []
    for name in references:
        if drawn and name in drawn:
            entries.append({"name": name, "label": label_of(name), "is_baseline": True, **drawn[name]})
        elif name in results:
            entries.append({"name": name, "label": label_of(name), "is_baseline": True,
                            "dist_means": results[name]["dist_means"], "dist_maxerr": results[name]["dist_maxerr"]})
    listed = {e["name"] for e in entries}
    for candidate in top_methods(table, key, n_top, comparisons):
        if candidate["name"] in listed or candidate.get("reference_only"):
            continue
        values = drawn[candidate["name"]] if only_baselines(table) and drawn and candidate["name"] in drawn else candidate
        entries.append({"name": candidate["name"], "label": candidate["label"], "is_baseline": False,
                        "dist_means": values["dist_means"], "dist_maxerr": values["dist_maxerr"]})
    entries = entries[:n_top + len(references)]
    if len(entries) < 2:
        return {}
    return {e["label"]: {"name": e["name"], "is_baseline": e["is_baseline"], "rmse": describe(e["dist_means"]), "maxerr": describe(e["dist_maxerr"])}
            for e in entries}


# Tail statistics

def tail_block(table, key, n_top, comparisons, draws_by_window):
    """Probability of superiority over the strongest baseline and exceedance of its 75th/90th/95th RMSE percentiles."""
    results = table["results"].get(key, {})
    baseline_only = only_baselines(table)
    per_seed = {}
    if table["window"] == "overall":
        for baseline in DRAWN_BASELINES:
            matrices = [draws_by_window[w][baseline][key] for w in table["source_windows"]
                        if draws_by_window.get(w) and baseline in draws_by_window[w] and key in draws_by_window[w][baseline]]
            if matrices:
                per_seed[baseline] = np.mean(matrices, axis=0)
    else:
        per_seed = {b: d[key] for b, d in (draws_by_window.get(table["window"]) or {}).items() if key in d}
    methods = {}
    for name in ((RANDOM,) if baseline_only else REFERENCE_BASELINES):
        if name in results and len(results[name]["dist_means"]):
            methods[name] = {"dist_means": results[name]["dist_means"], "per_seed": per_seed.get(local_name(name)), "label": label_of(name)}
    for candidate in top_methods(table, key, n_top, comparisons):
        name = candidate["name"]
        if name in methods or candidate.get("reference_only") or name not in results or len(results[name]["dist_means"]) == 0:
            continue
        methods[name] = {"dist_means": results[name]["dist_means"], "label": candidate["label"],
                         "per_seed": per_seed.get(local_name(name)) if baseline_only else None}
    if len(methods) < 2:
        return {}
    values = lambda m: m["per_seed"].ravel() if m["per_seed"] is not None else m["dist_means"]   # noqa: E731
    reference = strongest_baseline(table["window"], table["methods"])
    block = {"tail_thresholds": {}, "methods": {}}
    if reference in methods:
        block["tail_thresholds"] = {f"P{p}": float(np.percentile(values(methods[reference]), p)) for p in (75, 90, 95)}
    for m in methods.values():
        ps, magnitude = probability_of_superiority(m["dist_means"], methods[reference]["dist_means"]) if reference in methods else (0.5, "N/A")
        block["methods"][m["label"]] = {"ps": ps, "ps_mag": magnitude,
                                        "tail_exceedance": {k: float(np.mean(values(m) > v)) for k, v in block["tail_thresholds"].items()}}
    return block


# Win/tie/loss against the per-seed draws

def determinism_block(table, key, n_top, comparisons, draws_by_window):
    """Win/tie/loss of the compared methods against every baseline's per-seed draws.

    Single window: counted over all (distribution, seed) pairs with a tie tolerance of 1% of the draw. Over all
    windows: the per-window counts against the median draw are summed.
    """
    if only_baselines(table) or comparisons is None:
        return {}
    compared, baselines = comparisons.get("comparisons", {}).get(key, {}), comparisons.get("baselines", [])
    if not compared or not baselines:
        return {}
    n_distributions = comparisons.get("n_distributions", 1000)
    top = [c for c in top_methods(table, key, n_top, comparisons)]
    top_names = {n for c in top if not c.get("reference_only") for n in (c["name"], local_name(c["name"]))}
    draws = draws_by_window.get(table["window"]) if table["window"] != "overall" else None
    results = table["results"].get(key, {})
    rows = []
    for name, record in compared.items():
        if not (name in top_names or any(name in c["name"] for c in top)) and len(compared) > n_top:
            continue
        match = next((c for c in top if name in c["name"] or c["name"].endswith("_" + name)), None)
        rmse = results[match["name"]]["dist_means"] if match and match["name"] in results else results[name]["dist_means"] if name in results else None
        against = {}
        for baseline in baselines:
            if baseline not in record:
                continue
            seeds = draws[baseline][key] if draws and baseline in draws and key in draws[baseline] and rmse is not None else None
            if seeds is not None and seeds.shape[0] == len(rmse):
                tolerance, column = TIE_TOLERANCE * seeds, rmse[:, None]
                wins_per_seed, losses_per_seed = np.sum(column < seeds - tolerance, axis=0), np.sum(column > seeds + tolerance, axis=0)
                total = seeds.size
                wins, losses = int(np.sum(wins_per_seed)), int(np.sum(losses_per_seed))
                low, high = exact_binomial_interval(wins, total)
                try:
                    p_value = float(stats.wilcoxon(wins_per_seed / seeds.shape[0] - 0.5, alternative="two-sided")[1])
                except Exception:   # undefined when all seeds give the same win rate
                    p_value = None
                against[baseline] = {"win": wins, "tie": total - wins - losses, "loss": losses, "n": total, "win_frac": float(wins / total if total else 0),
                                     "ci_lo": low, "ci_hi": high, "wilcoxon_p": p_value, "gap_pct": float(record[baseline].get("gap_pct", 0))}
                continue
            counts = record[baseline].get("dist_wtl", {})
            wins, total = int(counts["win"]), int(counts.get("n", n_distributions))
            low, high = exact_binomial_interval(wins, total)
            against[baseline] = {"win": wins, "tie": int(counts.get("tie", 0)), "loss": abs(int(counts.get("loss", 0))), "n": total,
                                 "win_frac": float(wins / total if total else 0), "ci_lo": low, "ci_hi": high,
                                 "wilcoxon_p": record[baseline].get("per_dist", {}).get("wilcoxon_pval"),
                                 "gap_pct": float(record[baseline].get("gap_pct", 0))}
        if against:
            rows.append({"name": name, "label": match["label"] if match else name.replace("_", " ").title()[:30], "baselines": against,
                         "avg_win_rate": np.mean([s["win_frac"] for s in against.values()])})
    rows.sort(key=lambda r: -r["avg_win_rate"])
    return {r["label"]: {"name": r["name"], "avg_win_rate": r["avg_win_rate"], "baselines": r["baselines"]} for r in rows}


# Error by prediction horizon

def horizon_block(table, key, n_top, comparisons):
    """Median split RMSE by number of test runs of the split (single windows only), with Spearman's rho."""
    results = table["results"].get(key, {})
    n_splits = next((len(e["split_errors"][0]) for e in results.values() if e.get("split_errors") and len(e["split_errors"][0]) > 0), None)
    if n_splits is None:
        return {}
    window = table["window"]
    n_runs = next((r for r in range(window + 1, 50) if comb(r, window) == n_splits), None)
    if n_runs is None:
        return {}
    by_horizon = defaultdict(list)
    position = 0
    for train in combinations(range(n_runs), window):
        if max(train) + 1 < n_runs:
            by_horizon[n_runs - max(train) - 1].append(position)
            position += 1
    horizons = sorted(by_horizon)
    if len(horizons) < 2:
        return {}
    block = {}
    for name, m in compared_methods(table, key, n_top, comparisons).items():
        errors = results.get(name, {}).get("split_errors")
        if not errors:
            continue
        medians = []
        for horizon in horizons:
            values = np.array([[row[s] for s in by_horizon[horizon] if s < len(row)] for row in errors if len(row) > 0])
            medians.append(float(np.median(values.flatten())) if values.size else np.nan)
        valid = [(h, v) for h, v in zip(horizons, medians) if not np.isnan(v)]
        rho = stats.spearmanr(*zip(*valid))[0] if len(valid) >= 3 else np.nan
        block[m["label"]] = {"horizons": {str(h): None if np.isnan(v) else float(v) for h, v in zip(horizons, medians)},
                             "spearman_rho": None if np.isnan(rho) else float(rho), "is_baseline": m["is_bl"]}
    return block


# Best method per cell, 7. all methods

def sensitivity(tables, keys):
    """Method with the lowest mean RMSE per (window size, difficulty level, subset size), and how often each method is best."""
    cells = []
    for window, table in sorted(tables.items()):
        levels = np.array(table["levels"])
        for key, methods in table["results"].items():
            for level in sorted(set(table["levels"])):
                scores = sorted(((name, float(np.mean(entry["dist_means"][levels == level]))) for name, entry in methods.items()
                                 if len(entry["dist_means"])), key=lambda item: item[1])
                if scores:
                    cells.append({"window": window, "bucket": level, "pct_key": key, "best_method": scores[0][0], "best_value": scores[0][1],
                                  "runner_up": scores[1][0] if len(scores) > 1 else None, "runner_up_value": scores[1][1] if len(scores) > 1 else None})
    if not cells:
        return {}
    frequency = defaultdict(int)
    for cell in cells:
        if cell["pct_key"] in keys:
            frequency[cell["best_method"]] += 1
    return {"win_freq": {label_of(m): count for m, count in frequency.items()}, "cells": [c for c in cells if c["pct_key"] in keys]}


def all_methods_block(table, keys):
    """Statistics and per-distribution values of every evaluated method."""
    block = {}
    for key in keys:
        block[key] = {name: {"label": label_of(name), "group": family_of(name), "rmse": describe(entry["dist_means"]),
                             "maxerr": describe(entry["dist_maxerr"]) if len(entry["dist_maxerr"]) else {},
                             "dist_means": entry["dist_means"].tolist(), "dist_maxerr": entry["dist_maxerr"].tolist()}
                      for name, entry in table["results"].get(key, {}).items() if len(entry["dist_means"])}
    return block


def analysis_block(table, keys, n_top, comparisons, draws_by_window):
    """One analysis block (a window size, or all window sizes)."""
    block = {"window": str(table["window"]), "n_distributions": table["n_distributions"], "n_methods": len(table["methods"]), "subset_sizes": {}}
    for key in keys:
        cell = {"ranking": ranking_block(table, key, n_top, comparisons),
                "distributions": distributions_block(table, key, n_top, comparisons, draws_by_window),
                "cdf_tail": tail_block(table, key, n_top, comparisons, draws_by_window),
                "determinism": determinism_block(table, key, n_top, comparisons, draws_by_window)}
        if table["window"] != "overall":
            cell["temporal_gap"] = horizon_block(table, key, n_top, comparisons)
        block["subset_sizes"][key] = cell
    block["all_methods"] = all_methods_block(table, keys)
    return block
