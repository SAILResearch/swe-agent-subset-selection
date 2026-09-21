"""Per-seed MaxErr of the stochastic baselines for the run-groups protocol (single_setup).

For every run group, window size and subset size, a baseline has a MaxErr per temporal split and seed. The seed axis
is kept: per seed, the split values are averaged (``fold_mean``) or maximised (``fold_max``); median, 95th percentile
and worst seed are then taken, next to the same aggregate of the deterministic method's split MaxErr. The ``pooled``
block averages these statistics over all (run group, window size) cells. The key names ``fold_mean`` / ``fold_max``
refer to temporal splits.
"""
import json
from pathlib import Path

import numpy as np

from evaluation.statistics import check_range, relative_change

BASELINES = ("random", "difficulty_stratified", "consistency_stratified", "repo_stratified", "repo_difficulty_stratified",
             "repo_consistency_stratified")
AGGREGATES = ("fold_mean", "fold_max")


def method_entry(families, window, key, name):
    """Stored entry of "<family>_<method>" for one window size and subset size, or None."""
    for family, stored in families.items():
        if name.startswith(family + "_"):
            return stored["windows"].get(str(window), {}).get(key, {}).get(name[len(family) + 1:])
    return None


def baseline_report(baseline, groups, method, keys, min_window):
    """(pooled accumulators, per-run-group statistics, consistency checks) of one baseline."""
    pooled = {key: {a: {s: [] for s in ("median", "p95", "max", "mean", "det")} for a in AGGREGATES} for key in keys}
    per_group, checks = {}, []
    for n_runs, group in groups.items():
        cells = {}
        for window in sorted(w for w in group["draws"] if w >= min_window):
            for key in keys:
                own, deterministic = method_entry(group["families"], window, key, f"baselines_{baseline}"), method_entry(group["families"], window, key, method)
                if own is None or deterministic is None:
                    continue
                with np.load(group["draws"][window]) as stored:
                    if f"{key}__{baseline}__maxerr" not in stored.files:
                        continue
                    maxerr = np.asarray(stored[f"{key}__{baseline}__maxerr"], dtype=float)
                    rmse = np.asarray(stored[f"{key}__{baseline}"], dtype=float) if f"{key}__{baseline}" in stored.files else None
                check_range(maxerr, 0.0, 100.0, f"{n_runs} runs w{window} {key} {baseline} per-seed MaxErr")
                if rmse is not None:
                    check_range(rmse, 0.0, 1.0, f"{n_runs} runs w{window} {key} {baseline} per-seed RMSE")
                split_maxerr = np.asarray(deterministic.get("split_maxerr", []), dtype=float)
                if split_maxerr.size != maxerr.shape[0]:
                    continue
                per_seed = {"fold_mean": maxerr.mean(axis=0), "fold_max": maxerr.max(axis=0)}
                value = {"fold_mean": float(np.mean(split_maxerr)), "fold_max": float(np.max(split_maxerr))}
                entry = {}
                for aggregate in AGGREGATES:
                    seeds = per_seed[aggregate]
                    statistics = {"median": float(np.median(seeds)), "p95": float(np.percentile(seeds, 95)), "max": float(np.max(seeds)), "mean": float(np.mean(seeds))}
                    entry[aggregate] = {"baseline_seed_median": statistics["median"], "baseline_seed_p95": statistics["p95"],
                                        "baseline_seed_max": statistics["max"], "baseline_seed_mean": statistics["mean"], "det": value[aggregate],
                                        "delta_pct_vs_seed_median": relative_change(statistics["median"], value[aggregate]),
                                        "delta_pct_vs_seed_p95": relative_change(statistics["p95"], value[aggregate]),
                                        "delta_pct_vs_seed_max": relative_change(statistics["max"], value[aggregate])}
                    for name in ("median", "p95", "max", "mean"):
                        pooled[key][aggregate][name].append(statistics[name])
                    pooled[key][aggregate]["det"].append(value[aggregate])
                cells.setdefault(str(window), {})[key] = entry
                family_maxerr = np.asarray(own.get("split_maxerr", []), dtype=float)
                if family_maxerr.size == maxerr.shape[0]:
                    draws_max = maxerr.max(axis=1)
                    checks.append({"baseline": baseline, "n_runs": n_runs, "window": window, "pct": key, "n_folds": int(maxerr.shape[0]),
                                   "n_seeds": int(maxerr.shape[1]), "npz_fold_max_mean": float(np.mean(draws_max)),
                                   "cached_fold_maxerr_mean": float(np.mean(family_maxerr)),
                                   "mean_abs_rel_diff_pct": float(np.mean(np.abs(draws_max - family_maxerr) / np.maximum(family_maxerr, 1e-9)) * 100)})
        if cells:
            per_group[str(n_runs)] = cells
    return pooled, per_group, checks


def report(folder, run_group_sizes, baselines, method, keys, min_window):
    """Per-seed report over the given baselines, in the layout read by the research-question scripts."""
    groups = {}
    for size in run_group_sizes:
        files = sorted(p for pattern in (f"*_{size}runs.json", f"*_{size}runs_w*.json") for p in Path(folder).glob(pattern)
                       if not p.name.startswith("determinism") and not p.name.endswith(".runlog.json"))
        families = {p.name.split(f"_{size}runs")[0]: json.loads(p.read_text()) for p in files}
        draws = {int(p.stem.rsplit("_w", 1)[1]): p for p in Path(folder).glob(f"determinism_{size}runs_w*.npz")}
        if families and draws:
            groups[size] = {"families": families, "draws": draws}
    result = {"dataset": "single_setup", "det_method": method, "baselines": list(baselines), "min_window": min_window, "run_groups": {},
              "sanity_checks": [], "pooled": {}}
    for baseline in baselines:
        pooled, per_group, checks = baseline_report(baseline, groups, method, keys, min_window)
        result["sanity_checks"] += checks
        if not per_group:
            continue
        result["run_groups"][baseline] = per_group
        result["pooled"][baseline] = {}
        for aggregate in AGGREGATES:
            for key in keys:
                values = pooled[key][aggregate]
                if not values["det"]:
                    continue
                median, p95, worst, det = (float(np.mean(values[s])) for s in ("median", "p95", "max", "det"))
                result["pooled"][baseline].setdefault(key, {})[aggregate] = {
                    "n_window_group_cells": len(values["det"]), "baseline_seed_median": median, "baseline_seed_p95": p95, "baseline_seed_max": worst,
                    "baseline_seed_mean": float(np.mean(values["mean"])), "det": det, "delta_pct_vs_seed_median": relative_change(median, det),
                    "delta_pct_vs_seed_p95": relative_change(p95, det), "delta_pct_vs_seed_max": relative_change(worst, det)}
    return result
