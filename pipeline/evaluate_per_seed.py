"""Evaluation stage: per-seed MaxErr of the stochastic baselines, the determinism band and the per-seed extract.

    python pipeline/evaluate_per_seed.py --dataset multi_model [--input SELECTION_DIR] [--output RESULT_DIR]
                                         [--method embedding_strata:centroid_pooled]

A stochastic baseline gives one MaxErr per seed (the largest error over temporal splits and test runs of that
seed's subsets); a deterministic method gives one MaxErr. Written to ``--output``:

    per_seed_maxerr_all_baselines.json               median, 95th percentile and worst seed of every baseline (all W),
                                                     next to the MaxErr of ``--method``
    per_seed_maxerr_vs_consistency_stratified.json   the same against Consist-Strat only (W >= 2)
    determinism_band.json, determinism_band_maxerr.json
                                                     share of distributions in which each non-baseline method lies below,
                                                     inside or above the central 95% of Consist-Strat's 500 draws
    per_seed_maxerr_extract.npz                      W >= 2: per-distribution x per-seed MaxErr of Consist-Strat and
                                                     Diff-Strat, per-distribution MaxErr of the Embedding-Within-Strata methods

Statistics are taken per distribution inside each window size and then averaged over window sizes with equal
weight. The stage draws no random numbers.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evaluation import inputs  # noqa: E402
from evaluation.statistics import check_range, relative_change  # noqa: E402
from selection.families import within_strata_methods  # noqa: E402

BASELINES = ("random", "difficulty_stratified", "consistency_stratified", "repo_stratified", "repo_difficulty_stratified", "repo_consistency_stratified")
BAND_BASELINE = "consistency_stratified"
EXTRACT_BASELINES = ("consistency_stratified", "difficulty_stratified")
MIN_WINDOW_TWO_RUN_BASELINES = 2


def baseline_report(baseline, method, windows, tables, draws, keys):
    """Per-seed statistics of one baseline next to the deterministic method; (results, consistency checks)."""
    collected, checks = {key: [] for key in keys}, []
    for window in windows:
        results = tables[window]["results"]
        for key in keys:
            if method not in results[key] or f"baselines:{baseline}" not in results[key]:
                continue
            seeds = draws[window].get(baseline, {}).get(f"{key}__maxerr")
            if seeds is None:
                continue
            seeds = np.asarray(seeds, dtype=float)
            deterministic, stored = results[key][method]["dist_maxerr"], results[key][f"baselines:{baseline}"]
            for values, what in ((seeds, "per-seed MaxErr"), (deterministic, "deterministic MaxErr"), (stored["dist_maxerr"], "baseline MaxErr")):
                check_range(values, 0.0, 100.0, f"w{window} {key} {baseline} {what}")
            if seeds.shape[0] != deterministic.shape[0]:
                continue
            statistics = {"median": np.median(seeds, axis=1), "p95": np.percentile(seeds, 95, axis=1), "max": np.max(seeds, axis=1), "mean": np.mean(seeds, axis=1)}
            collected[key].append((statistics, deterministic, stored["dist_maxerr"], stored["dist_maxerr_mean"]))
            draws_max, family_max = float(np.mean(statistics["max"])), float(np.mean(stored["dist_maxerr"]))
            checks.append({"baseline": baseline, "window": window, "pct": key, "n_seeds": int(seeds.shape[1]), "npz_max_over_seeds_mean": draws_max,
                           "cached_dist_maxerr_mean": family_max, "rel_diff_pct": abs(draws_max - family_max) / max(family_max, 1e-9) * 100.0})
    report = {}
    for key in keys:
        if not collected[key]:
            continue
        over_windows = lambda rows: np.mean(np.vstack(rows), axis=0)   # noqa: E731
        det = over_windows([c[1] for c in collected[key]])
        seed = {s: over_windows([c[0][s] for c in collected[key]]) for s in ("median", "p95", "max", "mean")}
        stored_max, stored_mean = over_windows([c[2] for c in collected[key]]), over_windows([c[3] for c in collected[key]])
        report[key] = {
            "n_distributions": int(det.size), "det_mean": float(np.mean(det)), "det_median": float(np.median(det)),
            "baseline_seed_median_mean": float(np.mean(seed["median"])), "baseline_seed_p95_mean": float(np.mean(seed["p95"])),
            "baseline_seed_max_mean": float(np.mean(seed["max"])), "baseline_seed_mean_mean": float(np.mean(seed["mean"])),
            "baseline_seed_median_median": float(np.median(seed["median"])), "baseline_seed_p95_median": float(np.median(seed["p95"])),
            "baseline_seed_max_median": float(np.median(seed["max"])),
            "delta_pct_median_vs_seed_median": relative_change(float(np.median(seed["median"])), float(np.median(det))),
            "delta_pct_median_vs_seed_p95": relative_change(float(np.median(seed["p95"])), float(np.median(det))),
            "baseline_cached_maxerr_mean": float(np.mean(stored_max)), "baseline_cached_maxerr_mean_field": float(np.nanmean(stored_mean)),
            "delta_pct_vs_seed_median": relative_change(float(np.mean(seed["median"])), float(np.mean(det))),
            "delta_pct_vs_seed_p95": relative_change(float(np.mean(seed["p95"])), float(np.mean(det))),
            "delta_pct_vs_seed_max": relative_change(float(np.mean(seed["max"])), float(np.mean(det))),
            "delta_pct_vs_cached_max": relative_change(float(np.mean(stored_max)), float(np.mean(det))),
            "frac_dists_det_below_seed_median": float(np.mean(det < seed["median"])), "frac_dists_det_below_seed_p95": float(np.mean(det < seed["p95"]))}
    return report, checks


def per_seed_report(dataset, method, baselines, windows, tables, draws, keys):
    """Report over several baselines, in the layout read by the research-question scripts."""
    report = {"dataset": dataset, "windows": windows, "det_method": method, "baselines": list(baselines), "sanity_checks": [], "results": {}}
    for baseline in baselines:
        results, checks = baseline_report(baseline, method, windows, tables, draws, keys)
        report["sanity_checks"] += checks
        if results:
            report["results"][baseline] = results
    return report


def band_placement(values, draws):
    """Share of distributions below / inside / above the central 95% of the draws, with the band's edges."""
    low, high = np.percentile(draws, 2.5, axis=1), np.percentile(draws, 97.5, axis=1)
    n, below, above = len(values), int(np.sum(values < low)), int(np.sum(values > high))
    return {"n": n, "below": round(100.0 * below / n, 1), "inside": round(100.0 * (n - below - above) / n, 1), "above": round(100.0 * above / n, 1),
            "median": round(float(np.median(values)), 4), "band_lo_median": round(float(np.median(low)), 4),
            "band_hi_median": round(float(np.median(high)), 4), "band_lo_min": round(float(np.min(low)), 4),
            "method_max": round(float(np.max(values)), 4), "margin_min": round(float(np.min(low - values)), 4)}


def band_report(dataset, metric, windows, tables, draws, keys):
    """Determinism band of every non-baseline method, over all window sizes and over W >= 2."""
    field, suffix = ("dist_means", "") if metric == "rmse" else ("dist_maxerr", "__maxerr")
    report = {"dataset": dataset, "metric": metric, "variants": {}}
    for variant, chosen in (("all_windows", windows), ("w>=2_only", [w for w in windows if w >= MIN_WINDOW_TWO_RUN_BASELINES])):
        def averaged_draws(key, _chosen=chosen):
            used = [w for w in _chosen if w in draws and f"{key}{suffix}" in draws[w].get(BAND_BASELINE, {})]
            matrices = [draws[w][BAND_BASELINE][f"{key}{suffix}"].astype(np.float64) for w in used]
            return (np.mean([m[:min(len(x) for x in matrices)] for m in matrices], axis=0), used) if matrices else (None, [])
        if averaged_draws("10pct")[0] is None:
            continue
        entry = report["variants"][variant] = {"windows": chosen, "draw_windows": averaged_draws("10pct")[1], "results": {}}
        methods = sorted({m for w in chosen for key in keys for m in tables[w]["results"].get(key, {}) if not m.startswith("baselines:")})
        for key in keys:
            band, _ = averaged_draws(key)
            if band is None:
                continue
            for method in methods:
                used = [w for w in chosen if method in tables[w]["results"].get(key, {}) and len(tables[w]["results"][key][method][field])]
                if not used:
                    continue
                rows = [tables[w]["results"][key][method][field] for w in used]
                values = np.mean([r[:min(len(x) for x in rows)] for r in rows], axis=0)
                if len(values) == band.shape[0]:
                    entry["results"].setdefault(key, {})[method] = {**band_placement(values, band), "windows_used": used}
    return report


def per_seed_extract(windows, tables, draws, keys):
    """Arrays of the per-seed extract (W >= 2), float32."""
    arrays = {}
    for window in (w for w in windows if w >= MIN_WINDOW_TWO_RUN_BASELINES and w in draws):
        for key in keys:
            for baseline in EXTRACT_BASELINES:
                arrays[f"{key}__w{window}__{baseline}"] = draws[window][baseline][f"{key}__maxerr"].astype(np.float32)
            for method in within_strata_methods():
                if f"embedding_strata:{method}" in tables[window]["results"][key]:
                    arrays[f"{key}__w{window}__{method}"] = np.asarray(tables[window]["results"][key][f"embedding_strata:{method}"]["dist_maxerr"], dtype=np.float32)
    return arrays


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(config.DATASETS))
    parser.add_argument("--input", type=Path, default=None, help="selection results (default: data/<dataset>/selection)")
    parser.add_argument("--output", type=Path, default=None, help="default: data/<dataset>/results")
    parser.add_argument("--method", default="embedding_strata:centroid_pooled", help="deterministic method shown next to the baselines")
    arguments = parser.parse_args()
    source = arguments.input or config.stage_dir("selection", arguments.dataset)
    target = arguments.output or config.stage_dir("results", arguments.dataset)
    keys = [f"{int(f * 100)}pct" for f in config.SUBSET_SIZES]
    if config.dataset(arguments.dataset)["protocol"] == "run_groups":
        from evaluation import run_groups_per_seed
        sizes = config.dataset(arguments.dataset)["run_groups"]
        method = arguments.method.replace(":", "_")
        target.mkdir(parents=True, exist_ok=True)
        for name, baselines, min_window in (("per_seed_maxerr_all_baselines.json", run_groups_per_seed.BASELINES, 1),
                                            ("per_seed_maxerr_vs_consistency_stratified.json", (BAND_BASELINE,), MIN_WINDOW_TWO_RUN_BASELINES)):
            result = run_groups_per_seed.report(source, sizes, baselines, method, keys, min_window)
            if len(baselines) == 1:   # single-baseline file: the baseline level of the layout is dropped
                result = {"dataset": result["dataset"], "det_method": method, "baseline": baselines[0], "min_window": min_window,
                          "run_groups": result["run_groups"].get(baselines[0], {}),
                          "sanity_checks": [{k: v for k, v in check.items() if k != "baseline"} for check in result["sanity_checks"]],
                          "pooled": result["pooled"].get(baselines[0], {})}
            (target / name).write_text(json.dumps(result, indent=2))
        print(f"{arguments.dataset}: wrote the two per-seed MaxErr files to {target}")
        return
    windows = [w for w in inputs.available_windows(source) if inputs.single_file(source, f"determinism_w{w}", ".npz")]
    tables = {w: inputs.load_window(source, w, with_splits=False) for w in windows}
    draws = {w: inputs.load_draws(source, w) for w in windows}
    target.mkdir(parents=True, exist_ok=True)
    two_run_windows = [w for w in windows if w >= MIN_WINDOW_TWO_RUN_BASELINES]
    outputs = {"per_seed_maxerr_all_baselines.json": per_seed_report(arguments.dataset, arguments.method, BASELINES, windows, tables, draws, keys),
               "per_seed_maxerr_vs_consistency_stratified.json": per_seed_report(arguments.dataset, arguments.method, (BAND_BASELINE,), two_run_windows, tables, draws, keys),
               "determinism_band.json": band_report(arguments.dataset, "rmse", windows, tables, draws, keys),
               "determinism_band_maxerr.json": band_report(arguments.dataset, "maxerr", windows, tables, draws, keys)}
    for name, payload in outputs.items():
        (target / name).write_text(json.dumps(payload, indent=2))
    np.savez_compressed(target / "per_seed_maxerr_extract.npz", **per_seed_extract(windows, tables, draws, keys))
    print(f"{arguments.dataset}: windows {windows}; wrote {', '.join(outputs)} and per_seed_maxerr_extract.npz to {target}")


if __name__ == "__main__":
    main()
