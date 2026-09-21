"""Single-setup results across the six run groups (5 to 10 reruns per instance).

Usage (the paths in this text are relative to the package root):

    python supporting/single_setup_consistency.py

Reads ``results/single_setup/`` and writes ``supporting/outputs/single_setup_consistency/`` by default (``--results`` and ``--outputs`` change both):

    consistency_summary.md     tables per run group and subset size
    consistency_summary.csv    the same values in long format

The paper reports the 10-run group (``results/single_setup/aggregated_results.json``); the run
groups with 5 to 9 reruns are in ``results/single_setup/run_group_<N>/``. Evaluation units are
window sizes (W). Two methods are paired on the window sizes both have: Consist-Strat and
Repo x Consist need two prior runs and have no W = 1 unit. This is the pairing used by
``rq2_trajectory_aware/reproduce.py`` for the 10-run group. The script is deterministic: it draws no random numbers.
"""
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import common  # noqa: E402
from common import BASELINES, BASELINE_LABELS, STRONGEST_BASELINE, SUBSET_SIZES  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "single_setup_consistency"


def results():
    """Folder of the single_setup result files."""
    return common.results_dir() / "single_setup"


RUN_GROUPS = (5, 6, 7, 8, 9, 10)
PAPER_RUN_GROUP = 10
ALGORITHMS = ("centroid", "core_edge", "medoid", "fl", "ks")
WITHIN_STRATA_METHODS = tuple(f"embedding_strata_{a}_{v}" for a in ALGORITHMS for v in ("pooled", "ts"))
CENTROID_POOLED = "embedding_strata_centroid_pooled"


def baseline_key(baseline):
    """Stored key of a baseline."""
    return f"baselines_{baseline}"


def load_run_group(n_runs):
    """Analysis blocks (overall and one per window size) of one run group."""
    folder = results() if n_runs == PAPER_RUN_GROUP else results() / f"run_group_{n_runs}"
    return common.load_json(folder / "aggregated_results.json")["analyses"]


def run_group_info():
    """{number of reruns: stored information of the run group (instances, resolve rate per run)}."""
    stored = common.load_json(results() / "run_groups.json")["run_groups"]
    return {int(n_runs): info for n_runs, info in stored.items()}


def resolve_rate_rows(info):
    """Per run group: lowest and highest resolve rate (%) among its runs, and their difference."""
    rows = []
    for n_runs in RUN_GROUPS:
        rates = info[n_runs]["resolve_rate_pct_per_run"]
        rows.append([n_runs, len(rates), f"{min(rates):.1f}", f"{max(rates):.1f}", f"{max(rates) - min(rates):.1f}"])
    return rows


def window_units(analyses, method, subset):
    """{window size: mean RMSE of the method at that window size}."""
    units = {}
    for tag, block in analyses.items():
        if tag != "overall" and method in block["all_methods"][subset]:
            units[int(tag[1:])] = block["all_methods"][subset][method]["rmse"]["mean"]
    return units


def compare(analyses, first, second, subset):
    """Window-aligned comparison of two methods; wins, ties and losses are those of ``first``."""
    first_units, second_units = window_units(analyses, first, subset), window_units(analyses, second, subset)
    windows = sorted(set(first_units) & set(second_units))
    first_errors = np.array([first_units[w] for w in windows])
    second_errors = np.array([second_units[w] for w in windows])
    p_value, delta, record = common.paired_comparison(first_errors, second_errors)
    return {"n": len(windows), "first_mean": first_errors.mean(), "second_mean": second_errors.mean(),
            "change": 100.0 * (first_errors.mean() - second_errors.mean()) / second_errors.mean(),
            "p": p_value, "delta": delta, "record": record}


def compare_family(analyses, pairs, subset):
    """Compare every (first, second) pair and add the Holm-adjusted p over the family of pairs."""
    results = [compare(analyses, first, second, subset) for first, second in pairs]
    for result, holm_p in zip(results, common.holm_adjust([r["p"] for r in results])):
        result["holm"] = holm_p
    return results


def summarize_cell(analyses, subset):
    """All statistics of one (run group, subset size) cell."""
    overall = analyses["overall"]["all_methods"][subset]
    reference, random = baseline_key(STRONGEST_BASELINE), baseline_key("random")
    others = [b for b in BASELINES if b != STRONGEST_BASELINE]
    versus_baselines = compare_family(analyses, [(reference, baseline_key(b)) for b in others], subset)
    versus_reference = compare_family(analyses, [(m, reference) for m in WITHIN_STRATA_METHODS], subset)
    versus_random = compare_family(analyses, [(m, random) for m in WITHIN_STRATA_METHODS], subset)
    centroid_index = WITHIN_STRATA_METHODS.index(CENTROID_POOLED)
    return {"baseline_rmse": {b: overall[baseline_key(b)]["rmse"] for b in BASELINES},
            "consist_vs_baselines": dict(zip(others, versus_baselines)),
            "centroid_vs_consist": versus_reference[centroid_index],
            "centroid_vs_random": versus_random[centroid_index]}


def summarize():
    """{(run group, subset size): cell statistics} for all run groups."""
    cells = {}
    for n_runs in RUN_GROUPS:
        analyses = load_run_group(n_runs)
        for subset in SUBSET_SIZES:
            cells[(n_runs, subset)] = summarize_cell(analyses, subset)
    return cells


# Output

def record_text(record):
    """Wins/ties/losses as text."""
    return "/".join(str(count) for count in record)


def markdown_table(header, rows):
    """A markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def comparison_rows(cells, key):
    """Table rows of one two-method comparison for every run group and subset size."""
    rows = []
    for (n_runs, subset), cell in cells.items():
        c = cell[key]
        rows.append([n_runs, common.SUBSET_LABELS[subset], c["n"], f"{c['first_mean']:.4f}", f"{c['second_mean']:.4f}",
                     f"{c['change']:+.1f}", record_text(c["record"]), f"{c['p']:.3f}", f"{c['holm']:.3f}", f"{c['delta']:+.3f}"])
    return rows


def write_markdown(cells, info):
    """Write consistency_summary.md (tables only)."""
    comparison_header = ["Run group", "Subset", "Paired window sizes", "{first} mean RMSE", "{second} mean RMSE",
                         "Change of mean RMSE (%)", "W/T/L", "p", "Holm p", "Cliff's delta"]
    lines = ["# Single-setup results across run groups", "",
             "Evaluation units are window sizes (W). Methods are paired on the window sizes both have (W >= 2 when Consist-Strat "
             "is involved). W/T/L = wins/ties/losses of the first-named method across window sizes (win = lower mean RMSE). "
             "p = two-sided Wilcoxon signed-rank test; Holm p = adjusted over the five baseline pairs (Consist-Strat vs baselines) "
             "or over the ten Embedding-Within-Strata methods (Centroid Pooled comparisons). Cliff's delta = (wins - losses) / n, "
             "positive when the first-named method has the lower error. With n paired window sizes the smallest attainable "
             "two-sided p is 2 / 2^n.", "", "## Run groups", ""]
    lines += markdown_table(["Run group (reruns)", "Instances", "Window sizes", "Paired window sizes with Consist-Strat"],
                            [[n, info[n]["n_instances"], len(load_run_group(n)) - 1, cells[(n, SUBSET_SIZES[0])]["centroid_vs_consist"]["n"]]
                             for n in RUN_GROUPS])
    lines += ["", "## Resolve rate per run within each run group (%)", ""]
    lines += markdown_table(["Run group", "Runs", "Lowest", "Highest", "Range (percentage points)"], resolve_rate_rows(info))
    lines += ["", "## Baseline RMSE over window sizes (mean / median)", ""]
    lines += markdown_table(["Run group", "Subset"] + [BASELINE_LABELS[b] for b in BASELINES],
                            [[n, common.SUBSET_LABELS[k]] + [f"{cell['baseline_rmse'][b]['mean']:.4f} / {cell['baseline_rmse'][b]['median']:.4f}"
                                                              for b in BASELINES] for (n, k), cell in cells.items()])
    others = [b for b in BASELINES if b != STRONGEST_BASELINE]
    lines += ["", "## Consist-Strat vs Random", ""]
    rows = []
    for (n, k), cell in cells.items():
        c = cell["consist_vs_baselines"]["random"]
        rows.append([n, common.SUBSET_LABELS[k], c["n"], f"{c['first_mean']:.4f}", f"{c['second_mean']:.4f}", f"{c['change']:+.1f}",
                     record_text(c["record"]), f"{c['p']:.3f}", f"{c['holm']:.3f}", f"{c['delta']:+.3f}"])
    lines += markdown_table([h.format(first="Consist-Strat", second="Random") for h in comparison_header], rows)
    lines += ["", "## Consist-Strat vs each other baseline: W/T/L of Consist-Strat (Holm p)", ""]
    lines += markdown_table(["Run group", "Subset"] + [BASELINE_LABELS[b] for b in others],
                            [[n, common.SUBSET_LABELS[k]] + [f"{record_text(cell['consist_vs_baselines'][b]['record'])} "
                                                              f"({cell['consist_vs_baselines'][b]['holm']:.3f})" for b in others]
                             for (n, k), cell in cells.items()])
    for title, key, second in (("Centroid Pooled vs Consist-Strat", "centroid_vs_consist", "Consist-Strat"),
                               ("Centroid Pooled vs Random", "centroid_vs_random", "Random")):
        lines += ["", f"## {title}", ""]
        lines += markdown_table([h.format(first="Centroid Pooled", second=second) for h in comparison_header],
                                comparison_rows(cells, key))
    common.write_text(common.output_dir() / "consistency_summary.md", "\n".join(lines) + "\n")


def write_csv(cells, info):
    """Write consistency_summary.csv (long format: one row per baseline statistic, comparison or resolve-rate range).

    In the resolve-rate rows the change column holds the range in percentage points.
    """
    header = ["run_group", "subset", "instances", "table", "first_method", "second_method", "paired_window_sizes",
              "first_mean_rmse", "first_median_rmse", "second_mean_rmse", "change_of_mean_rmse_pct", "wins", "ties", "losses",
              "wilcoxon_p", "holm_p", "cliff_delta"]
    rows = []
    for (n_runs, subset), cell in cells.items():
        base = [n_runs, common.SUBSET_LABELS[subset], info[n_runs]["n_instances"]]
        for baseline in BASELINES:
            stats = cell["baseline_rmse"][baseline]
            rows.append(base + ["baseline_rmse", BASELINE_LABELS[baseline], "", stats["n"], f"{stats['mean']:.6f}",
                                f"{stats['median']:.6f}"] + [""] * 8)
        comparisons = [("consist_strat_vs_baseline", BASELINE_LABELS[STRONGEST_BASELINE], BASELINE_LABELS[b], c)
                       for b, c in cell["consist_vs_baselines"].items()]
        comparisons += [("centroid_pooled_vs_consist_strat", "Centroid Pooled", BASELINE_LABELS[STRONGEST_BASELINE], cell["centroid_vs_consist"]),
                        ("centroid_pooled_vs_random", "Centroid Pooled", BASELINE_LABELS["random"], cell["centroid_vs_random"])]
        for table, first, second, c in comparisons:
            rows.append(base + [table, first, second, c["n"], f"{c['first_mean']:.6f}", "", f"{c['second_mean']:.6f}",
                                f"{c['change']:.2f}", *c["record"], f"{c['p']:.4f}", f"{c['holm']:.4f}", f"{c['delta']:.3f}"])
    for n_runs, n_rates, lowest, highest, spread in resolve_rate_rows(info):
        rows.append([n_runs, "", info[n_runs]["n_instances"], "resolve_rate_per_run_pct", f"lowest {lowest}", f"highest {highest}", n_rates]
                    + [""] * 3 + [spread] + [""] * 6)
    common.write_csv(common.output_dir() / "consistency_summary.csv", header, rows)


def main():
    """Write all outputs of this script."""
    common.configure(__doc__.split("\n")[0], DEFAULT_OUTPUT_DIR)
    cells, info = summarize(), run_group_info()
    write_markdown(cells, info)
    write_csv(cells, info)
    print(f"single_setup_consistency: summary tables for {len(RUN_GROUPS)} run groups x {len(SUBSET_SIZES)} subset sizes "
          f"-> {common.shown(common.output_dir())}")


if __name__ == "__main__":
    main()
