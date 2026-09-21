"""RQ2: trajectory-aware embedding methods vs the strongest baseline (paper Section 6.2).

Usage (the paths in this text are relative to the package root):

    python rq2_trajectory_aware/reproduce.py

Reads ``results/`` and writes ``rq2_trajectory_aware/outputs/`` by default (``--results`` and ``--outputs`` change both):

    tables/table5_embedding_rmse.{tex,csv}        paper Table 5
    tables/table6_embedding_maxerr.{tex,csv}      paper Table 6
    tables/single_setup_embedding.{tex,csv,md}    single-setup counterpart of Tables 5 and 6
    figures/figure6_centroid_vs_consist_multi_model.pdf   paper Figure 6
    centroid_vs_strongest_baseline_by_window.csv  sensitivity to the window size
    pooled_vs_time_series.csv                     pooled vs time-series variant of each algorithm
    tables/table6_paper_vs_reproduced.csv         printed and reproduced value of every Table 6 cell
    reproduced_values.{md,csv}                    every number of Section 6.2, recomputed

The script is deterministic: it draws no random numbers.
"""
import collections
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import common  # noqa: E402
import paper_values  # noqa: E402
from common import STRONGEST_BASELINE, SUBSET_SIZES, SYNTHETIC_SCENARIOS  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
GROUP = "embedding_strata"
ALGORITHMS = ("centroid", "core_edge", "medoid", "fl", "ks")
ALGORITHM_LABELS = common.ALGORITHM_LABELS
POOLED_METHODS = tuple(f"{a}_pooled" for a in ALGORITHMS)
ALL_METHODS = tuple(f"{a}_{variant}" for a in ALGORITHMS for variant in ("pooled", "ts"))
REFERENCE_LABEL = "Consist-Strat (ref.)"
SINGLE_SETUP_RUN_GROUP = "10"
ALPHA, MINUS, DASH = common.ALPHA, common.MINUS, "–"

plt.rcParams.update({**common.PLOT_STYLE, "legend.fontsize": 6})


def method_label(method):
    """Display name such as 'Centroid Pooled' or 'KS TS'."""
    algorithm, variant = method.rsplit("_", 1)
    return f"{ALGORITHM_LABELS[algorithm]} {'Pooled' if variant == 'pooled' else 'TS'}"


# Formatting

def format_p(p_value):
    """Holm-adjusted p as printed in the paper."""
    return "<0.001" if p_value < 0.001 else f"{p_value:.2f}"


def format_delta(delta, p_value):
    """Cliff's delta with magnitude letter; a dash when the difference is not significant."""
    if p_value >= ALPHA:
        return DASH
    return f"{common.signed(delta, 3)} ({common.effect_size_letter(delta)})"


# Table 5

def stored_comparison(scenario, subset, method, window="overall"):
    """Stored paired comparison of a within-strata method against the strongest baseline."""
    return common.comparison_row(scenario, subset, "vs_best_baseline",
                                 common.method_key(scenario, GROUP, method), window)


def table5_cells(scenario, subset, method):
    """Med, relative change of mean RMSE (negative = lower error), Holm p, Cliff's delta."""
    row = stored_comparison(scenario, subset, method)
    return [f"{row['median']:.4f}", common.signed(-row["imp_rmse"], 1), format_p(row["holm_p"]),
            format_delta(row["cliff_d"], row["holm_p"])]


def table5_rows():
    """(subset, label, cells) rows of Table 5: reference row then the five pooled methods."""
    rows = []
    for subset in SUBSET_SIZES:
        reference = [cell for s in SYNTHETIC_SCENARIOS for cell in
                     [f"{common.method_stats(s, 'baselines', STRONGEST_BASELINE, subset)['rmse']['median']:.4f}",
                      DASH, DASH, DASH]]
        rows.append((subset, REFERENCE_LABEL, reference))
        for method in POOLED_METHODS:
            cells = [cell for s in SYNTHETIC_SCENARIOS for cell in table5_cells(s, subset, method)]
            rows.append((subset, method_label(method), cells))
    return rows


# Table 6

def deterministic_maxerr(arrays, subset, method):
    """Per-distribution MaxErr of a deterministic method, averaged over windows."""
    return np.mean([arrays[(subset, w, method)] for w in common.extract_windows(arrays)], axis=0)


def baseline_seed_statistics(arrays, subset):
    """Per-distribution (median, P95, worst) per-seed MaxErr of Consist-Strat."""
    return common.seed_statistics(arrays, subset, STRONGEST_BASELINE)


def maxerr_comparison(arrays, subset, method):
    """Mean MaxErr of a method and its relative change vs the baseline's median/P95/worst seed."""
    value = deterministic_maxerr(arrays, subset, method).mean()
    changes = [100.0 * (value - s.mean()) / s.mean() for s in baseline_seed_statistics(arrays, subset)]
    return value, changes


def table6_rows(extracts):
    """(subset, label, cells) rows of Table 6."""
    rows = []
    for subset, method in itertools.product(SUBSET_SIZES, POOLED_METHODS):
        cells = []
        for scenario in SYNTHETIC_SCENARIOS:
            value, changes = maxerr_comparison(extracts[scenario], subset, method)
            cells += [f"{value:.2f}"] + [common.signed(change, 1) for change in changes]
        rows.append((subset, method_label(method), cells))
    return rows


def write_table6_comparison(extracts):
    """Write the printed and the reproduced value of every Table 6 cell."""
    columns = ["MaxErr", "Delta%Med", "Delta%P95", "Delta%W"]
    rows = []
    for (subset, name, cells), printed in zip(table6_rows(extracts), paper_values.TABLE6):
        for index, (reproduced, paper) in enumerate(zip(cells, printed)):
            scenario = SYNTHETIC_SCENARIOS[index // len(columns)]
            rows.append([common.SUBSET_LABELS[subset], name, common.SCENARIO_TITLES[scenario], columns[index % len(columns)],
                         paper, reproduced])
    common.write_csv(common.output_dir() / "tables" / "table6_paper_vs_reproduced.csv",
                     ["Subset", "Method", "Dataset", "Column", "Printed value", "Reproduced value"], rows)


# Table output

def write_scenario_table(rows, stem, column_names, caption, label, rows_per_block):
    """Write rows of (subset, label, cells) as .csv and as a booktabs .tex table."""
    header = ["Subset", "Method"] + [f"{common.SCENARIO_ABBREVIATIONS[s]} {c}"
                                     for s in SYNTHETIC_SCENARIOS for c in column_names]
    common.write_csv(common.output_dir() / "tables" / f"{stem}.csv", header,
                     [[common.SUBSET_LABELS[subset], name] + cells for subset, name, cells in rows])
    width = len(column_names)
    groups = " & ".join(f"\\multicolumn{{{width}}}{{c}}{{{common.SCENARIO_TITLES[s]}}}" for s in SYNTHETIC_SCENARIOS)
    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{ll{'r' * width * len(SYNTHETIC_SCENARIOS)}}}", "\\toprule",
             f"Subset & Method & {groups} \\\\",
             " &  & " + " & ".join(column_names * len(SYNTHETIC_SCENARIOS)) + " \\\\", "\\midrule"]
    for index, (subset, name, cells) in enumerate(rows):
        if index and index % rows_per_block == 0:
            lines.append("\\midrule")
        first = common.tex_escape(common.SUBSET_LABELS[subset]) if index % rows_per_block == 0 else ""
        tex_cells = [c.replace(MINUS, "$-$").replace(DASH, "--").replace("<", "$<$") for c in cells]
        lines.append(f"{first} & {name} & " + " & ".join(tex_cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    common.write_text(common.output_dir() / "tables" / f"{stem}.tex", "\n".join(lines))


# Single-setup table

def single_setup_units(method_group, method, subset):
    """{window size: mean RMSE} of a method for the single-setup 10-run group."""
    units = {}
    for window in common.window_sizes("single_setup"):
        stats = common.method_stats("single_setup", method_group, method, subset, window)
        if stats is not None:
            units[window] = stats["rmse"]["mean"]
    return units


def single_setup_rmse_row(subset, method, reference_units):
    """Window-aligned paired statistics of one method vs Consist-Strat (raw p, no Holm yet)."""
    units = single_setup_units(GROUP, method, subset)
    windows = sorted(set(units) & set(reference_units))
    method_errors = np.array([units[w] for w in windows])
    reference_errors = np.array([reference_units[w] for w in windows])
    p_value, delta, record = common.paired_comparison(method_errors, reference_errors)
    change = 100.0 * (method_errors.mean() - reference_errors.mean()) / reference_errors.mean()
    median = common.method_stats("single_setup", GROUP, method, subset)["rmse"]["median"]
    return {"method": method, "median": median, "change": change,
            "p": p_value, "delta": delta, "record": record}


def single_setup_maxerr(subset, method):
    """Mean MaxErr over windows W >= 2 and its change vs Consist-Strat's median/P95/worst seed.

    MaxErr per window is the mean over temporal splits (the aggregation of Table 4's
    single-setup columns); Consist-Strat's per-seed statistics come from the 10-run group
    of the stored per-seed file.
    """
    stored = common.load_json(common.results_dir() / "single_setup" / "per_seed_maxerr_vs_consistency_stratified.json")
    cells = stored["run_groups"][SINGLE_SETUP_RUN_GROUP]
    key = common.SINGLE_SETUP_SPLIT_AGGREGATION_KEY
    windows = sorted(int(w) for w in cells)
    value = np.mean([common.method_stats("single_setup", GROUP, method, subset, w)["maxerr"]["mean"] for w in windows])
    seeds = [np.mean([cells[str(w)][subset][key][name] for w in windows])
             for name in ("baseline_seed_median", "baseline_seed_p95", "baseline_seed_max")]
    return value, [100.0 * (value - s) / s for s in seeds], seeds


def single_setup_rows():
    """Rows of the single-setup table, with Holm over the ten within-strata methods per subset."""
    rows = []
    for subset in SUBSET_SIZES:
        reference_units = single_setup_units("baselines", STRONGEST_BASELINE, subset)
        raw = {m: single_setup_rmse_row(subset, m, reference_units) for m in ALL_METHODS}
        adjusted = common.holm_adjust([raw[m]["p"] for m in ALL_METHODS])
        for method, holm_p in zip(ALL_METHODS, adjusted):
            if method in POOLED_METHODS:
                value, changes, _ = single_setup_maxerr(subset, method)
                rows.append(dict(raw[method], subset=subset, holm=holm_p, maxerr=value, maxerr_changes=changes))
    return rows


def write_single_setup_table():
    """Write the single-setup counterpart of Tables 5 and 6 (10-run group) and return its rows."""
    rows = single_setup_rows()
    header = ["Subset", "Method", "Med", "Delta%R", "p (Wilcoxon)", "p (Holm, 10 methods)", "Cliff delta",
              "W/T/L (n=8)", "MaxErr", "Delta%Med", "Delta%P95", "Delta%W"]
    table = []
    for subset in SUBSET_SIZES:
        reference = common.method_stats("single_setup", "baselines", STRONGEST_BASELINE, subset)["rmse"]["median"]
        _, _, seeds = single_setup_maxerr(subset, POOLED_METHODS[0])
        table.append([common.SUBSET_LABELS[subset], REFERENCE_LABEL, f"{reference:.4f}", DASH, DASH, DASH, DASH, DASH,
                      "/".join(f"{s:.2f}" for s in seeds) + " (median/P95/worst seed)", DASH, DASH, DASH])
        for row in (r for r in rows if r["subset"] == subset):
            wins, ties, losses = row["record"]
            table.append([common.SUBSET_LABELS[subset], method_label(row["method"]), f"{row['median']:.4f}",
                          common.signed(row["change"], 1), f"{row['p']:.3f}", f"{row['holm']:.3f}",
                          f"{common.signed(row['delta'], 3)} ({common.effect_size_letter(row['delta'])})",
                          f"{wins}/{ties}/{losses}", f"{row['maxerr']:.2f}"]
                         + [common.signed(c, 1) for c in row["maxerr_changes"]])
    common.write_csv(common.output_dir() / "tables" / "single_setup_embedding.csv", header, table)
    markdown = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    markdown += ["| " + " | ".join(line) + " |" for line in table]
    common.write_text(common.output_dir() / "tables" / "single_setup_embedding.md", "\n".join(markdown) + "\n")
    tex = ["\\begin{tabular}{ll" + "r" * 10 + "}", "\\toprule", " & ".join(header).replace("%", "\\%") + " \\\\", "\\midrule"]
    tex += [" & ".join(c.replace(MINUS, "$-$").replace(DASH, "--").replace("%", "\\%") for c in line) + " \\\\" for line in table]
    common.write_text(common.output_dir() / "tables" / "single_setup_embedding.tex", "\n".join(tex + ["\\bottomrule", "\\end{tabular}", ""]))
    return rows


# Figure 6

def rmse_difference(scenario, subset):
    """Per-distribution RMSE of Consist-Strat minus Centroid Pooled (positive = ours is better)."""
    baseline = np.asarray(common.method_stats(scenario, "baselines", STRONGEST_BASELINE, subset)["dist_means"])
    ours = np.asarray(common.method_stats(scenario, GROUP, "centroid_pooled", subset)["dist_means"])
    return baseline - ours


def published_centroid_maxerr(scenario, subset):
    """Centroid Pooled per-distribution MaxErr as drawn in the published figure.

    The published figure takes the average over all window sizes (W >= 1), while the
    Consist-Strat curves and Table 6 use W >= 2.
    """
    return np.asarray(common.method_stats(scenario, GROUP, "centroid_pooled", subset)["dist_maxerr"])


def draw_difference_panel(axis, differences):
    """Violin and box plot of the paired RMSE differences with the break-even line."""
    axis.axhline(0, color="#d55e00", ls="--", lw=1.2, zorder=1)
    violin = axis.violinplot([differences], positions=[0.5], showmedians=False, showextrema=False, widths=0.5)
    violin["bodies"][0].set_facecolor("#56B4E9")
    violin["bodies"][0].set_alpha(0.6)
    line = dict(color="black", lw=1.2)
    axis.boxplot([differences], positions=[0.5], widths=0.3, patch_artist=True, showfliers=False,
                 boxprops=dict(facecolor="white", **line), whiskerprops=line, capprops=line,
                 medianprops=dict(color="black", lw=1.5))
    axis.set_xlim(-0.5, 1.5)
    axis.set_xticks([])


def draw_density_panel(axis, grid, series):
    """Overlaid kernel densities of the three Consist-Strat seed statistics and Centroid Pooled."""
    styles = (("CS median seed", "#636363", "#969696", "-", 1.2, 0.25),
              ("CS P95 seed", "#E69F00", "#E69F00", "--", 1.2, 0.20),
              ("CS worst seed", "#D55E00", "#D55E00", ":", 1.2, 0.15),
              ("Centroid Pooled", "#009E73", "#009E73", "-", 1.8, 0.30))
    for values, (name, color, fill, style, width, alpha) in zip(series, styles):
        density = gaussian_kde(values, bw_method="scott")(grid)
        axis.fill_between(grid, density, alpha=alpha, color=fill)
        axis.plot(grid, density, color=color, lw=width, ls=style, label=name)
    axis.set_yticks([])
    axis.set_xlim(grid[0], grid[-1])


def write_figure6(extracts, scenario="multi_model"):
    """Figure 6: paired RMSE difference (top) and MaxErr densities (bottom), multi-model."""
    differences = {k: rmse_difference(scenario, k) for k in SUBSET_SIZES}
    series = {k: list(baseline_seed_statistics(extracts[scenario], k)) + [published_centroid_maxerr(scenario, k)]
              for k in SUBSET_SIZES}
    everything = np.concatenate([v for values in series.values() for v in values])
    low, high = np.percentile(everything, 0.2), np.percentile(everything, 99.8)
    grid = np.linspace(low - 0.05 * (high - low), high + 0.05 * (high - low), 500)
    y_low = min(np.percentile(d, 0.5) for d in differences.values())
    y_high = max(np.percentile(d, 99.5) for d in differences.values())
    margin = 0.08 * (y_high - y_low)
    figure, axes = plt.subplots(2, len(SUBSET_SIZES), figsize=(7.5, 4.5))
    for column, subset in enumerate(SUBSET_SIZES):
        draw_difference_panel(axes[0, column], differences[subset])
        axes[0, column].set_ylim(y_low - margin, y_high + margin)
        axes[0, column].set_title(("Subset: " if column == 0 else "") + common.SUBSET_LABELS[subset])
        draw_density_panel(axes[1, column], grid, series[subset])
    axes[0, 0].set_ylabel("$\\Delta$ RMSE (Consist-Strat $-$ Ours)")
    axes[1, 0].set_xlabel("Max Error (pp)")
    axes[1, 0].set_ylabel("Density")
    figure.tight_layout()
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    path = common.output_dir() / "figures" / f"figure6_centroid_vs_consist_{scenario}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


# Sensitivity tables

def window_comparison_rows():
    """Centroid Pooled vs the strongest baseline available at each window size, per scenario and subset."""
    rows = []
    for scenario in SYNTHETIC_SCENARIOS:
        for window, subset in itertools.product(common.window_sizes(scenario), SUBSET_SIZES):
            block = common.analysis(scenario, window)
            reference_key = block["subset_sizes"][subset]["ranking"]["best_baseline_used"]
            reference = block["all_methods"][subset][reference_key]
            ours = common.method_stats(scenario, GROUP, "centroid_pooled", subset, window)
            _, _, record = common.paired_comparison(np.asarray(ours["dist_means"]), np.asarray(reference["dist_means"]))
            rows.append({"scenario": scenario, "window": window, "subset": subset, "baseline": reference["label"],
                         "ours_mean": ours["rmse"]["mean"], "ours_median": ours["rmse"]["median"],
                         "baseline_mean": reference["rmse"]["mean"], "baseline_median": reference["rmse"]["median"],
                         "record": record})
    return rows


def variant_comparison_rows():
    """Pooled vs time-series variant of each algorithm, overall, per scenario and subset."""
    rows = []
    for scenario, algorithm, subset in itertools.product(SYNTHETIC_SCENARIOS, ALGORITHMS, SUBSET_SIZES):
        pooled = common.method_stats(scenario, GROUP, f"{algorithm}_pooled", subset)["rmse"]
        series = common.method_stats(scenario, GROUP, f"{algorithm}_ts", subset)["rmse"]
        rows.append({"scenario": scenario, "algorithm": ALGORITHM_LABELS[algorithm], "subset": subset,
                     "pooled_mean": pooled["mean"], "ts_mean": series["mean"],
                     "pooled_median": pooled["median"], "ts_median": series["median"]})
    return rows


def lower_of(first, second, first_name, second_name):
    """Name of the smaller value ('tie' when equal at four decimals)."""
    if round(first, 4) == round(second, 4):
        return "tie"
    return first_name if first < second else second_name


def write_sensitivity_tables(window_rows, variant_rows):
    """Write the two sensitivity tables as CSV."""
    common.write_csv(common.output_dir() / "centroid_vs_strongest_baseline_by_window.csv",
                     ["Scenario", "W", "Subset", "Strongest baseline", "Centroid mean RMSE", "Baseline mean RMSE", "Mean diff %",
                      "Centroid median RMSE", "Baseline median RMSE", "Median diff %", "Centroid W/T/L over distributions"],
                     [[r["scenario"], r["window"], common.SUBSET_LABELS[r["subset"]], r["baseline"],
                       f"{r['ours_mean']:.4f}", f"{r['baseline_mean']:.4f}", f"{100 * (r['ours_mean'] - r['baseline_mean']) / r['baseline_mean']:+.1f}",
                       f"{r['ours_median']:.4f}", f"{r['baseline_median']:.4f}", f"{100 * (r['ours_median'] - r['baseline_median']) / r['baseline_median']:+.1f}",
                       "/".join(str(v) for v in r["record"])] for r in window_rows])
    common.write_csv(common.output_dir() / "pooled_vs_time_series.csv",
                     ["Scenario", "Algorithm", "Subset", "Pooled mean RMSE", "TS mean RMSE", "Lower mean",
                      "Pooled median RMSE", "TS median RMSE", "Lower median"],
                     [[r["scenario"], r["algorithm"], common.SUBSET_LABELS[r["subset"]], f"{r['pooled_mean']:.4f}", f"{r['ts_mean']:.4f}",
                       lower_of(r["pooled_mean"], r["ts_mean"], "pooled", "time-series"), f"{r['pooled_median']:.4f}", f"{r['ts_median']:.4f}",
                       lower_of(r["pooled_median"], r["ts_median"], "pooled", "time-series")] for r in variant_rows])


def values_window_sensitivity(values, window_rows):
    """Sensitivity to the window size: Centroid Pooled vs the strongest baseline available at each W."""
    where = "Sensitivity: window"
    for scenario in SYNTHETIC_SCENARIOS:
        rows = [r for r in window_rows if r["scenario"] == scenario]
        for statistic in ("mean", "median"):
            losing = [f"W={r['window']} {r['subset']} ({r['ours_' + statistic]:.4f} vs {r['baseline_' + statistic]:.4f})"
                      for r in rows if r["ours_" + statistic] >= r["baseline_" + statistic]]
            values.statement(where, f"Centroid Pooled has lower RMSE than the strongest baseline available at every window size, {scenario}, by {statistic} RMSE",
                         "every W", not losing, f"lower in {len(rows) - len(losing)} of {len(rows)} (W, subset) cells"
                         + (f"; exceptions: {'; '.join(losing)}" if losing else ""),
                         "rq2_trajectory_aware/outputs/centroid_vs_strongest_baseline_by_window.csv")


def values_variant_comparison(values, variant_rows):
    """Pooled vs time-series variant of each algorithm (Table 5 caption, Section 7.2)."""
    counts = {}
    for statistic in ("mean", "median"):
        outcome = collections.Counter(lower_of(r["pooled_" + statistic], r["ts_" + statistic], "pooled", "time-series") for r in variant_rows)
        counts[statistic] = outcome
    text = "; ".join(f"by {k} RMSE: pooled lower in {v['pooled']}, tie in {v['tie']}, time-series lower in {v['time-series']} of {len(variant_rows)} cells"
                     for k, v in counts.items())
    for statistic, outcome in counts.items():
        values.statement("Section 7.2 (implications)",
                     f"Pooled embeddings match or outperform time-series embeddings in most configurations, by {statistic} RMSE", "most",
                     outcome["pooled"] + outcome["tie"] > len(variant_rows) / 2,
                     f"pooled lower in {outcome['pooled']}, tie in {outcome['tie']}, time-series lower in {outcome['time-series']} "
                     f"of {len(variant_rows)} cells", "rq2_trajectory_aware/outputs/pooled_vs_time_series.csv")
    exceptions = [f"{r['scenario']} {r['subset']} {r['algorithm']}" for r in variant_rows
                  if lower_of(r["pooled_mean"], r["ts_mean"], "pooled", "time-series") == "time-series"]
    values.statement("Table 5 caption", "The pooled variant of each algorithm matches or outperforms the time-series variant",
                     "pooled matches or outperforms", not exceptions,
                     text + ". Time-series lower by mean RMSE in: " + ("; ".join(exceptions) or "no cell"),
                     "rq2_trajectory_aware/outputs/pooled_vs_time_series.csv")


# Reproduced values

def values_tables(values, extracts):
    """Cell-by-cell comparison of Tables 5 and 6 with the printed values."""
    rows5 = table5_rows()
    values.table("Table 5", [cells for _, _, cells in rows5], paper_values.TABLE5,
                 row_names=[f"{common.SUBSET_LABELS[k]} {name}" for k, name, _ in rows5],
                 column_names=[f"{common.SCENARIO_TITLES[s]} {c}" for s in SYNTHETIC_SCENARIOS for c in ("Med", "Delta%R", "p", "delta")])
    rows6 = table6_rows(extracts)
    centroid = [i for i, (_, name, _) in enumerate(rows6) if name == method_label("centroid_pooled")]
    others = [i for i in range(len(rows6)) if i not in centroid]
    maxerr_columns = (0, 4)  # the MaxErr cell of each dataset; the other cells are relative changes
    gap, index, column = max((abs(float(rows6[i][2][c]) - float(paper_values.TABLE6[i][c])), i, c) for i in others for c in maxerr_columns)
    largest = (f"; largest absolute difference in MaxErr {gap:.2f} percentage points ({common.SCENARIO_TITLES[SYNTHETIC_SCENARIOS[column // 4]]}, "
               f"{common.SUBSET_LABELS[rows6[index][0]]}, {rows6[index][1]}: printed {paper_values.TABLE6[index][column]}, reproduced "
               f"{rows6[index][2][column]}); every cell is listed in rq2_trajectory_aware/outputs/tables/table6_paper_vs_reproduced.csv; see the README "
               f"section 'Differences from the printed tables'")
    for label, selection, note in (("Centroid Pooled rows", centroid, ""),
                                   ("Core/Edge Pooled, Medoid Pooled, FL Pooled and KS Pooled rows", others, largest)):
        values.table("Table 6", [rows6[i][2] for i in selection], [paper_values.TABLE6[i] for i in selection],
                     row_names=[f"{common.SUBSET_LABELS[rows6[i][0]]} {rows6[i][1]}" for i in selection],
                     rows_label=label, note=note, describe_gaps=not note)


def values_table6_conclusions(values, extracts):
    """Conclusions drawn from Table 6, checked under the computed values."""
    where = "Table 6 conclusions"
    lowest = {}
    for scenario, subset in itertools.product(SYNTHETIC_SCENARIOS, SUBSET_SIZES):
        lowest[(scenario, subset)] = min(POOLED_METHODS, key=lambda m: maxerr_comparison(extracts[scenario], subset, m)[0])
    small = {cell: m for cell, m in lowest.items() if cell[1] in ("5pct", "10pct")}
    values.statement(where, "Centroid Pooled or Core/Edge Pooled has the lowest MaxErr at 5% and 10%", "4 cells",
                 all(m in ("centroid_pooled", "core_edge_pooled") for m in small.values()),
                 "; ".join(f"{s} {k}: {method_label(m)}" for (s, k), m in lowest.items()), "per_seed_maxerr_extract.npz")
    changes = [maxerr_comparison(extracts[s], k, m)[1] for s in SYNTHETIC_SCENARIOS for k in SUBSET_SIZES for m in POOLED_METHODS]
    values.statement(where, "Every method at every size is well below Consist-Strat's P95 and worst draw", "all 40 cells",
                 max(c[1] for c in changes) < 0 and max(c[2] for c in changes) < 0,
                 f"change vs P95 seed from {min(c[1] for c in changes):+.1f}% to {max(c[1] for c in changes):+.1f}%; "
                 f"vs worst seed from {min(c[2] for c in changes):+.1f}% to {max(c[2] for c in changes):+.1f}%; "
                 f"vs median seed from {min(c[0] for c in changes):+.1f}% to {max(c[0] for c in changes):+.1f}%", "same")


def values_motivation_and_headline(values, extracts):
    """Motivation of RQ2 and the headline ranges of the abstract and Finding 2.2."""
    where = "Motivation / headline"
    typical = [common.per_seed_maxerr(s, STRONGEST_BASELINE, "5pct")[0] for s in SYNTHETIC_SCENARIOS]
    values.number_range(where, "Best baseline worst-case error at 5%: about 14% on a typical draw", 14, 14, typical,
                        "per-seed median, multi-model and multi-agent")
    values.number(where, "... and up to 35% on its worst draw", 35,
                  max(common.per_seed_maxerr(s, STRONGEST_BASELINE, "5pct")[2] for s in SYNTHETIC_SCENARIOS), 0, "per-seed worst")
    rmse, typical_draw, p95_draw = [], [], []
    for scenario, subset in itertools.product(SYNTHETIC_SCENARIOS, ("5pct", "10pct")):
        rmse.append(stored_comparison(scenario, subset, "centroid_pooled")["imp_rmse"])
        _, changes = maxerr_comparison(extracts[scenario], subset, "centroid_pooled")
        typical_draw.append(-changes[0])
        p95_draw.append(-changes[1])
    for place in ("Abstract / RQ2 headline", "Finding 2.2 and Summary box"):
        if place.startswith("Abstract"):
            values.number_range(place, "Centroid Pooled reduces mean RMSE at 5% and 10% by (%)", 3, 11, rmse,
                                "stored imp_rmse, both datasets")
        values.number_range(place, "MaxErr reduction vs the typical (median-seed) draw at 5% and 10% (%)", 4, 11,
                            typical_draw, "extract: 100*(CS median seed - Centroid)/CS median seed")
        values.number_range(place, "MaxErr reduction vs the 95th-percentile draw at 5% and 10% (%)", 38, 46,
                            p95_draw, "extract: same with the P95 seed")


def values_finding_2_1(values):
    """Finding 2.1: RMSE of the trajectory-aware methods vs Consist-Strat."""
    where = "Finding 2.1"
    best = max(stored_comparison(s, k, m)["imp_rmse"] for s in SYNTHETIC_SCENARIOS for k in SUBSET_SIZES for m in POOLED_METHODS)
    values.number(where, "Mean RMSE reduced by up to (%)", 11.4, best, 1, "max stored imp_rmse over Table 5")
    row = stored_comparison("multi_model", "10pct", "centroid_pooled")
    values.number(where, "Multi-model 10% Centroid Pooled: Cliff's delta", 0.492, row["cliff_d"], 3, "stored cliff_d")
    values.statement(where, "... p < 0.001, large effect", "p<0.001; large", row["holm_p"] < 0.001 and row["cliff_mag"] == "large",
                 f"Holm p={row['holm_p']:.2g}, {row['cliff_mag']}", "stored holm_p, cliff_mag")
    failing = [f"{s} {k} {m}" for s in SYNTHETIC_SCENARIOS for k in ("5pct", "10pct") for m in ("centroid_pooled", "core_edge_pooled")
               if not (stored_comparison(s, k, m)["holm_p"] < 0.001 and stored_comparison(s, k, m)["imp_rmse"] > 0)]
    values.statement(where, "Centroid and Core/Edge Pooled significantly outperform Consist-Strat at 5% and 10% in both datasets (p<0.001)",
                 "8 cells", not failing, "exceptions: " + (", ".join(failing) or "none"), "stored holm_p and imp_rmse")
    failing = [f"{k} {m}" for k in ("20pct", "30pct") for m in ("centroid_pooled", "core_edge_pooled")
               if not (stored_comparison("multi_model", k, m)["holm_p"] < ALPHA and stored_comparison("multi_model", k, m)["imp_rmse"] > 0)]
    values.statement(where, "At 20-30% the advantage holds in multi-model", "holds", not failing,
                 "exceptions: " + (", ".join(failing) or "none"), "stored rows")
    centroid = [stored_comparison("multi_agent", k, "centroid_pooled") for k in ("20pct", "30pct")]
    on_par = all(r["holm_p"] >= ALPHA or r["cliff_mag"] == "negligible" for r in centroid)
    values.statement(where, "At 20-30% in multi-agent Centroid Pooled is on par with Consist-Strat (negligible delta)", "on par",
                 on_par, "; ".join(f"Holm p={r['holm_p']:.2g}, delta={r['cliff_d']:+.3f} ({r['cliff_mag']})" for r in centroid), "stored rows")
    core_edge = [stored_comparison("multi_agent", k, "core_edge_pooled") for k in ("20pct", "30pct")]
    values.statement(where, "... and Core/Edge Pooled falls behind it", "behind",
                 all(r["imp_rmse"] < 0 and r["holm_p"] < ALPHA for r in core_edge),
                 "; ".join(f"change {-r['imp_rmse']:+.1f}%, Holm p={r['holm_p']:.2g}" for r in core_edge), "stored rows")
    ranks = []
    for scenario, subset in itertools.product(SYNTHETIC_SCENARIOS, ("5pct", "10pct")):
        ordered = sorted(ALL_METHODS, key=lambda m: common.method_stats(scenario, GROUP, m, subset)["rmse"]["mean"])
        ranks.append((scenario, subset, ordered.index("centroid_pooled") + 1, ordered.index("core_edge_pooled") + 1))
    values.statement(where, "Centroid Pooled and Core/Edge Pooled consistently rank among the best at 5% and 10%", "top ranks",
                 all(max(a, b) <= 2 for _, _, a, b in ranks),
                 "; ".join(f"{s} {k}: ranks {a} and {b} of 10 by mean RMSE" for s, k, a, b in ranks), "all_methods rmse.mean")
    for subset in ("5pct", "10pct"):
        row = stored_comparison("multi_model", subset, "centroid_pooled")
        values.statement(where, f"Centroid Pooled has lower RMSE than Consist-Strat on the large majority of distributions, {subset}",
                     "large majority", row["wins"] > 600, f"W/T/L = {row['wins']}/{row['ties']}/{row['losses']}", "stored wins/ties/losses")


def values_figure6(values):
    """Figure 6 caption."""
    where = "Fig. 6 caption"
    details, ok = [], True
    for subset in SUBSET_SIZES:
        q25, q50, q75 = np.percentile(rmse_difference("multi_model", subset), [25, 50, 75])
        share_positive = (q75 - max(q25, 0)) / (q75 - q25) if q75 > 0 else 0.0
        ok = ok and share_positive > 0.5
        details.append(f"{subset}: quartiles {q25:+.4f}/{q50:+.4f}/{q75:+.4f}, {100 * share_positive:.0f}% of the IQR above 0")
    values.statement(where, "The majority of the interquartile range sits above the break-even line", "all four sizes", ok,
                 "; ".join(details), "quartiles of dist_means(Consist-Strat) - dist_means(Centroid Pooled)")
    p_values = [stored_comparison("multi_model", k, "centroid_pooled")["holm_p"] for k in SUBSET_SIZES]
    values.statement(where, "p < 0.001 across all subset sizes", "p<0.001", max(p_values) < 0.001,
                 ", ".join(f"{p:.2g}" for p in p_values), "stored holm_p")


def values_finding_2_2(values, extracts):
    """Finding 2.2: MaxErr of the trajectory-aware methods vs Consist-Strat's draws."""
    where = "Finding 2.2"
    for scenario in SYNTHETIC_SCENARIOS:
        median_seed, p95_seed, _ = (s.mean() for s in baseline_seed_statistics(extracts[scenario], "10pct"))
        values.number(where, f"At 10% the baseline's typical draw is about 10 pp, {scenario}", 10, median_seed, 0, "extract, CS median seed")
        values.statement(where, f"... with the 95th-percentile draw reaching 15-16 pp, {scenario}", "15-16",
                     15 <= round(p95_seed) <= 16, f"{p95_seed:.2f}", "extract, CS P95 seed")
        values.number(where, f"... whereas Centroid Pooled limits MaxErr to roughly 9%, {scenario}", 9,
                      deterministic_maxerr(extracts[scenario], "10pct", "centroid_pooled").mean(), 0, "extract")
    for method, paper in (("medoid_pooled", 1.4), ("fl_pooled", 1.3)):
        row = stored_comparison("multi_model", "5pct", method)
        values.number(where, f"Multi-model 5% {method_label(method)}: RMSE change vs Consist-Strat (%)", paper, -row["imp_rmse"], 1, "stored imp_rmse")
        values.statement(where, f"... not statistically significant ({method_label(method)})", "n.s.", row["holm_p"] >= ALPHA,
                     f"Holm p={row['holm_p']:.2f}", "stored holm_p")


def values_sensitivity(values, extracts):
    """Sensitivity to the difficulty level, the subset size and the dataset."""
    where = "Sensitivity: difficulty levels"
    for scenario in SYNTHETIC_SCENARIOS:
        cells = common.load_aggregated(scenario)["sensitivity"]["cells"]
        summary = {}
        for name, (low, high) in (("10-30%", (10, 30)), ("40-60%", (40, 60)), ("70-90%", (70, 90))):
            winners = collections.Counter(c["best_method"].split(":")[-1] for c in cells if low <= c["bucket"] <= high)
            summary[name] = winners
        text = "; ".join(f"{name}: " + ", ".join(f"{m} {n}" for m, n in winners.most_common(3)) for name, winners in summary.items())
        values.statement(where, f"Centroid Pooled achieves the lowest RMSE in the middle range (40-60%), {scenario}", "most frequent winner",
                     summary["40-60%"].most_common(1)[:1] == [("centroid_pooled", summary["40-60%"]["centroid_pooled"])], text,
                     "sensitivity.cells best_method counts")
        core_edge_low = summary["10-30%"]["core_edge_pooled"]
        more_often = core_edge_low > max(summary["40-60%"]["core_edge_pooled"], summary["70-90%"]["core_edge_pooled"])
        values.statement(where, f"Core/Edge Pooled achieves the lowest RMSE more often at the lowest levels (10-30%), {scenario}",
                         "more often", more_often, text, "sensitivity.cells best_method counts")
    where = "Sensitivity: sizes and datasets"
    values.number(where, "Multi-agent 30%: Consist-Strat median RMSE", 0.0208,
                  common.method_stats("multi_agent", "baselines", STRONGEST_BASELINE, "30pct")["rmse"]["median"], 4, "rmse.median")
    values.number(where, "Multi-agent 30%: Centroid Pooled median RMSE", 0.0216,
                  common.method_stats("multi_agent", GROUP, "centroid_pooled", "30pct")["rmse"]["median"], 4, "rmse.median")
    values.number(where, "At 30% Centroid Pooled reduces MaxErr in multi-model by (%)", 9,
                  -maxerr_comparison(extracts["multi_model"], "30pct", "centroid_pooled")[1][0], 0, "extract, vs median seed")
    change = maxerr_comparison(extracts["multi_agent"], "30pct", "centroid_pooled")[1][0]
    values.statement(where, "... but shows no improvement in multi-agent relative to the typical draw", "no improvement", change >= 0,
                 f"{change:+.1f}%", "extract, vs median seed")
    gaps = {k: np.mean([stored_comparison(s, k, "centroid_pooled")["imp_rmse"] for s in SYNTHETIC_SCENARIOS]) for k in SUBSET_SIZES}
    values.statement(where, "Improvement is greatest at 5% and 10% and narrows at 20% and 30%", "narrows",
                 min(gaps["5pct"], gaps["10pct"]) > max(gaps["20pct"], gaps["30pct"]),
                 ", ".join(f"{k}: {v:+.1f}%" for k, v in gaps.items()) + " (Centroid Pooled, mean of both datasets)", "stored imp_rmse")


def values_single_setup(values, rows):
    """Single-setup counterpart referred to in the Table 5 caption (10-run group)."""
    where = "Table 5 caption"
    centroid = [r for r in rows if r["method"] == "centroid_pooled"]
    text = "; ".join(f"{r['subset']}: RMSE {r['change']:+.1f}% (W/T/L {r['record'][0]}/{r['record'][1]}/{r['record'][2]}, Holm p={r['holm']:.2f}), "
                     f"MaxErr vs median seed {r['maxerr_changes'][0]:+.1f}%" for r in centroid)
    consistent = all(r["change"] < 0 for r in centroid if r["subset"] in ("5pct", "10pct"))
    values.statement(where, "Single-setup results are consistent (with multi-model and multi-agent)", "consistent", consistent,
                 "Centroid Pooled vs Consist-Strat, 10-run group, window-aligned: " + text,
                 "rq2_trajectory_aware/outputs/tables/single_setup_embedding.csv")

def collect_values():
    """All reproduced values of Section 6.2."""
    extracts = {scenario: common.load_extract(scenario) for scenario in SYNTHETIC_SCENARIOS}
    values = common.ReproducedValues("Section 6.2 (RQ2): values stated in the paper and reproduced from `results/`")
    values_tables(values, extracts)
    values_table6_conclusions(values, extracts)
    values_motivation_and_headline(values, extracts)
    values_finding_2_1(values)
    values_figure6(values)
    values_finding_2_2(values, extracts)
    values_window_sensitivity(values, window_comparison_rows())
    values_variant_comparison(values, variant_comparison_rows())
    values_sensitivity(values, extracts)
    values_single_setup(values, single_setup_rows())
    return values


def main():
    """Write all outputs of this script."""
    common.configure(__doc__.split("\n")[0], DEFAULT_OUTPUT_DIR)
    extracts = {scenario: common.load_extract(scenario) for scenario in SYNTHETIC_SCENARIOS}
    write_scenario_table(table5_rows(), "table5_embedding_rmse", ["Med", "$\\Delta$\\%R", "$p$", "$\\delta$"],
                         "RQ2: RMSE of Embedding-Within-Strata methods vs. the strongest baseline (Consist-Strat), shown as a reference row. "
                         "We show the pooled variant of each algorithm. Med = median RMSE. $\\Delta$\\%R = relative change in mean RMSE vs. "
                         "Consist-Strat (negative = lower error). $p$ = Holm-adjusted Wilcoxon. $\\delta$ = Cliff's delta. "
                         "MaxErr is reported separately in Table 6.",
                         "tab:rq2-rmse", rows_per_block=6)
    write_scenario_table(table6_rows(extracts), "table6_embedding_maxerr",
                         ["MaxErr", "$\\Delta$\\%Med", "$\\Delta$\\%P95", "$\\Delta$\\%W"],
                         "RQ2: MaxErr (\\%) of Embedding-Within-Strata methods vs. the per-seed MaxErr distribution of Consist-Strat (Table 4). "
                         "Each embedding method produces a single deterministic MaxErr. $\\Delta$\\%Med / $\\Delta$\\%P95 / $\\Delta$\\%W = relative "
                         "change vs. Consist-Strat's median / 95th-percentile / worst per-seed MaxErr (negative = lower error).",
                         "tab:rq2-maxerr", rows_per_block=5)
    write_single_setup_table()
    write_table6_comparison(extracts)
    write_figure6(extracts)
    write_sensitivity_tables(window_comparison_rows(), variant_comparison_rows())
    n_values = collect_values().write(common.output_dir())
    print(f"rq2: tables 5 and 6, single-setup table, table 6 comparison, 2 sensitivity tables, figure 6, {n_values} reproduced values "
          f"-> {common.shown(common.output_dir())}")


if __name__ == "__main__":
    main()
