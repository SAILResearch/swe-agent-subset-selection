"""RQ1: effectiveness of the baseline subset selection approaches (paper Section 6.1).

Usage (the paths in this text are relative to the package root):

    python rq1_baselines/reproduce.py

Reads ``results/`` and writes ``rq1_baselines/outputs/`` by default (``--results`` and ``--outputs`` change both):

    tables/table3_baseline_rmse.{tex,csv}      paper Table 3
    tables/table4_baseline_maxerr.{tex,csv}    paper Table 4
    figures/figure5_baselines_multi_model.pdf  paper Figure 5
    reproduced_values.{md,csv}                 every number of Section 6.1, recomputed

The script is deterministic: it draws no random numbers.
"""
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import common  # noqa: E402
import paper_values  # noqa: E402
from common import BASELINES, BASELINE_LABELS, SCENARIOS, STRONGEST_BASELINE, SUBSET_SIZES  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIGURE_SCENARIO = "multi_model"
ALPHA = common.ALPHA

plt.rcParams.update(common.PLOT_STYLE)


# Data access

def baseline_stats(scenario, baseline, subset, window="overall"):
    """Stored statistics of one baseline (``None`` if not evaluated at this window)."""
    return common.method_stats(scenario, "baselines", baseline, subset, window)


def rmse_stat(scenario, baseline, subset, statistic, window="overall"):
    """One RMSE summary statistic (e.g. ``median``, ``p95``) of a baseline."""
    return baseline_stats(scenario, baseline, subset, window)["rmse"][statistic]


def per_distribution_rmse(scenario, baseline, subset):
    """Mean RMSE per evaluation unit (distribution, or window for single-setup)."""
    return np.asarray(baseline_stats(scenario, baseline, subset)["dist_means"])


def per_distribution_maxerr(scenario, baseline, subset):
    """MaxErr per evaluation unit, taken as the maximum over the 500 seeds."""
    return np.asarray(baseline_stats(scenario, baseline, subset)["dist_maxerr"])


# Tables

def table3_rows():
    """Rows of Table 3: median and 95th-percentile RMSE per baseline, subset and scenario."""
    rows = []
    for subset, baseline in itertools.product(SUBSET_SIZES, BASELINES):
        values = []
        for scenario in SCENARIOS:
            values += [rmse_stat(scenario, baseline, subset, "median"),
                       rmse_stat(scenario, baseline, subset, "p95")]
        rows.append((subset, baseline, values))
    return rows


def table4_rows():
    """Rows of Table 4: per-seed MaxErr (median, P95, worst) per baseline, subset, scenario."""
    rows = []
    for subset, baseline in itertools.product(SUBSET_SIZES, BASELINES):
        values = []
        for scenario in SCENARIOS:
            values += list(common.per_seed_maxerr(scenario, baseline, subset))
        rows.append((subset, baseline, values))
    return rows


def render_tex(rows, column_names, decimals, caption, label):
    """Render table rows as a booktabs LaTeX table with one column group per scenario."""
    per_scenario = len(column_names)
    header_groups = " & ".join(
        f"\\multicolumn{{{per_scenario}}}{{c}}{{{common.SCENARIO_TITLES[s]}}}" for s in SCENARIOS)
    rules = " ".join(
        f"\\cmidrule(lr){{{3 + i * per_scenario}-{2 + (i + 1) * per_scenario}}}"
        for i in range(len(SCENARIOS)))
    lines = [
        "\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{ll{'r' * per_scenario * len(SCENARIOS)}}}", "\\toprule",
        f"Subset & Approach & {header_groups} \\\\", rules,
        " &  & " + " & ".join(column_names * len(SCENARIOS)) + " \\\\", "\\midrule",
    ]
    for index, (subset, baseline, values) in enumerate(rows):
        if index and index % len(BASELINES) == 0:
            lines.append("\\midrule")
        name = common.tex_escape(BASELINE_LABELS[baseline])
        if baseline == STRONGEST_BASELINE:
            name = f"\\textbf{{{name}}}"
        subset_cell = common.tex_escape(common.SUBSET_LABELS[subset]) if index % len(BASELINES) == 0 else ""
        cells = " & ".join(f"{value:.{decimals}f}" for value in values)
        lines.append(f"{subset_cell} & {name} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def write_table(rows, stem, column_names, decimals, caption, label):
    """Write one table as ``.tex`` and ``.csv``."""
    header = ["Subset", "Approach"] + [
        f"{common.SCENARIO_ABBREVIATIONS[s]} {name}" for s in SCENARIOS for name in column_names]
    csv_rows = [[common.SUBSET_LABELS[subset], BASELINE_LABELS[baseline]]
                + [f"{value:.{decimals}f}" for value in values]
                for subset, baseline, values in rows]
    common.write_csv(common.output_dir() / "tables" / f"{stem}.csv", header, csv_rows)
    common.write_text(common.output_dir() / "tables" / f"{stem}.tex",
                      render_tex(rows, column_names, decimals, caption, label))


def write_tables():
    """Write Table 3 and Table 4."""
    write_table(
        table3_rows(), "table3_baseline_rmse", ["Med", "P95"], 4,
        "RQ1: Baseline RMSE across datasets and subset sizes. Med = median RMSE, P95 = 95th percentile RMSE "
        "across $N{=}1{,}000$ synthetic distributions. The strongest baseline (Consist-Strat) is bolded. "
        "MaxErr is reported separately in Table 4.", "tab:rq1-rmse")
    write_table(
        table4_rows(), "table4_baseline_maxerr", ["Med", "P95", "Worst"], 2,
        "RQ1: Per-seed worst-case error (MaxErr, \\%) across datasets and subset sizes. Each of 500 random seeds "
        "selects one subset whose MaxErr is computed independently. Med = median across seeds (the error a typical "
        "draw produces), P95 = 95th percentile (an unlucky draw), Worst = maximum across all seeds. "
        "The strongest baseline (Consist-Strat) is bolded.", "tab:rq1-maxerr")


# Figure 5

def shared_axis_limits(getter):
    """Y-limits spanning the 0.5th-99.5th percentiles of all baselines in all scenarios."""
    arrays = [getter(scenario, baseline, subset)
              for scenario in SCENARIOS for subset in SUBSET_SIZES for baseline in BASELINES
              if baseline_stats(scenario, baseline, subset) is not None]
    low = min(np.percentile(a, 0.5) for a in arrays)
    high = max(np.percentile(a, 99.5) for a in arrays)
    margin = 0.05 * (high - low)
    return max(0.0, low - margin), high + margin


def draw_panel(axis, arrays):
    """Draw one violin + box panel for the six baselines."""
    positions = np.arange(len(arrays))
    grays = ["#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252", "#252525"]
    violins = axis.violinplot(arrays, positions=positions, showmedians=False,
                              showextrema=False, widths=0.7)
    for body, gray in zip(violins["bodies"], grays):
        body.set_facecolor(gray)
        body.set_alpha(0.6)
    line = dict(color="black", linewidth=1.5)
    axis.boxplot(arrays, positions=positions, widths=0.55, patch_artist=True, showfliers=False,
                 boxprops=dict(facecolor="white", **line), whiskerprops=line, capprops=line,
                 medianprops=dict(color="black", linewidth=2.0))
    axis.set_xticks(positions)


def write_figure5():
    """Figure 5: RMSE (top) and MaxErr (bottom) of the baselines, multi-model, per subset size.

    The bottom row shows, per distribution, the maximum MaxErr over the 500 seeds.
    """
    metrics = [("RMSE", per_distribution_rmse), ("Max Error (pp)", per_distribution_maxerr)]
    figure, axes = plt.subplots(len(metrics), len(SUBSET_SIZES), figsize=(7.5, 4.8), sharey="row")
    for row, (ylabel, getter) in enumerate(metrics):
        limits = shared_axis_limits(getter)
        for column, subset in enumerate(SUBSET_SIZES):
            axis = axes[row, column]
            draw_panel(axis, [getter(FIGURE_SCENARIO, b, subset) for b in BASELINES])
            is_bottom = row == len(metrics) - 1
            axis.set_xticklabels([BASELINE_LABELS[b] for b in BASELINES] if is_bottom else [],
                                 rotation=45, ha="right")
            axis.set_ylim(*limits)
            if row == 0:
                axis.set_title(common.SUBSET_LABELS[subset])
            if column == 0:
                axis.set_ylabel(ylabel)
    figure.tight_layout()
    path = common.output_dir() / "figures" / "figure5_baselines_multi_model.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


# Statistics

def strongest_vs_other_baselines(scenario, subset):
    """Compare Consist-Strat with each other baseline; Holm over the five comparisons.

    Units where a method was not evaluated are dropped by aligning on the shorter series
    (single-setup only: Consist-Strat has no window-size-1 value).
    """
    reference = per_distribution_rmse(scenario, STRONGEST_BASELINE, subset)
    others = [b for b in BASELINES if b != STRONGEST_BASELINE]
    results = []
    for other in others:
        errors = per_distribution_rmse(scenario, other, subset)
        n = min(len(reference), len(errors))
        results.append(common.paired_comparison(reference[-n:], errors[-n:]))
    adjusted = common.holm_adjust([r[0] for r in results])
    return {other: (adjusted[i], results[i][1], results[i][2]) for i, other in enumerate(others)}


def median_reduction_vs_random(scenario, subset, baseline=STRONGEST_BASELINE):
    """Percent reduction of median RMSE relative to Random."""
    random_median = rmse_stat(scenario, "random", subset, "median")
    return 100.0 * (random_median - rmse_stat(scenario, baseline, subset, "median")) / random_median


# Reproduced values

def values_finding_1_1(values):
    """Finding 1.1: Consist-Strat is the strongest baseline."""
    where = "Finding 1.1"
    lowest_median, lowest_p95 = [], []
    for scenario, subset in itertools.product(SCENARIOS, SUBSET_SIZES):
        for statistic, failures in (("median", lowest_median), ("p95", lowest_p95)):
            best = min(BASELINES, key=lambda b: rmse_stat(scenario, b, subset, statistic))
            if best != STRONGEST_BASELINE:
                failures.append(f"{scenario} {subset}: {BASELINE_LABELS[best]}")
    values.statement(where, "Consist-Strat has the lowest median RMSE in all datasets and subset sizes",
                 "all 12 cells", not lowest_median, "exceptions: " + ("none" if not lowest_median else "; ".join(lowest_median)),
                 "all_methods.<subset>.<baseline>.rmse.median")
    values.statement(where, "The same holds at the 95th percentile of RMSE", "all 12 cells", not lowest_p95,
                 "exceptions: " + ("none" if not lowest_p95 else "; ".join(lowest_p95)),
                 "all_methods.<subset>.<baseline>.rmse.p95")
    paper_ranges = {"single_setup": (51, 51), "multi_model": (31, 32), "multi_agent": (40, 42)}
    for scenario, (low, high) in paper_ranges.items():
        reductions = [median_reduction_vs_random(scenario, s) for s in SUBSET_SIZES]
        values.number_range(where, f"Median RMSE reduction vs Random, {scenario} (% over the four subset sizes)",
                            low, high, reductions, "100*(median_Random - median_ConsistStrat)/median_Random")
    paper_5pct = {"single_setup": (0.07, 0.04), "multi_model": (0.11, 0.07), "multi_agent": (0.10, 0.06)}
    for scenario, (random_value, consist_value) in paper_5pct.items():
        values.number(where, f"5% subset, Random median RMSE, {scenario}", random_value,
                      rmse_stat(scenario, "random", "5pct", "median"), 2, "rmse.median")
        values.number(where, f"5% subset, Consist-Strat median RMSE, {scenario}", consist_value,
                      rmse_stat(scenario, STRONGEST_BASELINE, "5pct", "median"), 2, "rmse.median")
    for scenario, subset in itertools.product(SCENARIOS, SUBSET_SIZES):
        row = common.comparison_row(scenario, subset, "vs_random",
                                    common.method_key(scenario, "baselines", STRONGEST_BASELINE))
        record = f"{row['wins']}/{row['ties']}/{row['losses']}"
        detail = f"Holm p={row['holm_p']:.3g}, delta={row['cliff_d']:.3f}, W/T/L={record}"
        is_true = row["holm_p"] < 0.001 and round(row["cliff_d"], 3) == 1.0 and record == "1000/0/0"
        values.statement(where, f"Consist-Strat vs Random, {scenario} {subset}: p<0.001, delta=1.000, 1000/0/0",
                     "p<0.001; 1.000; 1000/0/0", is_true, detail, "subset_sizes.<subset>.ranking.vs_random (stored)")


def values_figure5_caption(values):
    """Figure 5 caption numbers."""
    where = "Fig. 5 caption"
    for subset, paper in (("5pct", 35), ("30pct", 11)):
        worst = common.per_seed_maxerr("multi_model", STRONGEST_BASELINE, subset)[2]
        values.number(where, f"Best baseline's worst draw at {common.SUBSET_LABELS[subset]} (multi-model, %)",
                      paper, worst, 0, "per_seed_maxerr_all_baselines: baseline_seed_max_mean")


def values_finding_1_2(values):
    """Finding 1.2: repository metadata does not help."""
    where = "Finding 1.2"
    values.number(where, "Multi-model 5%: Repo-Strat median RMSE", 0.10,
                  rmse_stat("multi_model", "repo_stratified", "5pct", "median"), 2, "rmse.median")
    gain = median_reduction_vs_random("multi_agent", "30pct", "repo_stratified")
    values.statement(where, "Multi-agent 30%: Repo-Strat improves on Random by less than 2%", "<2%",
                 0 <= gain < 2, f"{gain:.2f}%", "median reduction vs Random")
    all_gains = [median_reduction_vs_random(s, k, "repo_stratified")
                 for s in common.SYNTHETIC_SCENARIOS for k in SUBSET_SIZES]
    values.statement("Summary box", "Repository Stratification provides negligible benefits (<2%)", "<2%",
                 max(all_gains) < 2, f"multi-model/multi-agent gains {min(all_gains):.2f}% to {max(all_gains):.2f}%; "
                 "single-setup: " + ", ".join(f"{median_reduction_vs_random('single_setup', k, 'repo_stratified'):.1f}%"
                                              for k in SUBSET_SIZES),
                 "median reduction vs Random, all subset sizes")
    repo_consist = rmse_stat("multi_model", "repo_consistency_stratified", "5pct", "median")
    consist = rmse_stat("multi_model", STRONGEST_BASELINE, "5pct", "median")
    values.number(where, "Multi-model 5%: Repo x Consist median RMSE", 0.08, repo_consist, 2, "rmse.median")
    values.number(where, "Multi-model 5%: Consist-Strat median RMSE", 0.07, consist, 2, "rmse.median")
    values.number(where, "Multi-model 5%: degradation of Repo x Consist vs Consist-Strat (%)", 11,
                  100 * (repo_consist - consist) / consist, 0, "100*(RepoConsist - Consist)/Consist, medians")
    worse = [(s, k) for s in SCENARIOS for k in SUBSET_SIZES
             if rmse_stat(s, "repo_consistency_stratified", k, "median") > rmse_stat(s, STRONGEST_BASELINE, k, "median")
             and rmse_stat(s, "repo_difficulty_stratified", k, "median") > rmse_stat(s, "difficulty_stratified", k, "median")]
    values.statement(where, "Repo-combined variants are worse than their non-repository variants in nearly all cells",
                 "nearly all", len(worse) >= 10, f"both repo variants worse in {len(worse)} of 12 cells",
                 "rmse.median comparisons")
    for subset, paper in (("10pct", 0.0340), ("20pct", 0.0538), ("30pct", 0.0385)):
        values.number(where, f"Single-setup Repo x Consist median RMSE at {common.SUBSET_LABELS[subset]}", paper,
                      rmse_stat("single_setup", "repo_consistency_stratified", subset, "median"), 4, "rmse.median")


def values_finding_1_3(values):
    """Finding 1.3 and the two paragraphs after Table 4: worst-case risk."""
    where = "Finding 1.3"
    source = "per_seed_maxerr_all_baselines (median / P95 / worst across 500 seeds)"
    quoted = [
        ("multi_model", "random", "5pct", 0, 20.21), ("multi_agent", "random", "5pct", 0, 21.76),
        ("multi_model", "random", "5pct", 2, 47.22), ("multi_agent", "random", "5pct", 2, 47.78),
        ("multi_model", STRONGEST_BASELINE, "5pct", 0, 14.42), ("multi_agent", STRONGEST_BASELINE, "5pct", 0, 14.28),
        ("multi_model", STRONGEST_BASELINE, "5pct", 2, 34.61), ("multi_agent", STRONGEST_BASELINE, "5pct", 2, 31.25),
        ("multi_model", STRONGEST_BASELINE, "30pct", 2, 11.44), ("multi_agent", STRONGEST_BASELINE, "30pct", 2, 10.5),
        ("multi_model", STRONGEST_BASELINE, "10pct", 0, 9.63), ("multi_model", STRONGEST_BASELINE, "10pct", 1, 15.87),
    ]
    names = ("median", "P95", "worst")
    for scenario, baseline, subset, index, paper in quoted:
        values.number(where, f"{BASELINE_LABELS[baseline]} per-seed MaxErr {names[index]}, {scenario} {subset}",
                      paper, common.per_seed_maxerr(scenario, baseline, subset)[index], 2, source)
    drops = {name: [] for name in names}
    for scenario, subset in itertools.product(common.SYNTHETIC_SCENARIOS, SUBSET_SIZES):
        random_values = common.per_seed_maxerr(scenario, "random", subset)
        consist_values = common.per_seed_maxerr(scenario, STRONGEST_BASELINE, subset)
        for name, r, c in zip(names, random_values, consist_values):
            drops[name].append(100 * (r - c) / r)
    every = [value for values in drops.values() for value in values]
    detail = "; ".join(f"{name}: " + ", ".join(f"{value:.1f}" for value in v) for name, v in drops.items())
    values.number_range(where, "Worst-case error drop from Random to Consist-Strat, multi-model and multi-agent "
                        f"(%; the text does not say which statistic. All 24 values, ordered multi-model 5/10/20/30% then "
                        f"multi-agent 5/10/20/30%: {detail})", 27, 35, every,
                        "100*(Random - ConsistStrat)/Random over median, P95, worst and all subset sizes")
    ratio = (common.per_seed_maxerr("multi_model", STRONGEST_BASELINE, "5pct")[2]
             / common.per_seed_maxerr("multi_model", STRONGEST_BASELINE, "30pct")[2])
    values.number(where, "5% to 30% cuts multi-model worst-seed error by a factor of", 3, ratio, 0,
                  "per_seed_maxerr_all_baselines: Consist-Strat worst seed at 5% / worst seed at 30%, multi-model")
    values.number(where, "Multi-model 10%: Consist-Strat median RMSE", 0.05,
                  rmse_stat("multi_model", STRONGEST_BASELINE, "10pct", "median"), 2, "rmse.median")
    values.number(where, "Worst-seed error at 30% subset size (%)", 11,
                  common.per_seed_maxerr("multi_model", STRONGEST_BASELINE, "30pct")[2], 0, "multi-model worst")


def values_sensitivity(values):
    """Sensitivity analysis paragraphs."""
    where = "Sensitivity: window"
    for scenario in SCENARIOS:
        absent = all(baseline_stats(scenario, STRONGEST_BASELINE, k, 1) is None for k in SUBSET_SIZES)
        values.statement(where, f"At W=1 Consist-Strat cannot be applied, {scenario}", "not applicable", absent,
                     "no W=1 entry" if absent else "W=1 entry exists", "analyses.w1.all_methods")
        used = {common.analysis(scenario, 1)["subset_sizes"][k]["ranking"]["best_baseline_used"] for k in SUBSET_SIZES}
        values.statement(where, f"At W=1 Diff-Strat is the strongest available baseline, {scenario}", "Diff-Strat",
                     used == {common.method_key(scenario, "baselines", "difficulty_stratified")},
                     f"stored reference baseline at W=1: {sorted(used)}; lowest median per subset: "
                     + ", ".join(BASELINE_LABELS[min((b for b in BASELINES if baseline_stats(scenario, b, k, 1)),
                                                     key=lambda b: rmse_stat(scenario, b, k, "median", 1))]
                                 for k in SUBSET_SIZES),
                     "analyses.w1 ranking.best_baseline_used and rmse.median")
        cells = [(w, k) for w in common.window_sizes(scenario) if w >= 2 for k in SUBSET_SIZES]
        losing = [f"W={w} {k}: {BASELINE_LABELS[best]}" for w, k in cells
                  for best in [min((b for b in BASELINES if baseline_stats(scenario, b, k, w)),
                                   key=lambda b: rmse_stat(scenario, b, k, "median", w))]
                  if best != STRONGEST_BASELINE]
        values.statement(where, f"When W>=2 Consist-Strat is the dominant baseline, {scenario}", "dominant",
                     not losing, f"lowest median RMSE in {len(cells) - len(losing)} of {len(cells)} (W, subset) cells"
                     + (f"; exceptions: {'; '.join(losing)}" if losing else ""), "analyses.w<W>.all_methods rmse.median")
    where = "Sensitivity: sizes and datasets"
    ordered = all(rmse_stat(s, STRONGEST_BASELINE, k, "median") < rmse_stat(s, "difficulty_stratified", k, "median")
                  < rmse_stat(s, "random", k, "median") for s in SCENARIOS for k in SUBSET_SIZES)
    values.statement(where, "Ranking Consist-Strat > Diff-Strat > Random holds in all datasets and subset sizes",
                 "all", ordered, "median RMSE ordering checked in 12 cells", "rmse.median")


def values_summary(values):
    """Summary box of RQ1."""
    where = "Summary box"
    reductions = [median_reduction_vs_random(s, k) for s in SCENARIOS for k in SUBSET_SIZES]
    values.number_range(where, "Consist-Strat reduces median RMSE vs Random (%)", 31, 51, reductions,
                        "all scenarios and subset sizes")
    for scenario, subset in itertools.product(SCENARIOS, SUBSET_SIZES):
        comparisons = strongest_vs_other_baselines(scenario, subset)
        failing = [BASELINE_LABELS[b] for b, (p, delta, _) in comparisons.items() if not (p < ALPHA and delta > 0)]
        detail = "; ".join(f"{BASELINE_LABELS[b]}: Holm p={p:.2g}, delta={d:+.3f}, W/T/L={w}/{t}/{l}"
                           for b, (p, d, (w, t, l)) in comparisons.items())
        values.statement(where, f"Consist-Strat significantly outperforms all other baselines, {scenario} {subset}",
                     "significant vs all five", not failing, detail,
                     "computed here: Wilcoxon on dist_means, Holm over the 5 baseline pairs, paired Cliff's delta")
    typical = [common.per_seed_maxerr(s, STRONGEST_BASELINE, "5pct")[0] for s in common.SYNTHETIC_SCENARIOS]
    values.number_range(where, "Consist-Strat typical-draw worst-case error at 5% (about 14%)", 14, 14, typical,
                        "per-seed median, multi-model and multi-agent")
    worst5 = [common.per_seed_maxerr(s, STRONGEST_BASELINE, "5pct")[2] for s in common.SYNTHETIC_SCENARIOS]
    values.number_range(where, "Worst seed at 5% (%)", 31, 35, worst5, "per-seed worst")
    worst30 = [common.per_seed_maxerr(s, STRONGEST_BASELINE, "30pct")[2] for s in common.SYNTHETIC_SCENARIOS]
    values.number_range(where, "Worst seed at 30% (%)", 10, 11, worst30, "per-seed worst")


def evaluation_units():
    """Number of evaluation units per scenario: synthetic distributions, or window sizes for single-setup."""
    return {s: common.analysis(s).get("n_distributions", common.analysis(s).get("n_observations")) for s in SCENARIOS}


def collect_values():
    """All reproduced values of Section 6.1."""
    values = common.ReproducedValues("Section 6.1 (RQ1): values stated in the paper and reproduced from `results/`")
    for name, rows, decimals, printed in (("Table 3", table3_rows(), 4, paper_values.TABLE3),
                                          ("Table 4", table4_rows(), 2, paper_values.TABLE4)):
        generated = [[f"{v:.{decimals}f}" for v in cells] for _, _, cells in rows]
        values.table(name, generated, [[f"{v:.{decimals}f}" for v in row] for row in printed])
    units = evaluation_units()
    values.statement("Table 3 caption", "N = 1,000 synthetic distributions", "1,000",
                     all(units[s] == 1000 for s in common.SYNTHETIC_SCENARIOS),
                     f"multi-model {units['multi_model']}, multi-agent {units['multi_agent']}; single-setup is evaluated over "
                     f"{units['single_setup']} window sizes", "analyses.overall.n_distributions / n_observations")
    values_finding_1_1(values)
    values_figure5_caption(values)
    values_finding_1_2(values)
    values_finding_1_3(values)
    values_sensitivity(values)
    values_summary(values)
    return values


def main():
    """Write all outputs of this script."""
    common.configure(__doc__.split("\n")[0], DEFAULT_OUTPUT_DIR)
    write_tables()
    write_figure5()
    n_values = collect_values().write(common.output_dir())
    print(f"rq1: tables 3 and 4, figure 5, {n_values} reproduced values -> {common.shown(common.output_dir())}")


if __name__ == "__main__":
    main()
