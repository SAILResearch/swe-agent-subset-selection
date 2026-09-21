"""RQ3: what drives the improvement - ablations and determinism validation (paper Section 6.3).

Usage (the paths in this text are relative to the package root):

    python rq3_ablation/reproduce.py

Reads ``results/`` and writes ``rq3_ablation/outputs/`` by default (``--results`` and ``--outputs`` change both):

    tables/table7_ablation.{tex,csv}                 paper Table 7
    tables/configuration_counts.csv                  the counts behind Figure 7 (76 configurations)
    tables/determinism_band.csv                      Finding 3.4, first test
    tables/stability_vs_difficulty_stratified.csv    Finding 3.4, second test
    reproduced_values.{md,csv}                       every number of Section 6.3, recomputed

The script is deterministic: it draws no random numbers.
"""
import collections
import itertools

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import common  # noqa: E402
import paper_values  # noqa: E402
from common import SUBSET_SIZES, SYNTHETIC_SCENARIOS  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ABLATION_SUBSET = "10pct"
REFERENCE = ("embedding_strata", "centroid_pooled")
FAMILIES = (("pure_embedding", "Pure Embedding"), ("clustering", "Clustering"), ("shortlist", "Shortlist"))
REFERENCE_FAMILY = "Embedding-Within-Strata (full method)"
CONTROL = "stability_stratified"
MIN_WINDOW = 2
MINUS, LONG_DASH = common.MINUS, "—"
PAPER_COUNTS = {"Uniform Random": 1, "Stratified sampling": 5, "Embedding + strata": 10,
                "Pure embedding": 20, "Clustering": 16, "Shortlist": 24}


def display_label(label):
    """Stored method label written as in the paper's Table 7.

    The pool multiplier is printed as 1.5x / 2x, and the clustering family's within-strata
    marker is not printed (the clustering winner clusters within outcome groups).
    """
    label = label.replace("15×", "1.5×").replace("20×", "2×")
    return label.replace(" +Strata [CL]", " [CL]")


# Table 7

def methods_at(scenario, subset=ABLATION_SUBSET):
    """All stored methods of a scenario at one subset size (overall analysis)."""
    return common.analysis(scenario)["all_methods"][subset]


def best_of_family(scenario, family):
    """The family's configuration with the lowest median RMSE."""
    candidates = {k: m for k, m in methods_at(scenario).items() if m["group"] == family}
    return min(candidates.values(), key=lambda m: m["rmse"]["median"])


def ablation_row(scenario, family):
    """Compare the family's best configuration with the full method (Centroid Pooled)."""
    reference = methods_at(scenario)[common.method_key(scenario, *REFERENCE)]
    ablation = best_of_family(scenario, family)
    p_value, delta, record = common.paired_comparison(np.asarray(reference["dist_means"]), np.asarray(ablation["dist_means"]))
    # Relative difference of the means, expressed relative to the ablation's mean RMSE.
    change = 100.0 * (ablation["rmse"]["mean"] - reference["rmse"]["mean"]) / ablation["rmse"]["mean"]
    return {"label": display_label(ablation["label"]), "median": ablation["rmse"]["median"], "change": change,
            "p": p_value, "delta": delta, "record": record}


def ablation_rows():
    """{scenario: [row per family]} with Holm adjustment over the three families."""
    result = {}
    for scenario in SYNTHETIC_SCENARIOS:
        rows = [ablation_row(scenario, family) for family, _ in FAMILIES]
        for row, holm_p in zip(rows, common.holm_adjust([r["p"] for r in rows])):
            row["holm"] = holm_p
        result[scenario] = rows
    return result


def table7_cells(rows):
    """Cell strings of Table 7, row by row; the Best Config. column names the multi-model winner."""
    reference_cells = []
    for scenario in SYNTHETIC_SCENARIOS:
        reference = methods_at(scenario)[common.method_key(scenario, *REFERENCE)]
        reference_cells += [f"{reference['rmse']['median']:.4f}", LONG_DASH, LONG_DASH, LONG_DASH]
    reference_label = methods_at("multi_model")[common.method_key("multi_model", *REFERENCE)]["label"]
    table = [[REFERENCE_FAMILY, reference_label] + reference_cells]
    for index, (_, family_label) in enumerate(FAMILIES):
        cells = []
        for scenario in SYNTHETIC_SCENARIOS:
            row = rows[scenario][index]
            p_text = "<0.001" if row["holm"] < 0.001 else f"{row['holm']:.2f}"
            cells += [f"{row['median']:.4f}", common.signed(row["change"], 1), p_text,
                      f"{common.signed(row['delta'], 3)} ({common.effect_size_letter(row['delta'])})"]
        table.append([family_label, rows["multi_model"][index]["label"]] + cells)
    return table


def write_table7(rows):
    """Write Table 7 as .csv and .tex and return its cell strings."""
    table = table7_cells(rows)
    columns = ["Med", "Delta%R", "p", "delta"]
    header = ["Family", "Best Config."] + [f"{common.SCENARIO_ABBREVIATIONS[s]} {c}" for s in SYNTHETIC_SCENARIOS for c in columns]
    common.write_csv(common.output_dir() / "tables" / "table7_ablation.csv", header, table)
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{RQ3 Ablation: best configuration of each ablation family compared against the full method "
             "(Centroid Pooled [ES]) at 10\\% subset size. Suffixes mark the family: [ES] Embedding-Within-Strata, "
             "[PE] Pure Embedding, [CL] Clustering, [SL] Shortlist. Med = median RMSE across distributions. "
             "$\\Delta$\\%R = relative change in the ablation's mean RMSE against the full method (positive = the ablation "
             "has the higher error). $p$ = Holm-adjusted Wilcoxon. $\\delta$ = Cliff's delta. "
             "Multi-model and Multi-agent datasets shown ($N{=}1{,}000$ distributions each).}", "\\label{tab:rq3-ablation}",
             "\\begin{tabular}{ll" + "r" * 8 + "}", "\\toprule",
             "Family & Best Config. & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{common.SCENARIO_TITLES[s]}}}" for s in SYNTHETIC_SCENARIOS) + " \\\\",
             " &  & " + " & ".join(["Med", "$\\Delta$\\%R", "$p$", "$\\delta$"] * 2) + " \\\\", "\\midrule"]
    for line in table:
        cells = [common.tex_escape(c).replace(MINUS, "$-$").replace(LONG_DASH, "---").replace("<", "$<$") for c in line]
        lines.append(" & ".join(cells) + " \\\\")
    common.write_text(common.output_dir() / "tables" / "table7_ablation.tex", "\n".join(lines + ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]))
    return table


# Figure 7 counts

def configuration_counts(scenario):
    """Number of stored configurations per family of Figure 7, plus the control method."""
    groups = collections.Counter(m["group"] for m in methods_at(scenario).values())
    baselines = [k.split(":")[-1] for k, m in methods_at(scenario).items() if m["group"] == "baselines"]
    stratified = [b for b in baselines if b not in ("random", CONTROL)]
    return {"Uniform Random": baselines.count("random"), "Stratified sampling": len(stratified),
            "Embedding + strata": groups["embedding_strata"], "Pure embedding": groups["pure_embedding"],
            "Clustering": groups["clustering"], "Shortlist": groups["shortlist"],
            "Stability-Stratified (control, not counted)": baselines.count(CONTROL)}


# Finding 3.4, test 1

def determinism_band(scenario, subset):
    """Share of distributions where Centroid Pooled falls below / inside / above the middle 95%
    of the 500 Consist-Strat draws (windows W >= 2)."""
    stored = common.load_json(common.results_dir() / scenario / "determinism_band.json")
    cell = stored["variants"]["w>=2_only"]["results"][subset][common.method_key(scenario, *REFERENCE)]
    return cell["below"], cell["inside"], cell["above"]


# Finding 3.4, test 2

def per_distribution(scenario, method, subset, field):
    """Per-distribution values of a method, averaged over the windows W >= 2."""
    stored = common.load_json(common.results_dir() / scenario / "per_distribution_mean_rmse.json")
    windows = [w for w in stored["methods"][method][subset] if int(w) >= MIN_WINDOW]
    return np.mean([stored["methods"][method][subset][w][field] for w in windows], axis=0)


def stability_comparison(scenario, subset):
    """Stability-Stratified (fixed, outcome-based pick) vs Diff-Strat (random pick in the same groups)."""
    control_rmse = per_distribution(scenario, CONTROL, subset, "dist_means")
    random_rmse = per_distribution(scenario, "diff_strat", subset, "dist_means")
    p_value, delta, _ = common.paired_comparison(control_rmse, random_rmse)
    return {"rmse_change": 100.0 * (control_rmse.mean() - random_rmse.mean()) / random_rmse.mean(),
            "p": p_value, "delta": delta, "unpaired_delta": common.unpaired_cliffs_delta(control_rmse, random_rmse),
            "maxerr_changes": stability_maxerr_changes(scenario, subset)}


def stability_maxerr_changes(scenario, subset):
    """Relative change (%) of Stability-Stratified's mean MaxErr vs Diff-Strat's median/P95/worst seed (W >= 2)."""
    control = per_distribution(scenario, CONTROL, subset, "dist_maxerr").mean()
    seeds = common.seed_statistics(common.load_extract(scenario), subset, "difficulty_stratified")
    return [100.0 * (control - s.mean()) / s.mean() for s in seeds]


def write_support_tables(comparisons):
    """Write the configuration counts and both determinism tests as CSV."""
    counts = {s: configuration_counts(s) for s in SYNTHETIC_SCENARIOS}
    common.write_csv(common.output_dir() / "tables" / "configuration_counts.csv", ["Family", "Paper n"] + list(SYNTHETIC_SCENARIOS),
                     [[family, PAPER_COUNTS.get(family, "-")] + [counts[s][family] for s in SYNTHETIC_SCENARIOS]
                      for family in counts["multi_model"]])
    common.write_csv(common.output_dir() / "tables" / "determinism_band.csv", ["Scenario", "Subset", "Below %", "Inside %", "Above %"],
                     [[s, common.SUBSET_LABELS[k], *determinism_band(s, k)] for s in SYNTHETIC_SCENARIOS for k in SUBSET_SIZES])
    common.write_csv(common.output_dir() / "tables" / "stability_vs_difficulty_stratified.csv",
                     ["Scenario", "Subset", "Mean RMSE change %", "Wilcoxon p", "Paired Cliff delta (positive = control better)", "Unpaired Cliff delta",
                      "MaxErr change % vs Diff-Strat median seed", "... vs P95 seed", "... vs worst seed"],
                     [[s, common.SUBSET_LABELS[k], f"{c['rmse_change']:+.1f}", f"{c['p']:.3g}", f"{c['delta']:+.3f}", f"{c['unpaired_delta']:+.3f}",
                       *[f"{v:+.1f}" for v in c["maxerr_changes"]]] for (s, k), c in comparisons.items()])


# Reproduced values

def values_counts(values, counts):
    """Configuration counts of Figure 7."""
    for scenario in SYNTHETIC_SCENARIOS:
        for family, paper in PAPER_COUNTS.items():
            values.number("Fig. 7 / Approach", f"Configurations in family '{family}', {scenario}", paper,
                          counts[scenario][family], 0, "all_methods group counts")
        total = sum(counts[scenario][f] for f in PAPER_COUNTS)
        values.number("Fig. 7 / Approach", f"Total configurations, {scenario} (control method excluded)", 76, total, 0,
                      "sum of the six families")


def values_ablations(values, rows, table):
    """Table 7 and Findings 3.1 to 3.3."""
    values.table("Table 7", [line[1:] for line in table], paper_values.TABLE7)
    paper = {"pure_embedding": ((36.7, 0.588), (30.5, 0.778)), "clustering": ((15.7, 0.620), (9.0, 0.500)),
             "shortlist": ((20.8, 0.866), (25.5, 0.872))}
    finding = {"pure_embedding": "Finding 3.1", "clustering": "Finding 3.2", "shortlist": "Finding 3.3"}
    for index, (family, family_label) in enumerate(FAMILIES):
        for scenario, (paper_change, paper_delta) in zip(SYNTHETIC_SCENARIOS, paper[family]):
            row = rows[scenario][index]
            where = finding[family]
            values.number(where, f"{family_label} best configuration ({row['label']}): RMSE difference vs Centroid Pooled, {scenario} (%)",
                          paper_change, row["change"], 1, "100*(mean_ablation - mean_full)/mean_ablation")
            values.number(where, f"... Cliff's delta, {scenario}", paper_delta, row["delta"], 3, "paired Cliff's delta on dist_means")
            values.statement(where, f"... significant (p<0.001, Holm) with large effect, {scenario}", "p<0.001; large",
                         row["holm"] < 0.001 and common.effect_size_letter(row["delta"]) == "L",
                         f"Holm p={row['holm']:.2g}, {common.effect_size_letter(row['delta'])}, W/T/L={row['record']}",
                         "Wilcoxon on dist_means, Holm over the three families")
    winners = "; ".join(f"{s}: " + ", ".join(r["label"] for r in rows[s]) for s in SYNTHETIC_SCENARIOS)
    same = all(rows["multi_model"][i]["label"] == rows["multi_agent"][i]["label"] for i in range(len(FAMILIES)))
    values.statement("Table 7", "Best configuration of each family (Pure Embedding, Clustering, Shortlist) per dataset", "one Best Config. column",
                     same, winners, "lowest median RMSE per family")
    changes = {f: [rows[s][i]["change"] for s in SYNTHETIC_SCENARIOS] for i, (f, _) in enumerate(FAMILIES)}
    for family, (low, high), text in (("pure_embedding", (30, 37), "Removing outcome grouping raises RMSE by (%)"),
                                      ("clustering", (9, 16), "Replacing outcome grouping with clustering raises RMSE by (%)"),
                                      ("shortlist", (21, 26), "Shortlist then random sampling raises RMSE by (%)")):
        values.number_range("Summary box / Finding titles", text, low, high, changes[family], "Table 7 values")


def values_determinism(values, comparisons):
    """Finding 3.4: both determinism-validation tests."""
    where = "Finding 3.4, first test"
    paper = {("multi_model", "10pct"): (20.4, 1.9), ("multi_agent", "10pct"): (30.7, 0.6),
             ("multi_model", "5pct"): (18.2, 0.3), ("multi_agent", "5pct"): (26.7, 0.5)}
    for (scenario, subset), (below, above) in paper.items():
        stored_below, _, stored_above = determinism_band(scenario, subset)
        values.number(where, f"Centroid Pooled below the middle 95% of Consist-Strat draws, {scenario} {subset} (% of distributions)",
                      below, stored_below, 1, "determinism_band.json, variant w>=2_only")
        values.number(where, f"... above it, {scenario} {subset}", above, stored_above, 1, "same")
    where = "Finding 3.4, second test"
    small = [comparisons[(s, k)]["rmse_change"] for s in SYNTHETIC_SCENARIOS for k in ("5pct", "10pct")]
    large = [comparisons[(s, k)]["rmse_change"] for s in SYNTHETIC_SCENARIOS for k in ("20pct", "30pct")]
    values.number(where, "At 5% and 10%: Stability-Stratified mean RMSE vs Diff-Strat, lowest (%)", -20.5, min(small), 1,
                  "per_distribution_mean_rmse.json, W>=2, 100*(control - DiffStrat)/DiffStrat")
    values.number(where, "... highest (%)", 18.3, max(small), 1, "same")
    values.number_range(where, "At 20% and 30%: worse than the random draw on mean RMSE by (%)", 19, 90, large, "same")
    p_values = [comparisons[(s, k)]["p"] for s in SYNTHETIC_SCENARIOS for k in ("20pct", "30pct")]
    values.statement(where, "... p < 0.001", "p<0.001", max(p_values) < 0.001, ", ".join(f"{p:.2g}" for p in p_values), "Wilcoxon on W>=2 per-distribution means")
    cells = [(s, k) for s in SYNTHETIC_SCENARIOS for k in ("20pct", "30pct")]
    values.number_range(where, "... |delta| from 0.37 to 0.72", 0.37, 0.72, [abs(comparisons[c]["unpaired_delta"]) for c in cells],
                        "classical (unpaired) Cliff's delta, W>=2 per-distribution means", decimals=2)
    source = "control: per_distribution_mean_rmse.json dist_maxerr; Diff-Strat median seed: per_seed_maxerr_extract.npz; W>=2"
    small_maxerr = [comparisons[(s, k)]["maxerr_changes"][0] for s in SYNTHETIC_SCENARIOS for k in ("5pct", "10pct")]
    large_maxerr = [comparisons[(s, k)]["maxerr_changes"][0] for s in SYNTHETIC_SCENARIOS for k in ("20pct", "30pct")]
    values.number(where, "At 5% and 10%: worst-case error vs a typical Diff-Strat draw, lowest (%)", -20.3, min(small_maxerr), 1, source)
    values.number(where, "... highest (%)", 11.4, max(small_maxerr), 1, source)
    values.number_range(where, "At 20% and 30%: worse than the typical draw on worst-case error by (%)", 21, 69, large_maxerr, source)


def stability_comparisons():
    """Stability-Stratified vs Diff-Strat for every dataset and subset size."""
    return {(s, k): stability_comparison(s, k) for s, k in itertools.product(SYNTHETIC_SCENARIOS, SUBSET_SIZES)}


def collect_values():
    """All reproduced values of Section 6.3."""
    rows = ablation_rows()
    values = common.ReproducedValues("Section 6.3 (RQ3): values stated in the paper and reproduced from `results/`")
    values_counts(values, {s: configuration_counts(s) for s in SYNTHETIC_SCENARIOS})
    values_ablations(values, rows, table7_cells(rows))
    values_determinism(values, stability_comparisons())
    return values


def main():
    """Write all outputs of this script."""
    common.configure(__doc__.split("\n")[0], DEFAULT_OUTPUT_DIR)
    write_table7(ablation_rows())
    write_support_tables(stability_comparisons())
    n_values = collect_values().write(common.output_dir())
    print(f"rq3: table 7, configuration counts, 2 determinism tables, {n_values} reproduced values "
          f"-> {common.shown(common.output_dir())}")


if __name__ == "__main__":
    main()
