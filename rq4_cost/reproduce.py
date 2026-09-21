"""RQ4: how much subset selection reduces evaluation cost (paper Section 6.4).

Usage (the paths in this text are relative to the package root):

    python rq4_cost/reproduce.py

Reads ``results/multi_model/`` and writes ``rq4_cost/outputs/`` by default (``--results`` and ``--outputs`` change both):

    tables/table8_cost_profile.{tex,csv}    paper Table 8
    tables/table9_cost_savings.{tex,csv}    paper Table 9
    reproduced_values.{md,csv}              every number of Section 6.4, recomputed

The script is deterministic: the random subset draws behind Table 9 are stored results.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import common  # noqa: E402
import paper_values  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def cost_dir():
    """Folder of the multi_model result files."""
    return common.results_dir() / "multi_model"


RUNS = (  # run identifier in the stored files, display name of Table 8
    ("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", "Sonnet 3.5 (Oct '24)"),
    ("20250415_openhands", "Sonnet 3.7 (Apr '25)"),
    ("20250520_openhands_devstral_small", "Devstral (May '25)"),
    ("20250524_openhands_claude_4_sonnet", "Sonnet 4 (May '25)"),
    ("20250716_openhands_kimi_k2", "Kimi K2 (Jul '25)"),
    ("20250807_openhands_gpt5", "GPT-5 (Aug '25)"),
)
MILLION = 1e6


def cost_profile():
    """Stored cost profile of the six multi-model runs."""
    return common.load_json(cost_dir() / "cost_profile.json")


def centroid_cost():
    """Stored token cost of the Centroid Pooled 10% subsets."""
    return common.load_json(cost_dir() / "cost_centroid_pooled_10pct.json")


# Tables

def table8_rows(profile):
    """Rows of Table 8: total and mean tokens (millions), mean steps, input/output ratio."""
    rows = []
    for run, name in RUNS:
        stats = profile["agent_profiles"][run]
        rows.append([name, f"{stats['total_triangular_tokens'] / MILLION:,.0f}", f"{stats['per_instance']['mean'] / MILLION:.2f}",
                     f"{stats['mean_steps']:.1f}", f"{stats['input_output_ratio']:.0f}×"])
    total = profile["aggregate"]
    mean_steps = sum(profile["agent_profiles"][run]["mean_steps"] for run, _ in RUNS) / len(RUNS)
    rows.append(["Aggregate", f"{total['total_triangular_tokens'] / MILLION:,.0f}", f"{total['per_trajectory']['mean'] / MILLION:.2f}",
                 f"{mean_steps:.1f}", f"{total['input_output_ratio']:.0f}×"])
    return rows


def widest_range(profile, subset):
    """Middle 95% of the random draws, from the run with the widest spread."""
    ranges = [(cell["cost_share"]["ci95_low"], cell["cost_share"]["ci95_high"])
              for cell in (profile["savings_empirical"]["per_agent"][run][subset] for run, _ in RUNS)]
    return max(ranges, key=lambda r: r[1] - r[0])


def table9_rows(profile):
    """Rows of Table 9: mean cost share, widest 95% range, savings."""
    rows = []
    for subset in common.SUBSET_SIZES:
        share = profile["savings_empirical"]["aggregate"][subset]["weighted_mean_cost_share_pct"]
        low, high = widest_range(profile, subset)
        rows.append([common.SUBSET_LABELS[subset], f"{share:.1f}%", f"[{low:.1f}%, {high:.1f}%]", f"{100 - share:.1f}%"])
    return rows


def write_table(stem, header, rows, caption, label):
    """Write one table as .csv and .tex."""
    common.write_csv(common.output_dir() / "tables" / f"{stem}.csv", header, rows)
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{l" + "r" * (len(header) - 1) + "}", "\\toprule",
             " & ".join(common.tex_escape(h) for h in header) + " \\\\", "\\midrule"]
    lines += [" & ".join(common.tex_escape(cell) for cell in row) + " \\\\" for row in rows]
    common.write_text(common.output_dir() / "tables" / f"{stem}.tex", "\n".join(lines + ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))


# Reproduced values

def collect_values():
    """All reproduced values of Section 6.4."""
    profile = cost_profile()
    table8, table9 = table8_rows(profile), table9_rows(profile)
    values = common.ReproducedValues("Section 6.4 (RQ4): values stated in the paper and reproduced from `results/`")
    values.table("Table 8", table8, paper_values.TABLE8, row_names=[row[0] for row in table8],
                 column_names=["OpenHands Run", "Total (M)", "Mean (M)", "Steps", "I/O Ratio"])
    values.table("Table 9", table9, paper_values.TABLE9)
    total, where = profile["aggregate"], "Section 6.4 text"
    values.number(where, "Trajectories measured", 3000, total["num_trajectories"], 0, "aggregate.num_trajectories")
    values.number(where, "Total tokens across all six runs (billions)", 3.44, total["total_triangular_tokens"] / 1e9, 2, "aggregate.total_triangular_tokens")
    per_run = [profile["agent_profiles"][run]["total_triangular_tokens"] / MILLION for run, _ in RUNS]
    values.number(where, "Cheapest full run (M tokens)", 335, min(per_run), 0, "agent_profiles.*.total_triangular_tokens")
    values.number(where, "Most expensive full run (M tokens)", 945, max(per_run), 0, "same")
    values.number(where, "Input share of the token volume (%)", 99.3, total["input_share_pct"], 1, "aggregate.input_share_pct")
    sonnet4, gpt5 = (profile["agent_profiles"][RUNS[i][0]] for i in (3, 5))
    values.number(where, "Sonnet 4 steps per instance", 69.7, sonnet4["mean_steps"], 1, "mean_steps")
    values.number(where, "GPT-5 steps per instance", 31.2, gpt5["mean_steps"], 1, "mean_steps")
    values.number(where, "Step ratio Sonnet 4 / GPT-5", 2.2, sonnet4["mean_steps"] / gpt5["mean_steps"], 1, "ratio of mean_steps")
    values.number(where, "Sonnet 4 mean cost per instance (M)", 1.89, sonnet4["per_instance"]["mean"] / MILLION, 2, "per_instance.mean")
    values.number(where, "GPT-5 mean cost per instance (M)", 0.70, gpt5["per_instance"]["mean"] / MILLION, 2, "per_instance.mean")
    values.number(where, "Cost ratio Sonnet 4 / GPT-5", 2.7, sonnet4["per_instance"]["mean"] / gpt5["per_instance"]["mean"], 1, "ratio of per_instance.mean")
    metadata = profile["metadata"]
    values.number("Approach", "Tool outputs truncated above (tokens)", 10000, metadata["max_observation_tokens"], 0, "metadata.max_observation_tokens")
    values.number("Approach", "Random subsets drawn per run and subset size", 10000, metadata["n_monte_carlo_simulations"], 0, "metadata.n_monte_carlo_simulations")
    where = "Finding 4.1"
    values.number(where, "Per-instance cost differs by up to (x)", 492, total["per_trajectory"]["cost_ratio_max_to_min"], 0,
                  "aggregate.per_trajectory.cost_ratio_max_to_min")
    shares = {k: profile["savings_empirical"]["aggregate"][k]["weighted_mean_cost_share_pct"] for k in common.SUBSET_SIZES}
    values.statement(where, "A k% subset consumes k% of the token cost on average (to one decimal place)", "k%",
                 all(f"{shares[k]:.1f}" == f"{float(k[:-3]):.1f}" for k in shares), ", ".join(f"{k}: {v}" for k, v in shares.items()),
                 "savings_empirical.aggregate.*.weighted_mean_cost_share_pct")
    where = "Finding 4.2"
    summary = centroid_cost()["summary"]
    share = 100 - summary["overall"]["aggregate_saved_pct"]
    values.number(where, "Centroid Pooled 10% subset consumes (% of the run's token cost)", 10.48, share, 2, "100 - summary.overall.aggregate_saved_pct")
    low, high = widest_range(profile, "10pct")
    values.statement(where, "... which sits inside the middle 95% of the random draws at the same size", "inside", low <= share <= high,
                 f"{share:.2f}% vs [{low}%, {high}%]", "Table 9 range")
    subset_tokens = profile["savings_empirical"]["aggregate"]["10pct"]["estimated_subset_tokens"]
    values.number(where, "A 10% subset cuts token consumption to roughly (M)", 345, subset_tokens / MILLION, 0,
                  "savings_empirical.aggregate.10pct.estimated_subset_tokens")
    return values

def main():
    """Write all outputs of this script."""
    common.configure(__doc__.split("\n")[0], DEFAULT_OUTPUT_DIR)
    profile = cost_profile()
    table8, table9 = table8_rows(profile), table9_rows(profile)
    write_table("table8_cost_profile", ["OpenHands Run", "Total (M)", "Mean (M)", "Steps", "I/O Ratio"], table8,
                "Per-run cost profile across 500 SWE-Bench Verified instances. All runs use the OpenHands framework with different models. "
                "Token counts reflect cumulative cost (input re-transmission at every turn). Total (M) = total \\#tokens in millions across "
                "all 500 instances. Mean (M) = mean \\#tokens per instance in millions. Steps = mean number of agent steps per instance. "
                "I/O Ratio = ratio of \\#input tokens to \\#output tokens.", "tab:rq4-profile")
    write_table("table9_cost_savings", ["Subset", "Cost Share", "95% Range", "Savings"], table9,
                "Empirical cost savings by subset size. For each of the six runs, we draw 10,000 random subsets of size k\\% and compute each "
                "subset's share of the run's total token cost. Cost Share is the mean over the 10,000 draws, averaged across the six runs. "
                "95\\% Range is the middle 95\\% of the draws (2.5th to 97.5th percentile), taken from the run with the widest spread. "
                "Savings = 100 $-$ Cost Share.", "tab:rq4-savings")
    n_values = collect_values().write(common.output_dir())
    print(f"rq4: tables 8 and 9, {n_values} reproduced values -> {common.shown(common.output_dir())}")


if __name__ == "__main__":
    main()
