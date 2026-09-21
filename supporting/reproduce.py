"""Supporting numbers outside the RQ sections (paper Sections 5.1, 5.2.1, 8.3 and 8.4).

Usage (the paths in this text are relative to the package root):

    python supporting/reproduce.py

Reads ``results/`` and writes ``supporting/outputs/reproduced_values.{md,csv}`` by default (``--results`` and ``--outputs`` change both).
The script is deterministic: it draws no random numbers.
"""
import csv

import numpy as np
from scipy.stats import binomtest, spearmanr

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import common  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def multi_model():
    """Folder of the multi_model result files."""
    return common.results_dir() / "multi_model"


SOURCE_RUN = "20241029_OpenHands-CodeAct-2.1-sonnet-20241022"
SONNET_37_RUN = "20250415_openhands"
MIN_WINDOW = 2
JACCARD_THRESHOLD = 0.30


def read_csv(path):
    """Rows of a CSV file as dictionaries."""
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


# Synthetic distributions

def distributions():
    """{name: {"instances": [...], "resolve_rate": level}} of the 1,000 synthetic distributions."""
    return common.load_json(multi_model() / "synthetic_distributions.json")


def jaccard_matrix(stored):
    """Pairwise Jaccard similarity of the distributions' instance sets, and their levels."""
    names = list(stored)
    pool = sorted({i for name in names for i in stored[name]["instances"]})
    index = {instance: column for column, instance in enumerate(pool)}
    membership = np.zeros((len(names), len(pool)), dtype=np.int32)
    for row, name in enumerate(names):
        membership[row, [index[i] for i in stored[name]["instances"]]] = 1
    shared = membership @ membership.T
    sizes = membership.sum(axis=1)
    similarity = shared / (sizes[:, None] + sizes[None, :] - shared)
    levels = np.array([round(100 * stored[name]["resolve_rate"]) for name in names])
    return similarity, levels


def mean_off_diagonal(similarity, rows=None):
    """Mean pairwise similarity over distinct pairs, optionally restricted to some distributions."""
    block = similarity if rows is None else similarity[np.ix_(rows, rows)]
    upper = np.triu_indices(len(block), k=1)
    return float(block[upper].mean())


def greedy_independent_count(similarity, threshold=JACCARD_THRESHOLD):
    """Greedy selection, in stored order, of distributions with pairwise similarity below the threshold."""
    kept = []
    for candidate in range(len(similarity)):
        if all(similarity[candidate, k] < threshold for k in kept):
            kept.append(candidate)
    return len(kept)


def values_distributions(values):
    """Section 5.2.1 and 8.4: the synthetic distributions and their overlap."""
    where = "Section 5.2.1"
    stored = distributions()
    similarity, levels = jaccard_matrix(stored)
    sizes = {len(d["instances"]) for d in stored.values()}
    values.number(where, "Synthetic distributions", 1000, len(stored), 0, "synthetic_distributions.json")
    values.statement(where, "Each distribution contains N = 250 instances", "250", sizes == {250}, f"sizes: {sorted(sizes)}", "len(instances)")
    per_level = {int(level): int(np.sum(levels == level)) for level in np.unique(levels)}
    values.statement(where, "Nine difficulty levels (10%..90%), approximately 111 distributions per level", "9 x ~111",
                 sorted(per_level) == list(range(10, 100, 10)) and all(110 <= n <= 112 for n in per_level.values()),
                 str(per_level), "resolve_rate per distribution")
    pool = common.load_json(multi_model() / "cross_run_resolve_rates.json")["pool_stats"]
    values.number(where, "Source run resolve rate (%)", 53, 100 * pool[SOURCE_RUN]["rate"], 0, "cross_run_resolve_rates.json pool_stats")
    values.number(where, "Instances failing on the source run", 235, pool[SOURCE_RUN]["total"] - pool[SOURCE_RUN]["resolved"], 0, "total - resolved")
    values.number(where, "A 10% distribution takes this many failing instances", 225, 250 * 0.9, 0, "250 * (1 - 0.10)")
    values.number(where, "Mean pairwise Jaccard similarity", 0.35, mean_off_diagonal(similarity), 2, "all C(1000,2) pairs")
    expected = 125 / (500 - 125)
    values.number(where, "Design floor of the mean Jaccard similarity (approximately)", 0.33, expected, 2, "125 / (250 + 250 - 125)")
    middle = [mean_off_diagonal(similarity, np.where(levels == level)[0]) for level in (40, 50, 60)]
    values.number_range(where, "Mean similarity within the middle difficulty levels (40% to 60%)", 0.33, 0.36, middle,
                        "pairs within the same level, levels 40/50/60", decimals=2)
    rows = read_csv(multi_model() / "difficulty_level_validity_per_distribution.csv")
    lowest = [float(r[SONNET_37_RUN]) for r in rows if r["bucket"] == "10"]
    values.number(where, "A lowest-level distribution (10% on the source run) resolves under Sonnet 3.7 at (%)", 42,
                  100 * float(np.mean(lowest)), 0, "mean resolve rate of the level-10 distributions on the Sonnet 3.7 run")
    values.number("Section 8.4", "Greedy selection of distributions with pairwise Jaccard below 0.30 retains", 3,
                  greedy_independent_count(similarity), 0, "greedy pass in stored order, multi-model")


# Cross-run validity

def values_cross_run(values):
    """Section 8.3: cross-run validity of the difficulty levels and instance churn (multi-model)."""
    where = "Section 8.3"
    rows = read_csv(multi_model() / "difficulty_level_validity_per_distribution.csv")
    runs = [column for column in rows[0] if column not in ("dist", "bucket", SOURCE_RUN)]
    levels = np.array([int(r["bucket"]) for r in rows])
    correlations, spreads, gaps, inversions = [], [], [], 0
    for run in runs:
        rates = np.array([float(r[run]) for r in rows])
        correlations.append(spearmanr(levels, rates)[0])
        level_means = [rates[levels == level].mean() for level in sorted(set(levels))]
        inversions += int(np.sum(np.diff(level_means) <= 0))
        gaps += list(100 * np.diff(level_means))
        spreads += [100 * rates[levels == level].std() for level in sorted(set(levels))]
    values.statement(where, "Spearman correlation between source-run level and each remaining run's resolve rate >= 0.99 (multi-model)",
                 ">=0.99", min(correlations) >= 0.99, ", ".join(f"{c:.4f}" for c in correlations),
                 "difficulty_level_validity_per_distribution.csv, 1,000 distributions per run")
    values.statement(where, "The ordering of difficulty levels never inverts (multi-model)", "never", inversions == 0,
                 f"{inversions} inversions of consecutive level means over {len(runs)} runs", "level means per run")
    values.statement(where, "Distributions within the same level land within about 2 percentage points of each other", "about 2 pp",
                 max(spreads) <= 2.5, f"within-level standard deviation {min(spreads):.2f}-{max(spreads):.2f} pp", "std of resolve rates per (run, level)")
    values.statement(where, "Neighboring levels sit 5-6 percentage points apart", "5-6 pp", 4.5 <= np.mean(gaps) <= 6.5,
                 f"gap between consecutive level means: mean {np.mean(gaps):.2f} pp, range {min(gaps):.2f}-{max(gaps):.2f} pp", "differences of level means")
    churn = read_csv(multi_model() / "instance_churn_per_distribution.csv")
    values.number(where, "Instances that resolve on one run and fail on the other, averaged over run pairs (multi-model, %)", 21,
                  100 * float(np.mean([float(r["gross_flip"]) for r in churn])), 0, "mean gross_flip over all distributions and run pairs")
    to_pass = float(np.mean([float(r["f2p"]) for r in churn]))
    to_fail = float(np.mean([float(r["p2f"]) for r in churn]))
    values.statement(where, "The change is mostly in one direction (newer runs resolve what older ones failed)", "mostly fail-to-pass",
                 to_pass > to_fail, f"fail-to-pass {100 * to_pass:.1f}% vs pass-to-fail {100 * to_fail:.1f}%", "mean f2p and p2f")


# Sign test per difficulty level

def level_sign_test(method, subset):
    """Number of levels whose median paired RMSE difference favors the method, and the sign-test p."""
    stored = common.load_json(multi_model() / "per_distribution_mean_rmse.json")
    windows = [w for w in stored["methods"][method][subset] if int(w) >= MIN_WINDOW]
    levels = np.array(stored["windows"][windows[0]]["buckets"])
    ours = np.mean([stored["methods"][method][subset][w]["dist_means"] for w in windows], axis=0)
    baseline = np.mean([stored["methods"]["consist_strat"][subset][w]["dist_means"] for w in windows], axis=0)
    favorable = sum(np.median((baseline - ours)[levels == level]) > 0 for level in np.unique(levels))
    n_levels = len(np.unique(levels))
    return favorable, n_levels, binomtest(int(favorable), n_levels, 0.5).pvalue


def values_sign_test(values):
    """Section 8.4: sign test over the nine difficulty levels."""
    where = "Section 8.4"
    paper = {("core_edge_pooled", "5pct"): (9, 0.004), ("core_edge_pooled", "10pct"): (9, 0.004),
             ("centroid_pooled", "5pct"): (9, 0.004), ("centroid_pooled", "10pct"): (8, 0.039)}
    for (method, subset), (paper_levels, paper_p) in paper.items():
        favorable, n_levels, p_value = level_sign_test(method, subset)
        values.number(where, f"{method} beats Consist-Strat on this many of the nine levels, {subset} (multi-model)", paper_levels,
                      favorable, 0, "median paired RMSE difference per level, W>=2")
        values.number(where, f"... sign test p, {method} {subset}", paper_p, p_value, 3, f"two-sided binomial test, {favorable} of {n_levels}")


# Dataset counts

def values_datasets(values):
    """Section 5.1: dataset sizes that the result files hold."""
    where = "Section 5.1"
    for scenario, paper_runs in (("single_setup", 10), ("multi_model", 6), ("multi_agent", 7)):
        values.number(where, f"Runs R, {scenario}" + (" (the 10-run group)" if scenario == "single_setup" else ""), paper_runs,
                      max(common.window_sizes(scenario)) + 1, 0, "largest stored window size + 1")
    cost = common.load_json(multi_model() / "cost_profile.json")["aggregate"]
    values.number(where, "Multi-model trajectories (6 x 500)", 3000, cost["num_trajectories"], 0, "cost_profile.json aggregate.num_trajectories")


def collect_values():
    """All reproduced values of Sections 5.1, 5.2.1, 8.3 and 8.4."""
    values = common.ReproducedValues("Sections 5 and 8: supporting values stated in the paper and reproduced from `results/`")
    values_datasets(values)
    values_distributions(values)
    values_cross_run(values)
    values_sign_test(values)
    return values


def main():
    """Write all outputs of this script."""
    common.configure(__doc__.split("\n")[0], DEFAULT_OUTPUT_DIR)
    n_values = collect_values().write(common.output_dir())
    print(f"supporting: {n_values} reproduced values -> {common.shown(common.output_dir())}")


if __name__ == "__main__":
    main()
