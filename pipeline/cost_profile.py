"""Cost stage 2: cost profile of every run and cost share of random subsets (paper Tables 8 and 9).

    python pipeline/cost_profile.py --dataset multi_model [--input RESULT_DIR] [--output RESULT_DIR]

Reads ``cost_per_trajectory.json`` and writes ``cost_profile.json`` (both default to data/<dataset>/results).

For every run and subset size, 10,000 random subsets are drawn (seed ``config.SEED``, one generator per run) and the
share of the run's triangular token cost that each subset consumes is recorded. The aggregate over runs weights
every run by its total cost.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

SUBSET_PERCENTAGES = (5, 10, 20, 30)
N_DRAWS = 10000


def run_profile(trajectories):
    """Cost totals and per-instance statistics of one run, and its per-instance cost array."""
    costs = np.array([t["triangular_total_tokens"] for t in trajectories.values()])
    total_input = int(np.array([t["triangular_input_tokens"] for t in trajectories.values()]).sum())
    total_output = int(np.array([t["triangular_output_tokens"] for t in trajectories.values()]).sum())
    steps = np.array([t["num_steps"] for t in trajectories.values()])
    profile = {
        "num_instances": len(costs),
        "total_triangular_tokens": int(costs.sum()),
        "total_triangular_input_tokens": total_input,
        "total_triangular_output_tokens": total_output,
        "total_flat_tokens": int(np.array([t["flat_tokens"] for t in trajectories.values()]).sum()),
        "input_output_ratio": round(total_input / total_output, 1) if total_output > 0 else 0,
        "per_instance": {
            "mean": round(float(np.mean(costs))), "median": round(float(np.median(costs))), "std": round(float(np.std(costs))),
            "min": int(np.min(costs)), "max": int(np.max(costs)),
            "p5": round(float(np.percentile(costs, 5))), "p25": round(float(np.percentile(costs, 25))),
            "p75": round(float(np.percentile(costs, 75))), "p95": round(float(np.percentile(costs, 95))),
        },
        "mean_steps": round(float(np.mean(steps)), 1),
    }
    return profile, costs


def random_subset_shares(costs, seed):
    """Cost share (%) of random subsets of one run, per subset size: mean, spread and middle 95% of the draws."""
    rng = np.random.default_rng(seed)
    total = costs.sum()
    n = len(costs)
    results = {}
    for percentage in SUBSET_PERCENTAGES:
        k = max(1, int(n * percentage / 100))
        shares = np.zeros(N_DRAWS)
        for draw in range(N_DRAWS):
            chosen = rng.choice(n, size=k, replace=False)
            shares[draw] = costs[chosen].sum() / total * 100
        savings = 100.0 - shares
        results[f"{percentage}pct"] = {
            "subset_size": k,
            "subset_fraction": percentage / 100,
            "cost_share": {"mean": round(float(np.mean(shares)), 2), "std": round(float(np.std(shares)), 2),
                           "ci95_low": round(float(np.percentile(shares, 2.5)), 2),
                           "ci95_high": round(float(np.percentile(shares, 97.5)), 2)},
            "savings": {"mean_pct": round(float(np.mean(savings)), 2), "std_pct": round(float(np.std(savings)), 2),
                        "ci95_low": round(float(np.percentile(savings, 2.5)), 2),
                        "ci95_high": round(float(np.percentile(savings, 97.5)), 2),
                        "mean_tokens": round(float(total * np.mean(savings) / 100))},
        }
    return results


def weighted_aggregate(profiles, shares):
    """Cost share and savings per subset size, averaged over runs with weights proportional to each run's total cost."""
    weights = np.array([profiles[run]["total_triangular_tokens"] for run in profiles])
    weights = weights / weights.sum()
    full = sum(profile["total_triangular_tokens"] for profile in profiles.values())
    aggregate = {}
    for percentage in SUBSET_PERCENTAGES:
        key = f"{percentage}pct"
        savings = float(np.average([shares[run][key]["savings"]["mean_pct"] for run in profiles], weights=weights))
        share = float(np.average([shares[run][key]["cost_share"]["mean"] for run in profiles], weights=weights))
        aggregate[key] = {"subset_fraction": percentage / 100, "weighted_mean_savings_pct": round(savings, 2),
                          "weighted_mean_cost_share_pct": round(share, 2), "full_benchmark_tokens": full,
                          "estimated_subset_tokens": round(full * share / 100),
                          "estimated_saved_tokens": round(full * savings / 100)}
    return aggregate


def overall_profile(profiles, all_costs):
    """Totals, input/output split and per-trajectory statistics over all runs."""
    total = int(all_costs.sum())
    total_input = sum(p["total_triangular_input_tokens"] for p in profiles.values())
    total_output = sum(p["total_triangular_output_tokens"] for p in profiles.values())
    return {
        "num_agents": len(profiles), "num_trajectories": len(all_costs), "total_triangular_tokens": total,
        "total_triangular_input_tokens": total_input, "total_triangular_output_tokens": total_output,
        "input_share_pct": round(total_input / total * 100, 1), "output_share_pct": round(total_output / total * 100, 1),
        "input_output_ratio": round(total_input / total_output, 1) if total_output > 0 else 0,
        "per_trajectory": {
            "mean": round(float(np.mean(all_costs))), "median": round(float(np.median(all_costs))),
            "std": round(float(np.std(all_costs))), "min": int(np.min(all_costs)), "max": int(np.max(all_costs)),
            "p5": round(float(np.percentile(all_costs, 5))), "p95": round(float(np.percentile(all_costs, 95))),
            "cost_ratio_max_to_min": round(float(np.max(all_costs) / np.min(all_costs)), 1),
        },
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=["multi_model"])   # the dataset with a cost per trajectory
    parser.add_argument("--input", type=Path, default=None, help="folder of cost_per_trajectory.json (default: data/<dataset>/results)")
    parser.add_argument("--output", type=Path, default=None, help="default: data/<dataset>/results")
    arguments = parser.parse_args()
    source = arguments.input or config.stage_dir("results", arguments.dataset)
    target = arguments.output or config.stage_dir("results", arguments.dataset)
    stored = json.loads((source / "cost_per_trajectory.json").read_text(encoding="utf-8"))
    profiles, shares, all_costs = {}, {}, []
    for run, agent in stored["agents"].items():
        profiles[run], costs = run_profile(agent["trajectories"])
        shares[run] = random_subset_shares(costs, config.SEED)
        all_costs.extend(costs.tolist())
    output = {
        "metadata": {"source": "cost_per_trajectory.json", "tokenizer": stored["metadata"]["tokenizer"],
                     "max_observation_tokens": stored["metadata"]["max_observation_tokens"],
                     "subset_percentages": list(SUBSET_PERCENTAGES), "n_monte_carlo_simulations": N_DRAWS,
                     "monte_carlo_seed": config.SEED},
        "aggregate": overall_profile(profiles, np.array(all_costs)),
        "agent_profiles": profiles,
        "savings_empirical": {"aggregate": weighted_aggregate(profiles, shares), "per_agent": shares},
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "cost_profile.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    share = output["savings_empirical"]["aggregate"]["10pct"]["weighted_mean_cost_share_pct"]
    print(f"{arguments.dataset}: wrote cost_profile.json ({len(all_costs)} trajectories; random 10% subsets cost {share}% on average)")


if __name__ == "__main__":
    main()
