"""Validity checks of the synthetic distributions on the runs they were not built from (paper Sections 5.2.1, 8.3).

    python pipeline/distribution_validity.py --dataset multi_model [--input PARSED_DIR] [--distributions-file FILE]
                                             [--output RESULT_DIR]

Reads the outcome of every parsed trajectory (default data/<dataset>/parsed) and the distributions file (default
data/<dataset>/distributions/synthetic_distributions.json) and writes to ``RESULT_DIR`` (default
data/<dataset>/results):

    cross_run_resolve_rates.{json,csv}                   resolve rate of every distribution on every run
    difficulty_level_validity_per_distribution.csv       the same as one row per distribution, source run first
    instance_churn_per_distribution.csv                  per distribution and pair of runs: share of instances whose
                                                         outcome differs, by direction, and the change of the resolve rate

The source run is the first run of the dataset; the distributions were built from its outcomes.
"""
import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def run_outcomes(run_dir):
    """{instance id: resolved} of the parsed trajectories of one run."""
    outcomes = {}
    for path in sorted(Path(run_dir).glob("*.json")):
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        outcomes[trajectory.get("trajectory_id", "unknown")] = bool(trajectory.get("final_outcome", False))
    return outcomes


def level_of(name):
    """Difficulty level of a distribution name: ``rate_10_run_0`` -> 10."""
    return int(name.split("_")[1])


def cross_run_rows(distributions, outcomes, run_order):
    """One row per (distribution, run): resolve rate over the distribution's instances that the run contains."""
    rows = []
    for name, entry in distributions.items():
        instances = entry["instances"]
        for run in run_order:
            matched = [outcomes[run][i] for i in instances if i in outcomes[run]]
            resolved = sum(1 for value in matched if value)
            rows.append({"distribution": name, "bucket": level_of(name), "source_resolve_rate": entry["resolve_rate"],
                         "eval_run": run, "eval_resolve_rate": resolved / len(matched) if matched else None,
                         "matched_instances": len(matched), "total_instances": len(instances),
                         "coverage": len(matched) / len(instances) if instances else 0})
    return rows


def validity_rows(distributions, outcomes, runs):
    """One row per distribution: its level and its resolve rate on every run."""
    rows = []
    for name, entry in distributions.items():
        row = {"dist": name, "bucket": level_of(name)}
        for run in runs:
            values = [outcomes[run][i] for i in entry["instances"] if i in outcomes[run]]
            row[run] = (sum(values) / len(values)) if values else float("nan")
        rows.append(row)
    return rows


def churn_rows(distributions, outcomes, runs):
    """One row per (distribution, pair of runs): gross outcome changes, by direction, and net resolve-rate change."""
    rows = []
    for name, entry in distributions.items():
        for first, second in combinations(runs, 2):
            common = [i for i in entry["instances"] if i in outcomes[first] and i in outcomes[second]]
            if not common:
                continue
            n = len(common)
            a = np.array([outcomes[first][i] for i in common], dtype=bool)
            b = np.array([outcomes[second][i] for i in common], dtype=bool)
            p2f = int(np.sum(a & ~b))
            f2p = int(np.sum(~a & b))
            gross = (p2f + f2p) / n
            net = abs(a.mean() - b.mean())
            rows.append({"dist": name, "bucket": level_of(name), "pair": f"{first} | {second}", "n": n, "gross_flip": gross,
                         "p2f": p2f / n, "f2p": f2p / n, "net_rate_change": net, "cancelled": gross - net})
    return rows


def write_csv(path, rows):
    """Write rows (dicts with the same keys) as csv; floats in their shortest exact form, missing values empty."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(rows[0].keys())
        for row in rows:
            writer.writerow(["" if value is None or value != value else
                             repr(float(value)) if isinstance(value, (float, np.floating)) else value for value in row.values()])


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True,
                        choices=sorted(n for n, d in config.DATASETS.items() if d["protocol"] == "synthetic_distributions"))
    parser.add_argument("--input", type=Path, default=None, help="parsed trajectories (default: data/<dataset>/parsed)")
    parser.add_argument("--distributions-file", type=Path, default=None, help="default: data/<dataset>/distributions/synthetic_distributions.json")
    parser.add_argument("--output", type=Path, default=None, help="default: data/<dataset>/results")
    arguments = parser.parse_args()
    parsed = arguments.input or config.stage_dir("parsed", arguments.dataset)
    distributions_file = arguments.distributions_file or config.stage_dir("distributions", arguments.dataset) / "synthetic_distributions.json"
    target = arguments.output or config.stage_dir("results", arguments.dataset)
    runs = config.runs_of(arguments.dataset)
    outcomes = {}
    for run in runs:
        outcomes[run] = run_outcomes(parsed / run)
        if not outcomes[run]:
            raise SystemExit(f"No parsed trajectories found in {parsed / run}")
    distributions = json.loads(distributions_file.read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)

    run_order = runs[1:] + runs[:1]   # the source run comes last in the cross-run files
    cross = cross_run_rows(distributions, outcomes, run_order)
    write_csv(target / "cross_run_resolve_rates.csv", cross)
    pool = {run: {"total": len(outcomes[run]), "resolved": sum(1 for v in outcomes[run].values() if v),
                  "rate": sum(1 for v in outcomes[run].values() if v) / len(outcomes[run])} for run in run_order}
    per_distribution = {}
    for row in cross:
        per_distribution.setdefault(row["distribution"], {})[row["eval_run"]] = {"resolve_rate": row["eval_resolve_rate"],
                                                                                 "matched": row["matched_instances"]}
    per_distribution = {name: per_distribution[name] for name in sorted(per_distribution)}
    (target / "cross_run_resolve_rates.json").write_text(json.dumps({"pool_stats": pool, "per_distribution": per_distribution}, indent=2),
                                                         encoding="utf-8")
    write_csv(target / "difficulty_level_validity_per_distribution.csv", validity_rows(distributions, outcomes, runs))
    churn = churn_rows(distributions, outcomes, runs)
    write_csv(target / "instance_churn_per_distribution.csv", churn)
    print(f"{arguments.dataset}: wrote cross_run_resolve_rates.{{json,csv}}, difficulty_level_validity_per_distribution.csv and "
          f"instance_churn_per_distribution.csv ({len(distributions)} distributions, {len(runs)} runs; "
          f"mean share of instances with a different outcome between two runs {100 * np.mean([r['gross_flip'] for r in churn]):.2f}%)")


if __name__ == "__main__":
    main()
