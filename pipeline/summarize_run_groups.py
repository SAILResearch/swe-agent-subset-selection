"""Run groups of a run-group dataset: instances and resolve rate of every run (``run_groups.json``).

    python pipeline/summarize_run_groups.py --dataset single_setup [--input VECTOR_DIR] [--repository-map FILE] [--output RESULT_DIR]

The numbers are those of the selection stage: instances with a vector in every run of their group, outcomes from the
vector file names, runs in run-number order.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from protocols import run_groups  # noqa: E402

DESCRIPTION = ("Single-setup run groups: number of reruns per instance, number of instances, and the resolve rate (%) of each run "
               "(resolved instances / instances). The 10-run group is results/single_setup/aggregated_results.json; groups 5 to 9 are "
               "in run_group_<N>/.")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(n for n, d in config.DATASETS.items() if "run_groups" in d))
    parser.add_argument("--input", type=Path, default=None, help="vector folder (default: data/<dataset>/vectors)")
    parser.add_argument("--repository-map", type=Path, default=None, help="repository_map.json (default: data/<dataset>/repository_map.json)")
    parser.add_argument("--output", type=Path, default=None, help="result folder (default: data/<dataset>/results)")
    arguments = parser.parse_args()
    vectors = arguments.input or config.stage_dir("vectors", arguments.dataset)
    repository_map = json.loads((arguments.repository_map or config.repository_map(arguments.dataset)).read_text())
    summary = {}
    for size in config.dataset(arguments.dataset)["run_groups"]:
        if not (vectors / "timeseries" / f"{size}_runs").is_dir():
            raise SystemExit(f"summarize_run_groups: the summary covers all run groups; no vectors of the {size}-run group in {vectors}")
        outcomes, _, run_names, _ = run_groups.load_group(vectors / "timeseries" / f"{size}_runs", repository_map)
        by_number = sorted(range(len(run_names)), key=lambda column: int(run_names[column].split("_")[1]))
        rates = [round(float(np.mean(outcomes[:, column])) * 100, 4) for column in by_number]
        summary[str(size)] = {"n_runs": size, "n_instances": int(outcomes.shape[0]), "resolve_rate_pct_per_run": rates,
                              "resolve_rate_pct_min": min(rates), "resolve_rate_pct_max": max(rates)}
    target = arguments.output or config.stage_dir("results", arguments.dataset)
    target.mkdir(parents=True, exist_ok=True)
    (target / "run_groups.json").write_text(json.dumps({"description": DESCRIPTION, "run_groups": summary}, indent=1) + "\n")
    print(f"summarize_run_groups: wrote {target / 'run_groups.json'}")


if __name__ == "__main__":
    main()
