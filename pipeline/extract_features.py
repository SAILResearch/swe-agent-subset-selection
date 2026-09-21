"""Trajectory features for the clustering configurations (paper Section 7.3).

    python pipeline/extract_features.py --dataset multi_model [--input PARSED_DIR] [--output FEATURE_DIR]

Reads the parsed trajectories ``PARSED_DIR/<run>/unified_<instance>.json`` (default data/<dataset>/parsed) and writes
``trajectory_features.npz`` (one float32 array [instances, 10] per run, rows in sorted instance-id order, NaN rows
for instances a run does not contain or whose trajectory has no steps) and ``trajectory_features_meta.json``
(instance ids, run names, feature names) to ``FEATURE_DIR`` (default data/<dataset>/features).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

FEATURE_NAMES = ("n_steps", "total_edit_log", "n_edit_steps", "max_edit_log", "n_errors", "n_unique_files", "n_test_steps",
                 "frac_bash", "frac_edit", "mean_obs_len_log")


def trajectory_features(trajectory):
    """The ten features of one parsed trajectory, in the order of FEATURE_NAMES; NaN when it has no steps."""
    steps = trajectory.get("steps", [])
    n = len(steps)
    if n == 0:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
    edits = [step.get("edit_size", 0) or 0 for step in steps]
    files = set()
    n_errors = n_bash = n_editor = n_test = 0
    observation_lengths = []
    for step in steps:
        if step.get("error_message", ""):
            n_errors += 1
        header = step.get("action_header", "")
        if header == "execute_bash":
            n_bash += 1
        elif header == "str_replace_editor":
            n_editor += 1
        files.update(step.get("files_touched", []))
        if step.get("tests_run"):
            n_test += 1
        observation_lengths.append(np.log1p(len(step.get("observation", "") or "")))
    return np.array([n, np.log1p(sum(edits)), sum(1 for e in edits if e > 0), np.log1p(max(edits)) if edits else 0.0,
                     n_errors, len(files), n_test, n_bash / n, n_editor / n, float(np.mean(observation_lengths))],
                    dtype=np.float32)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True,   # clustering is evaluated on the datasets with synthetic distributions
                        choices=sorted(n for n, d in config.DATASETS.items() if d["protocol"] == "synthetic_distributions"))
    parser.add_argument("--input", type=Path, default=None, help="parsed trajectories (default: data/<dataset>/parsed)")
    parser.add_argument("--output", type=Path, default=None, help="default: data/<dataset>/features")
    arguments = parser.parse_args()
    parsed = arguments.input or config.stage_dir("parsed", arguments.dataset)
    target = arguments.output or config.stage_dir("features", arguments.dataset)
    runs = sorted(config.runs_of(arguments.dataset))
    files = {}
    for run in runs:
        files[run] = {path.stem.replace("unified_", "", 1): path for path in sorted((parsed / run).glob("unified_*.json"))}
        if not files[run]:
            raise SystemExit(f"No parsed trajectories found in {parsed / run}")
    instance_ids = sorted(set().union(*files.values()))
    row_of = {identifier: row for row, identifier in enumerate(instance_ids)}
    arrays, stats = {}, {}
    for run in runs:
        arrays[run] = np.full((len(instance_ids), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
        for identifier, path in files[run].items():
            arrays[run][row_of[identifier]] = trajectory_features(json.loads(path.read_text(encoding="utf-8")))
        # "errors" belongs to the layout of the meta file; a trajectory that cannot be read stops the script.
        stats[run] = {"found": len(files[run]), "missing": len(instance_ids) - len(files[run]), "errors": 0}
        print(f"  {run}: {len(files[run])} trajectories")
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "trajectory_features.npz", **arrays)
    meta = {"instance_ids": instance_ids, "run_names": runs, "feature_names": list(FEATURE_NAMES), "n_features": len(FEATURE_NAMES),
            "n_instances": len(instance_ids), "stats": {"per_run": stats}}
    (target / "trajectory_features_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{arguments.dataset}: wrote trajectory_features.npz and trajectory_features_meta.json "
          f"({len(instance_ids)} instances, {len(runs)} runs) to {target}")


if __name__ == "__main__":
    main()
