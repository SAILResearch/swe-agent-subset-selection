"""Map every trajectory of a run-group dataset to its instance and repository (``repository_map.json``).

    python pipeline/build_repository_map.py --dataset single_setup [--input SANITIZED_DIR] [--output FILE]

Vector files of run groups are named by trajectory id; the selection stage needs the instance (the same instance is
rerun in every run of its group) and the repository. Entry: trajectory id -> {"repo", "outcome", "instance_id"}, read
from the sanitized files ``<N>_runs/run_<i>/unified_<instance>.json``.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def build_map(sanitized_dir):
    """{trajectory id: {"repo", "outcome", "instance_id"}} of all sanitized files below a folder."""
    mapping = {}
    for path in sorted(sanitized_dir.rglob("unified_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("trajectory_id"):
            mapping[record["trajectory_id"]] = {"repo": record.get("repo_name") or "unknown", "outcome": record.get("final_outcome"),
                                                "instance_id": path.stem.replace("unified_", "", 1)}
    return mapping


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(n for n, d in config.DATASETS.items() if "run_groups" in d))
    parser.add_argument("--input", type=Path, default=None, help="sanitized folder (default: data/<dataset>/sanitized)")
    parser.add_argument("--output", type=Path, default=None, help="default: data/<dataset>/repository_map.json")
    arguments = parser.parse_args()
    source = arguments.input or config.stage_dir("sanitized", arguments.dataset)
    mapping = build_map(source)
    target = arguments.output or config.repository_map(arguments.dataset)
    target.write_text(json.dumps(mapping, indent=2))
    print(f"build_repository_map: {len(mapping)} trajectories from {source} -> {target}")


if __name__ == "__main__":
    main()
