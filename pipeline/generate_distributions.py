"""Synthetic distributions of a dataset's source run (paper Section 5.2.1).

    python pipeline/generate_distributions.py --dataset multi_model [--input PARSED_DIR] [--output FILE]

Reads the parsed trajectories of the source run (the first run of the dataset; default data/<dataset>/parsed) and
writes 1,000 distributions of 250 instances each to ``FILE`` (default
data/<dataset>/distributions/generated_distributions.json): for each of the nine difficulty levels (10% to 90%
resolve rate on the source run), resolved and unresolved instances are drawn without replacement in the
proportion of the level.

The draws index into the list of resolved and of unresolved instances, so the result depends on the order of
those lists. Files are listed in sorted order here, which gives the same distributions on every machine. The
distributions files of the study (``results/<dataset>/synthetic_distributions.json``) rest on the directory listing
order of the machine that generated them: they have the same levels, sizes, resolved counts and pairwise overlaps as
the output of this script, with other instance ids. ``reproduce.py`` passes the study's files to the selection
stage, and this script writes to another file name.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

N_DISTRIBUTIONS = 1000
DISTRIBUTION_SIZE = 250
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def source_outcomes(run_dir):
    """(instance id, resolved) of every parsed trajectory of a run, in sorted file order."""
    outcomes = []
    for path in sorted(Path(run_dir).glob("*.json")):
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        outcomes.append((trajectory.get("trajectory_id", "unknown"), trajectory.get("final_outcome", False)))
    return outcomes


def draw_distribution(rng, resolved_ids, unresolved_ids, level):
    """One distribution at a difficulty level: instance ids in shuffled order and the resolve rate obtained.

    When a list is too short for the level, all of it is used and the distribution is smaller than 250.
    """
    target_resolved = int(DISTRIBUTION_SIZE * level)
    n_resolved = min(target_resolved, len(resolved_ids))
    n_unresolved = min(DISTRIBUTION_SIZE - target_resolved, len(unresolved_ids))
    size = n_resolved + n_unresolved
    rate = n_resolved / size if size > 0 else 0.0
    members = list(rng.choice(resolved_ids, size=n_resolved, replace=False)) + list(rng.choice(unresolved_ids, size=n_unresolved, replace=False))
    rng.shuffle(members)
    return {"instances": [str(i) for i in members], "resolve_rate": round(rate, 4)}


def generate(outcomes, seed=config.SEED):
    """All distributions, keyed ``rate_<level>_run_<i>``, from (instance id, resolved) pairs in a given order."""
    resolved_ids = np.array([i for i, resolved in outcomes if resolved], dtype=object)
    unresolved_ids = np.array([i for i, resolved in outcomes if not resolved], dtype=object)
    rng = np.random.default_rng(seed)
    per_level, remainder = divmod(N_DISTRIBUTIONS, len(LEVELS))
    distributions = {}
    for index, level in enumerate(LEVELS):
        for i in range(per_level + (1 if index < remainder else 0)):
            distributions[f"rate_{int(level * 100)}_run_{i}"] = draw_distribution(rng, resolved_ids, unresolved_ids, level)
    return distributions


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True,
                        choices=sorted(n for n, d in config.DATASETS.items() if d["protocol"] == "synthetic_distributions"))
    parser.add_argument("--input", type=Path, default=None, help="parsed trajectories (default: data/<dataset>/parsed)")
    parser.add_argument("--output", type=Path, default=None, help="default: data/<dataset>/distributions/generated_distributions.json")
    arguments = parser.parse_args()
    source_run = config.runs_of(arguments.dataset)[0]
    run_dir = (arguments.input or config.stage_dir("parsed", arguments.dataset)) / source_run
    target = arguments.output or config.stage_dir("distributions", arguments.dataset) / "generated_distributions.json"
    outcomes = source_outcomes(run_dir)
    if not outcomes:
        raise SystemExit(f"No parsed trajectories found in {run_dir}")
    distributions = generate(outcomes)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(distributions, indent=2), encoding="utf-8")
    resolved = sum(1 for _, r in outcomes if r)
    print(f"{arguments.dataset}: wrote {len(distributions)} distributions to {target} "
          f"(source run {source_run}: {resolved} of {len(outcomes)} resolved)")


if __name__ == "__main__":
    main()
