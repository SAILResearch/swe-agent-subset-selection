"""Loading of run outcomes, synthetic distributions and embedding file indexes.

Vector files are named ``<OUTCOME>_<repository>_<instance id>_<ts|pooled>.npy`` with OUTCOME = SUCCESS or
FAIL (see embed.py; PASS and TRUE are also read as passing). The pass/fail outcome of an instance on a run is read from this prefix.
"""
import json
import re
from pathlib import Path

import numpy as np

OUTCOME_PREFIX = re.compile(r"^(PASS|FAIL|SUCCESS|TRUE)_")
VARIANT_SUFFIX = re.compile(r"_(ts|pooled)\.npy$")
PASSING_PREFIXES = ("PASS", "SUCCESS", "TRUE")
DISTRIBUTION_LEVEL = re.compile(r"rate_(\d+)_run")
MIN_DISTRIBUTION_SIZE = 50


def _stem(file_name):
    """Vector file name without outcome prefix and variant suffix: ``<repository>_<instance id>``."""
    return VARIANT_SUFFIX.sub("", OUTCOME_PREFIX.sub("", file_name))


def instance_id(file_name):
    """Instance id of a vector file."""
    parts = _stem(file_name).split("_", 1)
    return parts[1] if len(parts) > 1 else parts[0]


def repository(file_name):
    """Repository of a vector file."""
    stem = _stem(file_name)
    return stem.split("_")[0] if stem else "unknown"


def load_outcomes(timeseries_dir, wanted_ids):
    """Pass/fail matrix of the instances that have a trajectory in every run.

    Returns (matrix[instances, runs] of 0/1 float32, instance ids, run names, repositories); instances and
    runs are sorted by name, so run order is chronological for date-prefixed run folders.
    """
    outcomes, repositories = {}, {}
    run_names = sorted(d.name for d in Path(timeseries_dir).iterdir() if d.is_dir())
    for run in run_names:
        for path in sorted((Path(timeseries_dir) / run).glob("*_ts.npy")):
            identifier = instance_id(path.name)
            if identifier in wanted_ids:
                outcomes.setdefault(identifier, {})[run] = 1.0 if path.name.startswith(PASSING_PREFIXES) else 0.0
                repositories.setdefault(identifier, repository(path.name))
    seen_runs = sorted({run for runs in outcomes.values() for run in runs})
    complete = sorted(i for i, runs in outcomes.items() if len(runs) == len(seen_runs))
    matrix = np.array([[outcomes[i][run] for run in seen_runs] for i in complete], dtype=np.float32)
    return matrix, np.array(complete), seen_runs, np.array([repositories[i] for i in complete])


def distribution_instance_ids(path):
    """All instance ids that occur in the synthetic distributions file."""
    with open(path) as handle:
        return {i for entry in json.load(handle).values() for i in entry["instances"]}


def load_distributions(path, instance_ids):
    """Synthetic distributions as (index, difficulty level, instance index array), in file order.

    ``index`` is the position among the distributions that are kept (those with a difficulty level in their name and
    at least ``MIN_DISTRIBUTION_SIZE`` known instances); it enters the random seeds, so it does not change when only
    some distributions are evaluated.
    """
    with open(path) as handle:
        stored = json.load(handle)
    position = {identifier: i for i, identifier in enumerate(instance_ids)}
    distributions = []
    for name, entry in stored.items():
        level = DISTRIBUTION_LEVEL.search(name)
        members = np.array([position[i] for i in entry["instances"] if i in position])
        if level is not None and len(members) >= MIN_DISTRIBUTION_SIZE:
            distributions.append((len(distributions), int(level.group(1)), members))
    return distributions


def embedding_index(vector_dir, variant):
    """{instance id: {run name: vector file}} for one embedding variant ("pooled" or "ts")."""
    index = {}
    for run_dir in sorted(Path(vector_dir).iterdir()):
        if run_dir.is_dir():
            for path in sorted(run_dir.glob(f"*_{variant}.npy")):
                index.setdefault(instance_id(path.name), {})[run_dir.name] = str(path)
    return index
