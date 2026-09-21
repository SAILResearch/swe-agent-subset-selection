"""Temporal cross-validation splits (paper Section 5.2.2)."""
from itertools import combinations


def temporal_splits(n_runs, window):
    """All (training runs, test runs) with ``window`` training runs and at least one later test run.

    Every combination of ``window`` runs is a training set; the test runs are all runs after its latest run.
    """
    splits = []
    for train in combinations(range(n_runs), window):
        test = list(range(max(train) + 1, n_runs))
        if test:
            splits.append((list(train), test))
    return splits
