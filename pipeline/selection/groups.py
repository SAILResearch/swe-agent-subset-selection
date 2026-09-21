"""Grouping of instances (strata) and proportional allocation of subset slots."""
import numpy as np


def proportional_quotas(groups, subset_size):
    """Slots per group, proportional to group size (largest-remainder method), keys in string order."""
    total = sum(len(members) for members in groups.values())
    if total == 0:
        return {}
    keys = sorted(groups, key=str)
    quotas = {key: min(int(np.floor(subset_size * len(groups[key]) / total)), len(groups[key])) for key in keys}
    remainder = subset_size - sum(quotas.values())
    if remainder > 0:
        fractions = [((subset_size * len(groups[key]) / total) % 1, key) for key in keys if len(groups[key]) > quotas[key]]
        fractions.sort(reverse=True, key=lambda item: item[0])
        for _, key in fractions[:remainder]:
            quotas[key] += 1
    return quotas


def pass_counts(population, train_runs):
    """Number of training runs each instance passed."""
    return population[:, train_runs].sum(axis=1)


def difficulty_groups(population, train_runs):
    """Two groups: instances solved in at least half of the training runs (1) and the others (0)."""
    labels = difficulty_labels(population, train_runs)
    return {value: np.flatnonzero(labels == value) for value in (0, 1) if np.any(labels == value)}


def difficulty_labels(population, train_runs):
    """1 for instances solved in at least half (rounded up) of the training runs, else 0."""
    return (pass_counts(population, train_runs) >= int(np.ceil(len(train_runs) / 2))).astype(int)


def consistency_groups(population, train_runs):
    """One group per exact number of passed training runs."""
    counts = pass_counts(population, train_runs)
    return {c: np.flatnonzero(counts == c) for c in range(len(train_runs) + 1) if np.any(counts == c)}


def outcome_groups(population, train_runs):
    """Outcome groups of the geometric methods: pass/fail with one training run, pass counts otherwise."""
    return difficulty_groups(population, train_runs) if len(train_runs) == 1 else consistency_groups(population, train_runs)


def repository_groups(repositories):
    """One group per repository."""
    return {name: np.flatnonzero(repositories == name) for name in np.unique(repositories)}


def crossed_groups(repositories, labels):
    """One group per (repository, label) pair."""
    groups = {}
    for position, (name, label) in enumerate(zip(repositories, labels)):
        groups.setdefault(f"{name}_{int(label)}", []).append(position)
    return {key: np.array(members) for key, members in groups.items()}
