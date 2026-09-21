"""Subset selection algorithms (paper Section 4.4).

Baselines draw at random (uniformly or within groups). The five geometric algorithms are deterministic:
Centroid, Facility Location, Medoid (PAM Build), Kennard-Stone and Core/Edge. All functions return
positions within the population. ``candidates`` are the positions an algorithm may choose from.
"""
import numpy as np
from scipy.spatial.distance import cdist

CORE_FRACTION = 0.9   # Core/Edge: share of the slots filled by Centroid; the rest by Kennard-Stone
MAX_CALIBRATION_SWAPS = 50


def uniform_random(rng, n_instances, subset_size):
    """Uniform Random: one draw without replacement."""
    return rng.choice(n_instances, size=subset_size, replace=False).astype(np.int32)


def stratified_random(rng, groups, quotas, subset_size, n_instances):
    """One random draw of each group's quota (groups in string order); a shortfall is drawn from the rest."""
    selected = []
    for key in sorted(quotas, key=str):
        count = min(quotas[key], len(groups[key]))
        if count > 0:
            selected.extend(rng.choice(groups[key], size=count, replace=False).tolist())
    shortfall = subset_size - len(selected)
    if shortfall > 0:
        rest = np.setdiff1d(np.arange(n_instances), selected)
        selected.extend(rng.choice(rest, size=min(shortfall, len(rest)), replace=False).tolist())
    return np.array(selected[:subset_size], dtype=np.int32)


def stability_stratified(groups, quotas, subset_size, n_instances, outcome_variance):
    """Stability-Stratified control: within each group, the instances whose outcome varied least in training."""
    selected = []
    for key in sorted(quotas, key=str):
        members = groups[key]
        count = min(quotas[key], len(members))
        if count > 0:
            selected.extend(members[np.argsort(outcome_variance[members])[:count]].tolist())
    shortfall = subset_size - len(selected)
    if shortfall > 0:
        chosen = set(selected)
        rest = np.array([i for i in range(n_instances) if i not in chosen])
        if len(rest) > 0:
            selected.extend(rest[np.argsort(outcome_variance[rest])[:shortfall]].tolist())
    return np.array(selected[:subset_size], dtype=np.int32)


def centroid(embeddings, candidates, k):
    """Centroid: the k candidates closest to the mean embedding of the candidates."""
    if len(candidates) <= k:
        return candidates[:k].copy()
    distances = np.linalg.norm(embeddings[candidates] - embeddings[candidates].mean(axis=0), axis=1)
    return candidates[np.argsort(distances)[:k]]


def facility_location(similarities, candidates, k):
    """Facility Location: greedy maximisation of the total similarity of all instances to the selected set.

    Candidates are visited in the iteration order of a Python set of their positions and the first candidate
    with the largest gain is taken. This matters for ties, and in particular for the first pick, where every
    candidate has the same (unbounded) gain.
    """
    if k >= len(candidates):
        return candidates[:k].copy()
    coverage = np.full(similarities.shape[0], -np.inf, dtype=np.float64)
    remaining, selected = set(candidates.tolist()), []
    for _ in range(k):
        best, best_gain = -1, -np.inf
        for candidate in remaining:
            gain = np.sum(np.maximum(similarities[:, candidate] - coverage, 0))
            if gain > best_gain:
                best, best_gain = candidate, gain
        selected.append(best)
        remaining.discard(best)
        np.maximum(coverage, similarities[:, best], out=coverage)
    return np.array(selected, dtype=np.int32)


def medoid_pam_build(distances, candidates, k):
    """Medoid (PAM Build): the most central candidate first, then greedily the largest reduction of distances.

    Ties are resolved by the iteration order of a Python set of the candidate positions.
    """
    if k >= len(candidates):
        return candidates[:k].copy()
    remaining = set(candidates.tolist())
    order = np.array(list(remaining))
    # The summation order is kept fixed (row-major sub-matrix): with float32 distances, another order can change the
    # first medoid when two candidates have nearly equal total distance.
    costs = distances[np.ix_(np.arange(distances.shape[0]), order)].sum(axis=0)
    first = order[np.argmin(costs)]
    remaining.discard(first)
    selected, nearest = [first], distances[first].copy()
    for _ in range(k - 1):
        order = np.array(list(remaining))
        gains = np.sum(np.maximum(nearest[:, None] - distances[:, order], 0), axis=0)
        best = order[np.argmax(gains)]
        selected.append(best)
        remaining.discard(best)
        np.minimum(nearest, distances[best], out=nearest)
    return np.array(selected, dtype=np.int32)


def kennard_stone(distances, candidates, k):
    """Kennard-Stone: start from the most central candidate, then repeatedly add the farthest one."""
    if k >= len(candidates):
        return candidates[:k].copy()
    within = distances[np.ix_(candidates, candidates)]
    chosen = [int(np.argmin(within.sum(axis=1)))]
    nearest = within[chosen[0]].copy()
    for _ in range(k - 1):
        chosen.append(int(np.argmax(nearest)))
        np.minimum(nearest, within[chosen[-1]], out=nearest)
    return candidates[np.array(chosen)]


def core_edge(embeddings, candidates, k):
    """Core/Edge: 90% of the slots by Centroid, 10% by Kennard-Stone among the remaining candidates."""
    if len(candidates) <= k:
        return candidates[:k].copy()
    n_core = max(1, int(k * CORE_FRACTION))
    n_edge = k - n_core
    to_center = np.linalg.norm(embeddings[candidates] - embeddings[candidates].mean(axis=0), axis=1)
    core = set(np.argsort(to_center)[:n_core].tolist())
    if n_edge <= 0:
        return candidates[np.array(sorted(core))]
    rest = np.array([i for i in range(len(candidates)) if i not in core])
    if len(rest) <= n_edge:
        return candidates[:k]
    within = cdist(embeddings[candidates[rest]], embeddings[candidates[rest]], "euclidean").astype(np.float32)
    chosen = [int(np.argmax(to_center[rest]))]
    nearest = within[chosen[0]].copy()
    for _ in range(n_edge - 1):
        chosen.append(int(np.argmax(nearest)))
        np.minimum(nearest, within[chosen[-1]], out=nearest)
    return candidates[np.array(sorted(core | set(rest[chosen].tolist())))]


def geometric(algorithm, embeddings, distances, similarities, candidates, k):
    """Run one of the five geometric algorithms (stored method keys: centroid, fl, medoid, ks, core_edge)."""
    if algorithm == "centroid":
        return centroid(embeddings, candidates, k)
    if algorithm == "fl":
        return facility_location(similarities, candidates, k)
    if algorithm == "medoid":
        return medoid_pam_build(distances, candidates, k)
    if algorithm == "ks":
        return kennard_stone(distances, candidates, k)
    if algorithm == "core_edge":
        return core_edge(embeddings, candidates, k)
    raise ValueError(f"Unknown geometric algorithm: {algorithm}")


def geometric_within_groups(algorithm, groups, quotas, embeddings, has_embedding, distances, similarities, n_instances):
    """Embedding-Within-Strata: apply a geometric algorithm inside every group, up to the group's quota.

    Only instances with an embedding are candidates. A group without any embedded instance contributes its
    first members; a group with fewer embedded instances than its quota is completed from its other members.
    """
    selected = []
    for key in sorted(quotas, key=str):
        k, members = quotas[key], groups[key]
        if k <= 0:
            continue
        candidates = members[has_embedding[members]]
        if len(candidates) == 0:
            selected.extend(members[:k].tolist())
        elif k >= len(candidates):
            selected.extend(candidates.tolist())
            selected.extend(members[~has_embedding[members]][:k - len(candidates)].tolist())
        else:
            selected.extend(geometric(algorithm, embeddings, distances, similarities, candidates, k).tolist())
    target = sum(quotas.values())
    if len(selected) < target:
        selected.extend(np.setdiff1d(np.arange(n_instances), selected)[:target - len(selected)].tolist())
    return np.array(selected[:target], dtype=np.int32)


def geometric_on_population(algorithm, embeddings, has_embedding, distances, similarities, k):
    """Pure Embedding: a geometric algorithm on the whole population, without outcome groups."""
    candidates = np.flatnonzero(has_embedding)
    if len(candidates) < k:
        return np.concatenate([candidates, np.flatnonzero(~has_embedding)[:k - len(candidates)]])[:k]
    return geometric(algorithm, embeddings, distances, similarities, candidates, k)


def calibrate_to_pass_counts(selected, train_pass_counts, subset_size):
    """Post-hoc calibration: greedy swaps until the subset's total training pass count matches the population's share."""
    n_instances = len(train_pass_counts)
    pool = np.setdiff1d(np.arange(n_instances), selected)
    target = int(np.round(int(train_pass_counts.sum()) * subset_size / n_instances))
    selected = selected.copy()
    for _ in range(MAX_CALIBRATION_SWAPS):
        gap = target - int(train_pass_counts[selected].sum())
        if gap == 0:
            break
        direction = 1 if gap > 0 else -1
        selected_order = np.argsort(direction * train_pass_counts[selected])
        pool_order = np.argsort(-direction * train_pass_counts[pool])
        swap = next(((s, p) for s in selected[selected_order] for p in pool[pool_order]
                     if direction * (train_pass_counts[p] - train_pass_counts[s]) > 0
                     and abs(gap - (train_pass_counts[p] - train_pass_counts[s])) <= abs(gap)), None)
        if swap is None:
            break
        outgoing, incoming = swap
        selected = np.append(selected[selected != outgoing], incoming)
        pool = np.append(pool[pool != incoming], outgoing)
    return selected
