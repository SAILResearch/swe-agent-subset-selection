"""Unsupervised clustering used by the Clustering ablation (paper Section 6.3): K-Means and HDBSCAN.

Feature spaces: the pooled and time-series trajectory embeddings, all ten trajectory features, and the five
count features (steps, edit steps, errors, unique files, test steps).
"""
import json
import warnings

import numpy as np
from scipy.spatial.distance import cdist

MIN_INSTANCES_TO_CLUSTER = 20
MAX_K = 15
KMEANS_SEED = 42
COUNT_FEATURE_COLUMNS = [0, 2, 4, 5, 6]   # n_steps, n_edit_steps, n_errors, n_unique_files, n_test_steps


def load_trajectory_features(feature_file, meta_file):
    """{"arrays": {run: features[instances, 10]}, "instance_ids": [...]} of the pre-extracted trajectory features."""
    stored = np.load(feature_file, allow_pickle=False)
    with open(meta_file) as handle:
        meta = json.load(handle)
    return {"arrays": {run: stored[run] for run in meta["run_names"] if run in stored}, "instance_ids": meta["instance_ids"]}


def averaged_features(features, instance_ids, train_runs):
    """Trajectory features of the given instances averaged over the training runs (NaN where missing)."""
    row_of = {identifier: row for row, identifier in enumerate(features["instance_ids"])}
    runs = [run for run in train_runs if run in features["arrays"]]
    if not runs:
        return None
    width = next(iter(features["arrays"].values())).shape[1]
    stacked = []
    for run in runs:
        values = np.full((len(instance_ids), width), np.nan, dtype=np.float32)
        for position, identifier in enumerate(instance_ids):
            if identifier in row_of:
                values[position] = features["arrays"][run][row_of[identifier]]
        stacked.append(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(stacked), axis=0)


def standardize(values):
    """Z-score per column; constant columns are left unscaled and missing values become 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean, spread = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    spread[spread < 1e-8] = 1.0
    return np.nan_to_num((values - mean) / spread, nan=0.0)


def kmeans_labels(values):
    """K-Means labels with K from 2 to min(sqrt(n/2), 15) chosen by the silhouette score."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n = values.shape[0]
    best_score, best_labels = -1.0, np.zeros(n, dtype=np.int32)
    for k in range(2, min(int(np.sqrt(n / 2)), MAX_K) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=KMEANS_SEED, max_iter=300).fit_predict(values)
        if len(set(labels)) >= 2:
            score = silhouette_score(values, labels)
            if score > best_score:
                best_score, best_labels = score, labels.copy()
    return best_labels


def hdbscan_labels(values):
    """HDBSCAN labels; noise points are assigned to the nearest cluster centre."""
    import hdbscan
    min_cluster_size = max(5, len(values) // 20)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=max(2, min_cluster_size // 2)).fit_predict(values)
    clusters = sorted(set(labels) - {-1})
    if not clusters:
        return np.zeros(len(labels), dtype=np.int32)
    noise = labels == -1
    if np.any(noise):
        centres = np.array([values[labels == c].mean(axis=0) for c in clusters])
        labels[noise] = np.array([clusters[i] for i in np.argmin(cdist(values[noise], centres), axis=1)])
    return labels


def cluster_labels(method, values):
    """Labels of one clustering method ("kmeans" or "hdbscan") on standardized values."""
    return kmeans_labels(standardize(values)) if method == "kmeans" else hdbscan_labels(standardize(values))


def population_clusters(values, usable, method):
    """{cluster: positions} from clustering all usable instances of the population."""
    positions = np.flatnonzero(usable)
    if len(positions) < MIN_INSTANCES_TO_CLUSTER:
        return {0: positions} if len(positions) > 0 else {}
    labels = cluster_labels(method, values[positions])
    return {c: positions[labels == c] for c in np.unique(labels)}


def clusters_within_groups(groups, values, usable, method):
    """{"<group>_k<cluster>": positions}: clustering inside every outcome group (small groups stay whole)."""
    compound = {}
    for key, members in groups.items():
        positions = members[usable[members]]
        if len(positions) < MIN_INSTANCES_TO_CLUSTER:
            compound[f"{key}_k0"] = members
            continue
        labels = cluster_labels(method, values[positions])
        for c in np.unique(labels):
            compound[f"{key}_k{c}"] = positions[labels == c]
    return compound
