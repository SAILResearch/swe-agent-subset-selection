"""Embeddings of the instances of a population, averaged over the training runs.

Pooled variant: each trajectory is a vector of 3,840 values; the instance embedding is the mean of its
pooled vectors over the training runs.

Time-series variant: each trajectory is a matrix of T steps x 768 values, with T differing between
trajectories. Every matrix is brought to a common length of L = 8 steps by linear interpolation along the
step axis (a single-step trajectory is repeated), flattened to 8 x 768 = 6,144 values, and the instance
embedding is the mean of these vectors over the training runs.
"""
import numpy as np
from scipy.spatial.distance import cdist

TIME_SERIES_LENGTH = 8


def resample_steps(matrix, length=TIME_SERIES_LENGTH):
    """Linear interpolation of a T x d step matrix to ``length`` x d."""
    n_steps = matrix.shape[0]
    if n_steps == 1:
        return np.repeat(matrix, length, axis=0)
    position = np.linspace(0, n_steps - 1, length)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, n_steps - 1)
    weight = (position - lower)[:, None]
    return (1.0 - weight) * matrix[lower] + weight * matrix[upper]


def trajectory_vector(path, variant):
    """Fixed-length float32 vector of one stored trajectory embedding."""
    stored = np.load(path).astype(np.float32)
    if variant == "ts":
        return resample_steps(stored).flatten()
    return stored.flatten()


def averaged_embeddings(index, instance_ids, train_runs, variant):
    """(embeddings[n, dim] float32, has_embedding[n]) of the given instances.

    ``index`` is an ``embedding_index``; an instance without a vector in any training run gets a zero row
    and ``has_embedding`` False.
    """
    vectors = [[trajectory_vector(index[i][run], variant) for run in train_runs if run in index.get(i, {})]
               for i in instance_ids]
    has_embedding = np.array([bool(v) for v in vectors])
    if not has_embedding.any():
        return None, has_embedding
    dimension = len(next(v for v in vectors if v)[0])
    total = np.zeros((len(instance_ids), dimension), np.float64)
    for row, instance_vectors in enumerate(vectors):
        if instance_vectors:
            total[row] = np.mean(instance_vectors, axis=0)
    embeddings = np.zeros((len(instance_ids), dimension), np.float32)
    embeddings[has_embedding] = total[has_embedding].astype(np.float32)
    return embeddings, has_embedding


def population_embeddings(index, instance_ids, train_runs, variant):
    """Embeddings, availability mask, Euclidean distance matrix and similarity matrix of a population."""
    embeddings, has_embedding = averaged_embeddings(index, instance_ids, train_runs, variant)
    if embeddings is None:
        embeddings = np.zeros((len(instance_ids), 1), np.float32)
    distances, similarities = distance_and_similarity(embeddings, has_embedding)
    return embeddings, has_embedding, distances, similarities


def distance_and_similarity(embeddings, has_embedding):
    """Euclidean distances and RBF similarities (bandwidth: median positive distance)."""
    rows = embeddings.copy()
    rows[~has_embedding] = 0.0
    distances = cdist(rows, rows, "euclidean").astype(np.float32)
    bandwidth = np.median(distances[distances > 0]) if np.any(distances > 0) else 1.0
    return distances, np.exp(-(distances ** 2) / (2.0 * bandwidth ** 2))
