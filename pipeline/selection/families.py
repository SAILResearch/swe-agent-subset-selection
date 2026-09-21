"""Method families of the study. Each family evaluates all of its methods on one temporal split.

A family is a function ``evaluate(split_context) -> {subset_key: {method: split record}}`` plus the list of
its method keys. Method keys are the ones stored in the result files:

    baselines          random, repo_stratified, difficulty_stratified, repo_difficulty_stratified,
                       consistency_stratified, repo_consistency_stratified (the last two need two training runs),
                       stability_stratified (the Stability-Stratified control of Finding 3.4)
    embedding_strata   <algorithm>_<pooled|ts>            Embedding-Within-Strata
    pure_embedding     <algorithm>_<pooled|ts>[_cal]      Pure Embedding, without / with post-hoc calibration
    clustering         <kmeans|hdbscan>_<space>_<strata|nostrata>
    shortlist          <fl|centroid|ks>_<pooled|ts>_<strata|nostrata>_<15x|20x>

Algorithm keys: centroid (Centroid), fl (Facility Location), medoid (Medoid, PAM Build), ks (Kennard-Stone),
core_edge (Core/Edge).
"""
import numpy as np

from . import algorithms, clustering, groups

VARIANTS = ("pooled", "ts")
WITHIN_STRATA_ALGORITHMS = ("fl", "medoid", "centroid", "ks", "core_edge")
PURE_ALGORITHMS = ("centroid", "fl", "medoid", "ks", "core_edge")
SHORTLIST_ALGORITHMS = ("fl", "centroid", "ks")
SHORTLIST_RATIOS = (1.5, 2.0)
CLUSTER_METHODS = ("kmeans", "hdbscan")
FEATURE_SPACES = ("pooled", "ts", "features", "features_top5")
GROUPING_MODES = ("strata", "nostrata")
STOCHASTIC_BASELINES = ("random", "repo_stratified", "difficulty_stratified", "repo_difficulty_stratified")
TWO_RUN_BASELINES = ("consistency_stratified", "repo_consistency_stratified")
CONTROL = "stability_stratified"


def subset_key(fraction):
    """Result key of a subset fraction, e.g. 0.05 -> "5pct"."""
    return f"{int(fraction * 100)}pct"


def wanted(context, method):
    """True when ``method`` is to be evaluated (``context["methods"]`` is None for the whole family)."""
    return context.get("methods") is None or method in context["methods"]


def subset_size(n_instances, fraction):
    """Number of selected instances for a subset fraction."""
    return max(1, int(n_instances * fraction))


# Baselines

def baseline_methods(window):
    """Baseline method keys available with ``window`` training runs (control last)."""
    stochastic = STOCHASTIC_BASELINES + (TWO_RUN_BASELINES if window >= 2 else ())
    return list(stochastic) + [CONTROL]


def baseline_groups(population, repositories, train_runs):
    """Groups of every stratified baseline for one split."""
    difficulty = groups.difficulty_groups(population, train_runs)
    result = {"repo_stratified": groups.repository_groups(repositories), "difficulty_stratified": difficulty,
              CONTROL: difficulty,
              "repo_difficulty_stratified": groups.crossed_groups(repositories, groups.difficulty_labels(population, train_runs))}
    if len(train_runs) >= 2:
        result["consistency_stratified"] = groups.consistency_groups(population, train_runs)
        result["repo_consistency_stratified"] = groups.crossed_groups(repositories, groups.pass_counts(population, train_runs))
    return result


def evaluate_baselines(context):
    """The stochastic baselines (``n_seeds`` seeded draws each) and the deterministic Stability-Stratified control."""
    population, n = context["population"], context["population"].shape[0]
    strata = baseline_groups(population, context["repositories"], context["train_runs"])
    variance = np.var(population[:, context["train_runs"]], axis=1)
    records = {}
    for fraction in context["subset_fractions"]:
        size, cell = subset_size(n, fraction), {}
        for method in baseline_methods(len(context["train_runs"]))[:-1]:
            if not wanted(context, method):
                continue
            if method == "random":
                draw = lambda rng, _size=size: algorithms.uniform_random(rng, n, _size)   # noqa: E731
            else:
                quotas = groups.proportional_quotas(strata[method], size)
                draw = lambda rng, _g=strata[method], _q=quotas, _size=size: algorithms.stratified_random(rng, _g, _q, _size, n)   # noqa: E731
            cell[method] = context["stochastic"](draw)
        if wanted(context, CONTROL):
            quotas = groups.proportional_quotas(strata[CONTROL], size)
            cell[CONTROL] = context["deterministic"](algorithms.stability_stratified(strata[CONTROL], quotas, size, n, variance))
        records[subset_key(fraction)] = cell
    return records


# Geometric families

def within_strata_methods(_window=None):
    """Embedding-Within-Strata method keys."""
    return [f"{algorithm}_{variant}" for variant in VARIANTS for algorithm in WITHIN_STRATA_ALGORITHMS]


def evaluate_within_strata(context):
    """Five geometric algorithms inside the outcome groups, for both embedding variants."""
    population, n = context["population"], context["population"].shape[0]
    strata = groups.outcome_groups(population, context["train_runs"])
    records = {}
    for fraction in context["subset_fractions"]:
        size = subset_size(n, fraction)
        quotas, cell = groups.proportional_quotas(strata, size), {}
        for variant in VARIANTS:
            for algorithm in WITHIN_STRATA_ALGORITHMS:
                if not wanted(context, f"{algorithm}_{variant}"):
                    continue
                vectors, has_embedding, distances, similarities = context["embeddings"](variant)
                selected = algorithms.geometric_within_groups(algorithm, strata, quotas, vectors, has_embedding, distances, similarities, n)
                cell[f"{algorithm}_{variant}"] = context["deterministic"](selected)
        records[subset_key(fraction)] = cell
    return records


def pure_embedding_methods(_window=None):
    """Pure Embedding method keys (without and with calibration)."""
    return [f"{algorithm}_{variant}{suffix}" for variant in sorted(VARIANTS) for algorithm in PURE_ALGORITHMS for suffix in ("", "_cal")]


def evaluate_pure_embedding(context):
    """Five geometric algorithms on the whole population, each also with post-hoc pass-count calibration."""
    population, n = context["population"], context["population"].shape[0]
    train_pass_counts = population[:, context["train_runs"]].sum(axis=1).astype(int)
    records = {}
    for fraction in context["subset_fractions"]:
        size, cell = subset_size(n, fraction), {}
        for variant in sorted(VARIANTS):
            for algorithm in PURE_ALGORITHMS:
                plain, calibrated_key = f"{algorithm}_{variant}", f"{algorithm}_{variant}_cal"
                if not (wanted(context, plain) or wanted(context, calibrated_key)):
                    continue
                vectors, has_embedding, distances, similarities = context["embeddings"](variant)
                selected = algorithms.geometric_on_population(algorithm, vectors, has_embedding, distances, similarities, size)
                if wanted(context, plain):
                    cell[plain] = context["deterministic"](selected)
                if wanted(context, calibrated_key):
                    calibrated = algorithms.calibrate_to_pass_counts(selected, train_pass_counts, size)
                    cell[calibrated_key] = context["deterministic"](calibrated)
        records[subset_key(fraction)] = cell
    return records


def shortlist_methods(_window=None):
    """Shortlist method keys."""
    return [f"{algorithm}_{variant}_{mode}_{str(ratio).replace('.', '')}x" for variant in sorted(VARIANTS)
            for algorithm in SHORTLIST_ALGORITHMS for mode in GROUPING_MODES for ratio in SHORTLIST_RATIOS]


def shortlist_pool(algorithm, mode, strata, space, pool_size, n):
    """Candidate pool of a shortlist configuration: geometric selection of ``pool_size`` instances."""
    vectors, has_embedding, distances, similarities = space
    if mode == "strata":
        quotas = groups.proportional_quotas(strata, pool_size)
        return algorithms.geometric_within_groups(algorithm, strata, quotas, vectors, has_embedding, distances, similarities, n)
    candidates = np.flatnonzero(has_embedding)
    if len(candidates) <= pool_size:
        return candidates.copy()
    return algorithms.geometric(algorithm, vectors, distances, similarities, candidates, pool_size)


def evaluate_shortlist(context):
    """Geometric shortlist of 1.5x or 2x the subset size, then a seeded random draw from the shortlist."""
    population, n = context["population"], context["population"].shape[0]
    strata = groups.outcome_groups(population, context["train_runs"])
    records = {}
    for fraction in context["subset_fractions"]:
        size, cell = subset_size(n, fraction), {}
        for method in shortlist_methods():
            if not wanted(context, method):
                continue
            algorithm, variant, mode, ratio = method.rsplit("_", 3)
            pool_size = min(n, max(size + 1, int(size * (1.5 if ratio == "15x" else 2.0))))
            pool = shortlist_pool(algorithm, mode, strata, context["embeddings"](variant), pool_size, n)
            cell[method] = context["stochastic"](
                lambda rng, _pool=pool, _size=size: rng.choice(_pool, size=min(_size, len(_pool)), replace=False).astype(np.int32))
        records[subset_key(fraction)] = cell
    return records


# Clustering

def clustering_methods(_window=None):
    """Clustering method keys."""
    return [f"{method}_{space}_{mode}" for space in sorted(FEATURE_SPACES) for method in CLUSTER_METHODS for mode in GROUPING_MODES]


def feature_space(context, space):
    """(values, usable mask) of the population in one feature space."""
    n = context["population"].shape[0]
    if space in VARIANTS:
        vectors, has_embedding = context["embeddings"](space)[:2]
        return vectors, has_embedding
    values = clustering.averaged_features(context["trajectory_features"], context["instance_ids"], context["train_run_names"])
    if values is None:
        return np.zeros((n, 1), np.float32), np.zeros(n, dtype=bool)
    if space == "features_top5":
        values = values[:, clustering.COUNT_FEATURE_COLUMNS]
    return np.nan_to_num(values, nan=0.0), ~np.any(np.isnan(values), axis=1)


def evaluate_clustering(context):
    """Stratified random sampling over clusters (whole population, or inside the outcome groups)."""
    population, n = context["population"], context["population"].shape[0]
    strata = groups.outcome_groups(population, context["train_runs"])
    cluster_groups = {}
    for space in FEATURE_SPACES:
        if not any(wanted(context, f"{method}_{space}_{mode}") for method in CLUSTER_METHODS for mode in GROUPING_MODES):
            continue
        values, usable = feature_space(context, space)
        for method in CLUSTER_METHODS:
            if wanted(context, f"{method}_{space}_strata"):
                cluster_groups[(method, space, "strata")] = clustering.clusters_within_groups(strata, values, usable, method)
            if wanted(context, f"{method}_{space}_nostrata"):
                cluster_groups[(method, space, "nostrata")] = clustering.population_clusters(values, usable, method)
    records = {}
    for fraction in context["subset_fractions"]:
        size, cell = subset_size(n, fraction), {}
        for name in clustering_methods():
            if not wanted(context, name):
                continue
            method, rest = name.split("_", 1)
            space, mode = rest.rsplit("_", 1)
            members = cluster_groups[(method, space, mode)]
            quotas = groups.proportional_quotas(members, size)
            cell[name] = context["stochastic"](
                lambda rng, _g=members, _q=quotas, _size=size: algorithms.stratified_random(rng, _g, _q, _size, n))
        records[subset_key(fraction)] = cell
    return records


def chosen_methods(family, window, requested):
    """Method keys of a family to evaluate, in the family's order.

    ``requested`` is None or a collection of names, bare (``centroid_pooled``) or qualified
    (``embedding_strata:centroid_pooled``). A family named by at least one requested method evaluates only the
    requested ones; a family that no requested name refers to is evaluated in full.
    """
    available = FAMILIES[family][0](window)
    every_window = {m for w in (1, 2) for m in FAMILIES[family][0](w)}
    names = {name.split(":", 1)[1] if name.startswith(f"{family}:") else name for name in (requested or ())
             if ":" not in name or name.startswith(f"{family}:")} & every_window
    return [method for method in available if method in names] if names else available


def check_method_names(requested, families, n_runs):
    """Stop when a requested method name matches no method of the given families."""
    known = set()
    for family in families:
        for window in range(1, n_runs):
            known.update(FAMILIES[family][0](window))
            known.update(f"{family}:{m}" for m in FAMILIES[family][0](window))
    unknown = sorted(set(requested) - known)
    if unknown:
        raise SystemExit(f"Unknown method(s) for {', '.join(families)}: {', '.join(unknown)}")


FAMILIES = {
    "baselines": (baseline_methods, evaluate_baselines),
    "embedding_strata": (within_strata_methods, evaluate_within_strata),
    "pure_embedding": (pure_embedding_methods, evaluate_pure_embedding),
    "clustering": (clustering_methods, evaluate_clustering),
    "shortlist": (shortlist_methods, evaluate_shortlist),
}
