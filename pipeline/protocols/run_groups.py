"""Evaluation protocol of single_setup (paper Section 5.2): real run groups instead of synthetic distributions.

A run group ``<N>_runs`` holds the instances that were rerun exactly N times; its population is all of its
instances. Runs are taken in the sorted order of their folder names. For each window size W = 1..N-1, every
temporal split (any W runs as training runs, all runs after the latest training run as test runs) is one evaluation
unit. There is no clustering family and no Stability-Stratified control in this protocol.

Random draws: seed = base + W * 100000 + split * 1000 + draw for the stochastic methods, and
base + W * 100000 + split * 10000 + draw for the per-seed baseline draws of the determinism stage. The errors of the
draws of a stochastic method are computed in one float32 array operation.

Vector files are named ``<OUTCOME>_<repository>_<trajectory id>_{ts,pooled}.npy``; the instance and the repository
of a trajectory are looked up in ``repository_map.json`` (``build_repository_map.py``).

Result files: ``<output>/<family>_<N>runs.json`` (``..._w<W>[-<W>...].json`` when only some window sizes are evaluated)
and, for the determinism stage, ``determinism_<N>runs_w<W>.npz`` with one row per temporal split.
"""
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from selection import algorithms, embeddings, groups, splits
from selection.data import PASSING_PREFIXES
from selection.families import (PURE_ALGORITHMS, STOCHASTIC_BASELINES, TWO_RUN_BASELINES, VARIANTS,
                                WITHIN_STRATA_ALGORITHMS, shortlist_methods, shortlist_pool, subset_key, subset_size, wanted)
from protocols.synthetic_distributions import write_with_runlog

FAMILIES = ("baselines", "embedding_strata", "pure_embedding", "shortlist")
TRAJECTORY_ID = re.compile(r"(chatcmpl-[a-f0-9]+)")


# Data of one run group

def trajectory_id(file_name):
    """Trajectory id inside a vector file name."""
    stem = re.sub(r"^(PASS|FAIL|SUCCESS|TRUE)_", "", re.sub(r"_(ts|pooled)\.npy$", "", file_name))
    match = TRAJECTORY_ID.search(stem)
    return match.group(1) if match else stem


def load_group(group_dir, repository_map):
    """(outcomes[instances, runs] float32, instance ids, run names, repositories) of the instances present in every run."""
    run_names = sorted(d.name for d in Path(group_dir).iterdir() if d.is_dir())
    outcomes, repositories = {}, {}
    for run in run_names:
        for path in sorted((Path(group_dir) / run).glob("*_ts.npy")):
            entry = repository_map.get(trajectory_id(path.name))
            if entry is None:
                continue
            outcomes.setdefault(entry["instance_id"], {})[run] = 1.0 if path.name.startswith(PASSING_PREFIXES) else 0.0
            repositories.setdefault(entry["instance_id"], entry.get("repo", "unknown"))
    complete = sorted(i for i, runs in outcomes.items() if len(runs) == len(run_names))
    matrix = np.zeros((len(complete), len(run_names)), dtype=np.float32)
    for row, instance in enumerate(complete):
        for column, run in enumerate(run_names):
            matrix[row, column] = outcomes[instance][run]
    return matrix, np.array(complete), run_names, np.array([repositories[i] for i in complete])


def embedding_index(group_dir, variant, repository_map):
    """{instance id: {run: vector file}} of one run group and embedding variant."""
    index = {}
    for run_dir in sorted(Path(group_dir).iterdir()):
        if run_dir.is_dir():
            for path in sorted(run_dir.glob(f"*_{variant}.npy")):
                entry = repository_map.get(trajectory_id(path.name))
                if entry is not None:
                    index.setdefault(entry["instance_id"], {})[run_dir.name] = str(path)
    return index


# Groups and evaluation

def repository_crossed_groups(repositories, labels, tag):
    """One group per (repository, label) pair, keyed ``<repository>_<tag><label>`` (the key order decides the draw order)."""
    members = {}
    for position in range(len(repositories)):
        members.setdefault(f"{repositories[position]}_{tag}{int(labels[position])}", []).append(position)
    return {key: np.array(value) for key, value in members.items()}


def baseline_groups(population, repositories, train_runs):
    """Groups of every stratified baseline for one temporal split."""
    counts = population[:, train_runs].sum(axis=1)
    labels = (counts >= int(np.ceil(len(train_runs) / 2))).astype(int)
    result = {"repo_stratified": groups.repository_groups(repositories),
              "difficulty_stratified": {v: np.flatnonzero(labels == v) for v in (0, 1) if np.any(labels == v)},
              "repo_difficulty_stratified": repository_crossed_groups(repositories, labels, "d")}
    if len(train_runs) >= 2:
        result["consistency_stratified"] = {c: np.flatnonzero(counts == c) for c in range(len(train_runs) + 1) if np.any(counts == c)}
        result["repo_consistency_stratified"] = repository_crossed_groups(repositories, counts, "c")
    return result


def split_errors(test_outcomes, selected, population_rates):
    """(RMSE, MaxErr in points, 95th-percentile absolute error in points) of one deterministic subset."""
    errors = np.mean(test_outcomes[selected], axis=0) - population_rates
    return (float(np.sqrt(np.mean(errors ** 2))), float(np.max(np.abs(errors)) * 100.0),
            float(np.percentile(np.abs(errors) * 100.0, 95)) if len(errors) > 0 else 0.0)


def evaluate_draws(select, n_seeds, test_outcomes, population_rates, base_seed, split_index, window):
    """(mean RMSE, largest MaxErr, mean P95) over ``n_seeds`` seeded draws, computed on float32 arrays."""
    selections = [select(np.random.default_rng(base_seed + window * 100000 + split_index * 1000 + draw)) for draw in range(n_seeds)]
    subset_rates = np.array([np.mean(test_outcomes[selected], axis=0) for selected in selections])
    errors = subset_rates - population_rates[None, :]
    absolute = np.abs(errors) * 100.0
    rmse, maxerr = np.sqrt(np.mean(errors ** 2, axis=1)), np.max(absolute, axis=1)
    p95 = np.percentile(absolute, 95, axis=1) if test_outcomes.shape[1] > 0 else np.zeros(n_seeds)
    return float(np.mean(rmse.astype(np.float64))), float(np.max(maxerr.astype(np.float64))), float(np.mean(p95.astype(np.float64)))


def family_methods(family, window):
    """Method keys of a family for one window size, in stored order."""
    if family == "baselines":
        return list(STOCHASTIC_BASELINES_ORDER if window == 1 else STOCHASTIC_BASELINES_ORDER + TWO_RUN_BASELINES)
    if family == "embedding_strata":
        return [f"{a}_{v}" for v in sorted(VARIANTS) for a in WITHIN_STRATA_ALGORITHMS]
    if family == "pure_embedding":
        return [f"{a}_{v}{s}" for v in sorted(VARIANTS) for a in PURE_ALGORITHMS for s in ("", "_cal")]
    return shortlist_methods()


STOCHASTIC_BASELINES_ORDER = ("random", "repo_stratified", "difficulty_stratified", "repo_difficulty_stratified")
assert set(STOCHASTIC_BASELINES_ORDER) == set(STOCHASTIC_BASELINES)


def evaluate_split(task):
    """{subset key: {method: (rmse, maxerr, p95)}} of one family on one temporal split."""
    (family, population, repositories, instance_ids, run_names, indices, window, split_index, train_runs, test_runs,
     subset_fractions, n_seeds, base_seed, requested) = task
    context = {"methods": requested}
    n = population.shape[0]
    test_outcomes = population[:, test_runs]
    rates = np.mean(test_outcomes, axis=0)
    spaces = {}

    def space(variant):
        if variant not in spaces:
            vectors, has_embedding = embeddings.averaged_embeddings(indices[variant], instance_ids.tolist(), [run_names[r] for r in train_runs], variant)
            if vectors is None:
                vectors, has_embedding = np.zeros((n, 1), np.float32), np.zeros(n, dtype=bool)
            spaces[variant] = (vectors, has_embedding, *embeddings.distance_and_similarity(vectors, has_embedding))
        return spaces[variant]

    draws = lambda select: evaluate_draws(select, n_seeds, test_outcomes, rates, base_seed, split_index, window)   # noqa: E731
    strata = groups.outcome_groups(population, train_runs) if family != "baselines" else baseline_groups(population, repositories, train_runs)
    train_pass_counts = population[:, train_runs].sum(axis=1).astype(int)
    result = {}
    for fraction in subset_fractions:
        size, cell = subset_size(n, fraction), {}
        for method in family_methods(family, window):
            if family == "pure_embedding":
                if method.endswith("_cal") or not (wanted(context, method) or wanted(context, method + "_cal")):
                    continue   # a calibrated configuration is evaluated together with its uncalibrated one
            elif not wanted(context, method):
                continue
            if family == "baselines":
                if method == "random":
                    cell[method] = draws(lambda rng, _size=size: algorithms.uniform_random(rng, n, _size))
                else:
                    quotas = groups.proportional_quotas(strata[method], size)
                    cell[method] = draws(lambda rng, _g=strata[method], _q=quotas, _size=size: algorithms.stratified_random(rng, _g, _q, _size, n))
            elif family == "embedding_strata":
                algorithm, variant = method.rsplit("_", 1)
                vectors, has_embedding, distances, similarities = space(variant)
                quotas = groups.proportional_quotas(strata, size)
                cell[method] = split_errors(test_outcomes, algorithms.geometric_within_groups(
                    algorithm, strata, quotas, vectors, has_embedding, distances, similarities, n), rates)
            elif family == "pure_embedding":
                algorithm, variant = method.rsplit("_", 1)
                vectors, has_embedding, distances, similarities = space(variant)
                selected = algorithms.geometric_on_population(algorithm, vectors, has_embedding, distances, similarities, size)
                if wanted(context, method):
                    cell[method] = split_errors(test_outcomes, selected, rates)
                if wanted(context, method + "_cal"):
                    cell[method + "_cal"] = split_errors(test_outcomes, algorithms.calibrate_to_pass_counts(selected, train_pass_counts, size), rates)
            else:
                algorithm, variant, mode, ratio = method.rsplit("_", 3)
                pool_size = min(n, max(size + 1, int(size * (1.5 if ratio == "15x" else 2.0))))
                pool = shortlist_pool(algorithm, mode, strata, space(variant), pool_size, n)
                cell[method] = draws(lambda rng, _pool=pool, _size=size: rng.choice(_pool, size=min(_size, len(_pool)), replace=False).astype(np.int32))
        result[subset_key(fraction)] = cell
    return window, split_index, result


# Entry points of the protocol

def load_inputs(options):
    """Repository map and the run groups to evaluate."""
    import config
    repository_map = json.loads(Path(options.repository_map).read_text())
    sizes = options.run_groups or list(config.dataset(options.dataset)["run_groups"])
    return repository_map, sizes


def window_suffix(windows, n_runs):
    """"" when all window sizes are evaluated, otherwise "_w<W>-<W>..."."""
    return "" if list(windows) == list(range(1, n_runs)) else "_w" + "-".join(str(w) for w in windows)


def run_family(dataset_name, family, options):
    """Evaluate one family on the requested run groups and window sizes; return the written files."""
    if family not in FAMILIES:
        raise SystemExit(f"The run-groups protocol has no '{family}' family. Available: {', '.join(FAMILIES)}")
    repository_map, sizes = load_inputs(options)
    written = []
    for n_runs in sizes:
        started = time.time()
        population, instance_ids, run_names, repositories = load_group(options.vectors / "timeseries" / f"{n_runs}_runs", repository_map)
        windows = [w for w in (options.windows or range(1, n_runs)) if w < n_runs]
        indices = {} if family == "baselines" else {v: embedding_index(options.vectors / {"ts": "timeseries", "pooled": "pooled"}[v] / f"{n_runs}_runs",
                                                                       v, repository_map) for v in VARIANTS}
        requested = None
        if options.methods:
            requested = frozenset(m.split(":", 1)[-1] for m in options.methods if ":" not in m or m.startswith(f"{family}:"))
        tasks = [(family, population, repositories, instance_ids, run_names, indices, w, s, train, test, options.subset_sizes,
                  options.seeds, options.base_seed, requested)
                 for w in windows for s, (train, test) in enumerate(splits.temporal_splits(n_runs, w))]
        print(f"{family}: {n_runs}_runs, {population.shape[0]} instances, windows {windows}, {len(tasks)} temporal splits, {options.jobs} jobs")
        if options.jobs == 1:
            results = [evaluate_split(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=options.jobs) as pool:
                results = list(pool.map(evaluate_split, tasks))
        stored = {}
        for w in windows:
            ordered = [r[2] for r in sorted((r for r in results if r[0] == w), key=lambda r: r[1])]
            stored[str(w)] = {key: {m: {"split_errors": [r[key][m][0] for r in ordered], "split_maxerr": [r[key][m][1] for r in ordered],
                                        "split_p95": [r[key][m][2] for r in ordered],
                                        "mean": float(np.mean([r[key][m][0] for r in ordered])),
                                        "maxerr": float(np.max([r[key][m][1] for r in ordered])),
                                        "p95": float(np.mean([r[key][m][2] for r in ordered]))}
                                    for m in ordered[0][key]} for key in ordered[0]}
        methods = sorted({m for w in stored.values() for cell in w.values() for m in cell})
        payload = {"dataset": dataset_name, "family": family, "n_runs": n_runs, "n_instances": int(population.shape[0]), "run_names": run_names,
                   "subset_sizes": list(options.subset_sizes), "methods": methods, "n_seeds": options.seeds, "base_seed": options.base_seed,
                   "windows": stored}
        path = options.output / f"{family}_{n_runs}runs{window_suffix(windows, n_runs)}.json"
        write_with_runlog(path, payload, started, {"family": family, "run_group": n_runs, "windows": windows, "seeds": options.seeds, "jobs": options.jobs})
        written.append(path)
    return written


def draw_groups(name, population, repositories, train_runs):
    """Groups of one baseline for the per-seed draws (None for Uniform Random)."""
    counts = population[:, train_runs].sum(axis=1)
    labels = (counts >= int(np.ceil(len(train_runs) / 2))).astype(int)
    if name == "random":
        return None
    if name == "difficulty_stratified":
        return {v: np.flatnonzero(labels == v) for v in np.unique(labels)}
    if name == "consistency_stratified":
        return {c: np.flatnonzero(counts == c) for c in range(len(train_runs) + 1) if np.any(counts == c)}
    by_repository = groups.repository_groups(repositories)
    if name == "repo_stratified":
        return by_repository
    second = labels if name == "repo_difficulty_stratified" else counts.astype(int)
    crossed = {}
    for key, members in by_repository.items():
        for member in members:
            crossed.setdefault(f"{key}_x_{second[member]}", []).append(member)
    return {key: np.array(value) for key, value in crossed.items()}


DRAW_BASELINES = ("random", "difficulty_stratified", "repo_stratified", "repo_difficulty_stratified", "consistency_stratified", "repo_consistency_stratified")


def compare_with_draws(results_dir, n_runs, window, suffix, names, arrays, subset_fractions):
    """Comparison of every non-baseline method of a window size with the per-seed draws (win rates, gaps)."""
    comparisons = {subset_key(f): {} for f in subset_fractions}
    for family in FAMILIES[1:]:
        path = Path(results_dir) / f"{family}_{n_runs}runs{suffix}.json"
        if not path.exists():
            continue
        cells = json.loads(path.read_text())["windows"].get(str(window), {})
        for key in comparisons:
            for method, entry in cells.get(key, {}).items():
                errors, mean = entry.get("split_errors", []), entry["mean"]
                record = {}
                for name in names:
                    draws = arrays[f"{key}__{name}"]
                    seed_means = np.mean(draws, axis=0)
                    win_rate = float(np.mean(mean < seed_means))
                    per_split = win_rate
                    if errors and len(errors) == draws.shape[0]:
                        wins = np.zeros(draws.shape[1])
                        for split_index in range(draws.shape[0]):
                            wins += (np.array(errors)[split_index] < draws[split_index, :])
                        per_split = float(np.mean(wins / draws.shape[0]))
                    median = float(np.median(seed_means))
                    record[name] = {"win_rate": round(win_rate * 100, 1), "per_fold_win_rate": round(per_split * 100, 1), "emb_rmse": round(mean, 6),
                                    "bl_median_rmse": round(median, 6), "gap_pct": round((median - mean) / median * 100 if median > 0 else 0, 1),
                                    "catastrophic_rate": round(float(np.mean(seed_means > 2 * mean)) * 100, 1)}
                comparisons[key][method if method.startswith(family) else f"{family}_{method}"] = record
    return comparisons


def run_determinism(dataset_name, options):
    """Per-seed baseline draws of every temporal split, ``determinism_<N>runs_w<W>.npz`` (rows = splits, columns = seeds),
    and their comparison with the non-baseline methods found in ``options.results``, ``determinism_<N>runs.json``."""
    repository_map, sizes = load_inputs(options)
    written = []
    for n_runs in sizes:
        summary = {"n_runs": n_runs, "subset_pcts": list(options.subset_sizes), "n_baseline_seeds": options.seeds, "windows": {}}
        population, _, _, repositories = load_group(options.vectors / "timeseries" / f"{n_runs}_runs", repository_map)
        n = population.shape[0]
        windows = [w for w in (options.windows or range(1, n_runs)) if w < n_runs]
        for window in windows:
            all_splits = splits.temporal_splits(n_runs, window)
            names = [b for b in DRAW_BASELINES if window >= 2 or b not in TWO_RUN_BASELINES]
            arrays = {}
            for fraction in options.subset_sizes:
                for name in names:
                    arrays[f"{subset_key(fraction)}__{name}"] = np.zeros((len(all_splits), options.seeds))
                    arrays[f"{subset_key(fraction)}__{name}__maxerr"] = np.zeros((len(all_splits), options.seeds))
            for split_index, (train_runs, test_runs) in enumerate(all_splits):
                test_outcomes = population[:, test_runs]
                rates = np.mean(test_outcomes, axis=0)
                strata = {name: draw_groups(name, population, repositories, train_runs) for name in names}
                for fraction in options.subset_sizes:
                    size, key = subset_size(n, fraction), subset_key(fraction)
                    for name in names:
                        quotas = groups.proportional_quotas(strata[name], size) if strata[name] else None
                        for draw in range(options.seeds):
                            rng = np.random.default_rng(options.base_seed + window * 100000 + split_index * 10000 + draw)
                            selected = (rng.choice(n, size=size, replace=False) if strata[name] is None
                                        else algorithms.stratified_random(rng, strata[name], quotas, size, n))
                            errors = np.mean(test_outcomes[selected], axis=0) - rates
                            arrays[f"{key}__{name}"][split_index, draw] = float(np.sqrt(np.mean(errors ** 2)))
                            arrays[f"{key}__{name}__maxerr"][split_index, draw] = float(np.max(np.abs(errors)) * 100.0)
            path = options.output / f"determinism_{n_runs}runs_w{window}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **{k: v.astype(np.float32) for k, v in arrays.items()})
            print(f"  wrote {path}")
            written.append(path)
            summary["windows"][str(window)] = {"baselines": names, "n_splits": len(all_splits), "comparisons": compare_with_draws(
                options.results, n_runs, window, window_suffix(windows, n_runs), names, arrays, options.subset_sizes)}
        path = options.output / f"determinism_{n_runs}runs{window_suffix(windows, n_runs)}.json"
        path.write_text(json.dumps(summary, indent=2))
        written.append(path)
    return written
