"""Evaluation protocol of multi_model and multi_agent (paper Section 5.2).

Every synthetic distribution is a population of 250 instances of one pool. For each window size W, every
temporal split (W training runs, all later runs as test runs) is evaluated for every subset size; stochastic
methods are repeated with independent seeds. The seed of a draw depends on the distribution's position in the
distributions file, the split and the draw number, so evaluating only some distributions or windows leaves
every random stream unchanged.

Result files: ``<output>/<family>_w<W>.json`` (``..._d<first>-<last>.json`` when only some distributions are
evaluated), each with a ``.runlog.json`` holding wall-clock time and peak memory.
"""
import json
import resource
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from selection import clustering, data, embeddings, metrics, splits
from selection.families import FAMILIES, check_method_names, chosen_methods, subset_key

_SHARED = {}


def _init_worker(shared):
    """Keep the read-only inputs in the worker process."""
    _SHARED.update(shared)


def evaluate_distribution(task):
    """All splits, subset sizes and methods of one family on one distribution."""
    family, index, level, members, window, methods = task
    _, evaluate = FAMILIES[family]
    population = _SHARED["outcomes"][members]
    instance_ids = [_SHARED["instance_ids"][i] for i in members]
    per_method = {}
    for split_index, (train_runs, test_runs) in enumerate(splits.temporal_splits(population.shape[1], window)):
        test_outcomes = population[:, test_runs]
        rates = np.mean(test_outcomes, axis=0)
        train_run_names = [_SHARED["run_names"][r] for r in train_runs]
        cache = {}

        def space(variant, _names=train_run_names, _cache=cache):
            if variant not in _cache:
                _cache[variant] = embeddings.population_embeddings(_SHARED["embedding_index"][variant], instance_ids, _names, variant)
            return _cache[variant]

        context = {
            "population": population, "repositories": _SHARED["repositories"][members], "instance_ids": instance_ids,
            "train_runs": train_runs, "train_run_names": train_run_names, "subset_fractions": _SHARED["subset_fractions"],
            "methods": methods,
            "embeddings": space, "trajectory_features": _SHARED.get("trajectory_features"),
            "deterministic": lambda selected, _t=test_outcomes, _r=rates: metrics.evaluate_deterministic(_t, selected, _r),
            "stochastic": lambda select, _t=test_outcomes, _r=rates, _s=split_index: metrics.evaluate_stochastic(
                select, _SHARED["n_seeds"], _t, _r, _SHARED["base_seed"], _s, index),
        }
        for key, cell in evaluate(context).items():
            for method, record in cell.items():
                per_method.setdefault(key, {}).setdefault(method, []).append(record)
    return index, level, len(members), per_method


def assemble(results, methods, subset_keys):
    """Per-method lists over distributions: split-level values and distribution-level aggregates."""
    table = {key: {m: {"split_errors": [], "split_maxerr": [], "split_maxerr_mean": [], "split_run_errors": [],
                       "selected_subsets": [], "dist_means": [], "dist_maxerr": [], "dist_maxerr_mean": [], "dist_p95": []}
                   for m in methods} for key in subset_keys}
    for _, _, _, per_method in results:
        for key in subset_keys:
            for method in methods:
                records, entry = per_method[key][method], table[key][method]
                entry["split_errors"].append([r["rmse"] for r in records])
                entry["split_maxerr"].append([r["maxerr"] for r in records])
                entry["split_maxerr_mean"].append([r["maxerr_mean"] for r in records])
                entry["split_run_errors"].append([r["run_errors"] for r in records])
                entry["selected_subsets"].append([r["subset"] for r in records])
                entry["dist_means"].append(float(np.mean([r["rmse"] for r in records])))
                entry["dist_maxerr"].append(float(np.max([r["maxerr"] for r in records])))
                entry["dist_maxerr_mean"].append(float(np.max([r["maxerr_mean"] for r in records])))
                entry["dist_p95"].append(float(np.mean([r["p95"] for r in records])))
    return table


def load_shared(options, family):
    """Read-only inputs of the workers."""
    wanted = data.distribution_instance_ids(options.distributions_file)
    outcomes, instance_ids, run_names, repositories = data.load_outcomes(options.vectors / "timeseries", wanted)
    shared = {"outcomes": outcomes, "instance_ids": instance_ids, "run_names": run_names, "repositories": repositories,
              "subset_fractions": options.subset_sizes, "n_seeds": options.seeds, "base_seed": options.base_seed,
              "embedding_index": {}}
    if family != "baselines":
        shared["embedding_index"] = {"pooled": data.embedding_index(options.vectors / "pooled", "pooled"),
                                     "ts": data.embedding_index(options.vectors / "timeseries", "ts")}
    requested = options.methods or ()
    needs_features = not requested or any("features" in m for m in requested if ":" not in m or m.startswith("clustering:"))
    if family == "clustering" and needs_features:
        shared["trajectory_features"] = clustering.load_trajectory_features(options.features, options.features_meta)
    return shared


def select_distributions(distributions, wanted):
    """Distributions to evaluate; ``wanted`` is None (all) or a list of positions."""
    return distributions if wanted is None else [d for d in distributions if d[0] in set(wanted)]


def peak_memory_mb():
    """Peak resident memory of this process and of its largest worker, in MB."""
    scale = 1024 ** 2 if sys.platform == "darwin" else 1024
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / scale)


def map_distributions(function, tasks, shared, jobs):
    """Run ``function`` over tasks in ``jobs`` worker processes; results ordered by distribution position."""
    if jobs == 1:
        _init_worker(shared)
        results = [function(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(shared,)) as pool:
            results = list(pool.map(function, tasks))
    return sorted(results, key=lambda item: item[0])


def file_suffix(positions, total):
    """"" for all distributions, otherwise "_d<first>-<last>" of the evaluated positions."""
    return "" if len(positions) == total else f"_d{positions[0]}-{positions[-1]}"


def output_path(options, stem, window, chosen, total):
    """Result file of a window; partial runs carry the range of distribution positions."""
    return options.output / f"{stem}_w{window}{file_suffix([c[0] for c in chosen], total)}.json"


def write_with_runlog(path, payload, started, extra):
    """Write a result file and its run log (wall-clock time, peak memory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    write_runlog(path, started, extra)


def write_runlog(path, started, extra):
    """Run log of a written result file: wall-clock time and peak memory."""
    own, workers = peak_memory_mb()
    log = {"result_file": path.name, "wall_clock_seconds": round(time.time() - started, 1),
           "peak_memory_mb_main_process": round(own, 1), "peak_memory_mb_largest_worker": round(workers, 1), **extra}
    path.with_suffix(".runlog.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB) in {log['wall_clock_seconds']} s; "
          f"peak memory {own:.0f} MB main, {workers:.0f} MB per worker")


def run_family(dataset_name, family, options):
    """Evaluate one method family for the requested windows; return the written files."""
    shared = load_shared(options, family)
    distributions = data.load_distributions(options.distributions_file, shared["instance_ids"])
    chosen = select_distributions(distributions, options.distributions)
    n_runs = shared["outcomes"].shape[1]
    windows = options.windows or list(range(1, n_runs))
    keys = [subset_key(f) for f in options.subset_sizes]
    requested = options.methods
    if requested is not None:
        check_method_names(requested, list(FAMILIES), n_runs)
    print(f"{family}: {shared['outcomes'].shape[0]} instances x {n_runs} runs from {options.vectors}; "
          f"{len(chosen)} of {len(distributions)} distributions; windows {windows}; {options.seeds} seeds; {options.jobs} jobs")
    written = []
    for window in windows:
        started = time.time()
        methods = chosen_methods(family, window, requested)
        if not methods:
            print(f"  window {window}: the requested methods of {family} do not exist for this window size; nothing to do")
            continue
        whole_family = methods == FAMILIES[family][0](window)
        tasks = [(family, index, level, members, window, None if whole_family else frozenset(methods)) for index, level, members in chosen]
        results = map_distributions(evaluate_distribution, tasks, shared, options.jobs)
        payload = {"dataset": dataset_name, "family": family, "window": window, "subset_sizes": list(options.subset_sizes),
                   "methods": methods, "n_seeds": options.seeds, "base_seed": options.base_seed,
                   "run_names": shared["run_names"], "instance_ids": shared["instance_ids"].tolist(),
                   "splits": [{"train": tr, "test": te} for tr, te in splits.temporal_splits(n_runs, window)],
                   "distribution_positions": [r[0] for r in results], "distribution_levels": [r[1] for r in results],
                   "distribution_sizes": [r[2] for r in results], "results": assemble(results, methods, keys)}
        path = output_path(options, family, window, chosen, len(distributions))
        write_with_runlog(path, payload, started, {"family": family, "window": window, "distributions": len(chosen),
                                                   "seeds": options.seeds, "jobs": options.jobs})
        written.append(path)
    return written


def draw_distribution(task):
    """Per-seed baseline draws of one distribution (worker function of the determinism stage)."""
    from selection import determinism
    index, _, members, window = task
    rmse, maxerr = determinism.per_seed_draws(_SHARED["outcomes"][members], _SHARED["repositories"][members], window,
                                             _SHARED["subset_fractions"], _SHARED["n_seeds"], _SHARED["base_seed"], index)
    return index, rmse, maxerr


def method_rmse_from_results(results_dir, window, chosen_positions, suffix):
    """{"<family>:<method>": {subset key: per-distribution RMSE}} of all non-baseline result files of a window."""
    collected = {}
    for family in FAMILIES:
        path = results_dir / f"{family}_w{window}{suffix}.json"
        if family == "baselines" or not path.exists():
            continue
        stored = json.loads(path.read_text())
        rows = [stored["distribution_positions"].index(p) for p in chosen_positions]
        for key, methods in stored["results"].items():
            for method, entry in methods.items():
                collected.setdefault(f"{family}:{method}", {})[key] = np.asarray(entry["dist_means"])[rows]
    return collected


def comparisons_payload(results_dir, window, subset_sizes, n_seeds, positions, levels, draws, suffix):
    """Comparison of every non-baseline method found in ``results_dir`` with the per-seed draws of a window.

    ``draws`` maps "<subset key>__<baseline>" to an array [distributions, seeds] of per-seed RMSE.
    """
    from selection import determinism
    keys, names = [subset_key(f) for f in subset_sizes], determinism.baselines_for(window)
    comparisons = {key: {} for key in keys}
    for method, per_key in method_rmse_from_results(results_dir, window, positions, suffix).items():
        for key in keys:
            if key in per_key:
                comparisons[key][method] = {name: determinism.compare_with_draws(per_key[key], draws[f"{key}__{name}"], levels)
                                            for name in names}
    return {"window": window, "subset_sizes": list(subset_sizes), "n_baseline_seeds": n_seeds,
            "n_distributions": len(positions), "baselines": names,
            "best_baseline": "consistency_stratified" if window >= 2 else "difficulty_stratified",
            "distribution_positions": list(positions), "distribution_levels": list(levels),
            "methods_compared": sorted(comparisons[keys[0]]), "comparisons": comparisons}


def run_determinism(dataset_name, options):
    """Per-seed baseline draws (npz) and, unless ``options.draws_only``, their comparison with every non-baseline
    method found in ``options.results`` (json)."""
    from selection import determinism
    shared = load_shared(options, "baselines")
    distributions = data.load_distributions(options.distributions_file, shared["instance_ids"])
    chosen = select_distributions(distributions, options.distributions)
    windows = options.windows or list(range(1, shared["outcomes"].shape[1]))
    keys = [subset_key(f) for f in options.subset_sizes]
    written = []
    for window in windows:
        started = time.time()
        tasks = [(index, level, members, window) for index, level, members in chosen]
        draws = map_distributions(draw_distribution, tasks, shared, options.jobs)
        arrays = {}
        for key in keys:
            for name in determinism.baselines_for(window):
                arrays[f"{key}__{name}"] = np.array([d[1][key][name] for d in draws]).astype(np.float32)
                arrays[f"{key}__{name}__maxerr"] = np.array([d[2][key][name] for d in draws]).astype(np.float32)
        path = output_path(options, "determinism", window, chosen, len(distributions))
        path.parent.mkdir(parents=True, exist_ok=True)
        positions, levels = [c[0] for c in chosen], [c[1] for c in chosen]
        np.savez_compressed(path.with_suffix(".npz"), distribution_positions=np.array(positions), distribution_levels=np.array(levels), **arrays)
        written.append(path.with_suffix(".npz"))
        full_precision = {f"{key}__{name}": np.array([d[1][key][name] for d in draws]) for key in keys for name in determinism.baselines_for(window)}
        if options.keep_float64:   # the comparisons use the RMSE draws before their reduction to float32
            np.savez_compressed(path.with_suffix(".float64.npz"), **full_precision)
        log = {"stage": "determinism", "window": window, "distributions": len(chosen), "seeds": options.seeds, "jobs": options.jobs}
        if options.draws_only:
            write_runlog(path.with_suffix(".npz"), started, log)
            continue
        payload = comparisons_payload(options.results, window, options.subset_sizes, options.seeds, positions, levels, full_precision,
                                      file_suffix(positions, len(distributions)))
        write_with_runlog(path, payload, started, log)
        written.append(path)
    return written


# Blocks of distributions

def assemble_blocks(block_files, target):
    """Join family result files of consecutive blocks of distributions into one result file."""
    parts = sorted((json.loads(Path(f).read_text()) for f in block_files), key=lambda part: part["distribution_positions"][0])
    joined = parts[0]
    for part in parts[1:]:
        for field in ("dataset", "family", "window", "subset_sizes", "methods", "n_seeds", "base_seed", "run_names", "instance_ids", "splits"):
            if part[field] != joined[field]:
                raise SystemExit(f"Blocks of {target.name} differ in '{field}'")
        for field in ("distribution_positions", "distribution_levels", "distribution_sizes"):
            joined[field] += part[field]
        for key, methods in part["results"].items():
            for method, entry in methods.items():
                for field, values in entry.items():
                    joined["results"][key][method][field] += values
    if joined["distribution_positions"] != sorted(set(joined["distribution_positions"])):
        raise SystemExit(f"Blocks of {target.name} overlap or are out of order")
    target.write_text(json.dumps(joined))
    return joined["distribution_positions"]


def assemble_draw_blocks(block_files, target):
    """Join per-seed draw files (npz) of consecutive blocks of distributions."""
    parts = sorted((dict(np.load(f)) for f in block_files), key=lambda part: int(part["distribution_positions"][0]) if "distribution_positions" in part else 0)
    np.savez_compressed(target, **{name: np.concatenate([part[name] for part in parts]) for name in parts[0]})
