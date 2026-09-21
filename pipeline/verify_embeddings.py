"""Check the embedding script against the published vectors on a sample of trajectories.

    python pipeline/verify_embeddings.py --dataset multi_model [--sanitized SANITIZED_DIR] [--vectors VECTOR_DIR]
                                         [--run-groups 10] [--per-run 5] [--seed SEED] [--output REPORT_DIR]
                                         [--device mps|cpu]

For each run of the dataset the sample holds ``--per-run`` sanitized trajectories: the first ones by sorted
file name, or a seeded random sample when ``--seed`` is given. Each is embedded with ``embed.py`` and compared
with the published vectors in ``VECTOR_DIR/pooled/<run>/`` and ``VECTOR_DIR/timeseries/<run>/``:

    pooled vector        maximum absolute difference, cosine similarity
    time-series matrix   shape equality, maximum absolute difference, lowest and mean cosine similarity per step

The report (``embedding_verification.json`` and ``.md``) lists these numbers per trajectory together with the
device and the time per trajectory. It applies no pass/fail threshold.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import embed  # noqa: E402


def sample_files(dataset_name, sanitized_dir, per_run, seed, run_groups=None):
    """Sanitized files of the sample: per run, the first ``per_run`` by name or a seeded random sample."""
    files = []
    for run in config.runs_of(dataset_name, run_groups):
        candidates = sorted((sanitized_dir / run).glob("unified_*.json"))
        if not candidates:
            raise SystemExit(f"No sanitized files in {sanitized_dir / run}")
        files += candidates[:per_run] if seed is None else sorted(random.Random(seed).sample(candidates, min(per_run, len(candidates))))
    return files


def cosine(first, second):
    """Cosine similarity of two vectors."""
    return float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))


def compare_trajectory(path, sanitized_dir, vector_dir, model, device):
    """Embed one trajectory and compare it with its published vectors."""
    record = json.loads(path.read_text(encoding="utf-8"))
    started = time.time()
    series, pooled = embed.embed_trajectory(record, model, device)
    seconds = time.time() - started
    run, name = path.relative_to(sanitized_dir).parent, embed.output_name(record, path.stem)
    stored_series = np.load(vector_dir / "timeseries" / run / f"{name}_ts.npy")
    stored_pooled = np.load(vector_dir / "pooled" / run / f"{name}_pooled.npy")
    result = {"run": str(run), "trajectory": name, "steps": int(len(series)), "seconds": round(seconds, 2),
              "pooled_max_abs_difference": float(np.max(np.abs(pooled - stored_pooled))),
              "pooled_cosine_similarity": cosine(pooled, stored_pooled),
              "timeseries_shape": list(series.shape), "stored_timeseries_shape": list(stored_series.shape),
              "timeseries_shape_equal": series.shape == stored_series.shape,
              "pooled_dtype": str(stored_pooled.dtype), "timeseries_dtype": str(stored_series.dtype)}
    if result["timeseries_shape_equal"]:
        per_step = [cosine(a, b) for a, b in zip(series, stored_series)]
        result.update({"timeseries_max_abs_difference": float(np.max(np.abs(series - stored_series))),
                       "timeseries_lowest_step_cosine": min(per_step), "timeseries_mean_step_cosine": float(np.mean(per_step))})
    return result


def write_report(output_dir, header, results):
    """Write the JSON and markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "embedding_verification.json").write_text(json.dumps({**header, "trajectories": results}, indent=2) + "\n", encoding="utf-8")
    lines = ["# Embedding verification", ""] + [f"- {key}: {value}" for key, value in header.items()] + [
        "", "| Run | Trajectory | Steps | Seconds | Pooled max abs diff | Pooled cosine | TS shape equal | TS max abs diff | TS lowest step cosine |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|"]
    for r in results:
        lines.append(f"| {r['run']} | {r['trajectory']} | {r['steps']} | {r['seconds']} | {r['pooled_max_abs_difference']:.2e} | "
                     f"{r['pooled_cosine_similarity']:.8f} | {r['timeseries_shape_equal']} | "
                     f"{r.get('timeseries_max_abs_difference', float('nan')):.2e} | {r.get('timeseries_lowest_step_cosine', float('nan')):.8f} |")
    (output_dir / "embedding_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(config.DATASETS))
    parser.add_argument("--sanitized", type=Path, default=None, help="sanitized folder (default: data/<dataset>/sanitized)")
    parser.add_argument("--vectors", type=Path, default=None, help="published vectors (default: data/<dataset>/vectors)")
    parser.add_argument("--output", type=Path, default=None, help="report folder (default: data/<dataset>/embedding_verification)")
    parser.add_argument("--run-groups", default=None, help="run-group sizes to sample, e.g. 10 (datasets with run groups only)")
    parser.add_argument("--per-run", type=int, default=5, help="trajectories per run")
    parser.add_argument("--seed", type=int, default=None, help="draw a seeded random sample instead of the first files by name")
    parser.add_argument("--device", default=None, choices=("mps", "cpu"), help="default: mps when available, else cpu")
    arguments = parser.parse_args()
    sanitized_dir = arguments.sanitized or config.stage_dir("sanitized", arguments.dataset)
    vector_dir = arguments.vectors or config.stage_dir("vectors", arguments.dataset)
    output_dir = arguments.output or config.stage_dir("embedding_verification", arguments.dataset)
    device = arguments.device or embed.default_device()
    run_groups = [int(g) for g in arguments.run_groups.split(",")] if arguments.run_groups else None
    files = sample_files(arguments.dataset, sanitized_dir, arguments.per_run, arguments.seed, run_groups)
    model = embed.load_model(device)
    results = []
    for index, path in enumerate(files, start=1):
        results.append(compare_trajectory(path, sanitized_dir, vector_dir, model, device))
        print(f"  [{index}/{len(files)}] {results[-1]['trajectory']}: pooled max abs diff {results[-1]['pooled_max_abs_difference']:.2e}, "
              f"cosine {results[-1]['pooled_cosine_similarity']:.8f}, {results[-1]['seconds']} s")
    header = {"dataset": arguments.dataset, "model": f"{embed.MODEL_NAME}@{embed.MODEL_REVISION}", "device": device,
              "sample": f"{arguments.per_run} per run, " + ("first by file name" if arguments.seed is None else f"random with seed {arguments.seed}"),
              "trajectories": len(results), "steps": sum(r["steps"] for r in results),
              "seconds_per_trajectory": round(sum(r["seconds"] for r in results) / len(results), 2),
              "seconds_per_step": round(sum(r["seconds"] for r in results) / sum(r["steps"] for r in results), 3),
              "largest_pooled_max_abs_difference": max(r["pooled_max_abs_difference"] for r in results),
              "lowest_pooled_cosine_similarity": min(r["pooled_cosine_similarity"] for r in results),
              "all_timeseries_shapes_equal": all(r["timeseries_shape_equal"] for r in results)}
    write_report(output_dir, header, results)
    print(f"verify_embeddings: read {len(files)} sanitized files from {sanitized_dir} and vectors from {vector_dir}; wrote {output_dir}")
    print(json.dumps(header, indent=2))


if __name__ == "__main__":
    main()
