"""Build pooled vectors from time-series matrices (``*_ts.npy`` -> ``*_pooled.npy``).

    python pipeline/pool_vectors.py --input TIMESERIES_DIR --output POOLED_DIR

The pooled vector of a trajectory is a function of its time-series matrix alone (``embed.pooled_vector``: first,
middle and last step, mean and standard deviation), so pooled vectors can be rebuilt without running the embedding
model. The folder structure below ``--input`` is kept.
"""
import argparse
from pathlib import Path

import numpy as np



def pooled_vector(series):
    """Same expressions as ``embed.pooled_vector`` (kept here so that this utility does not import the embedding model)."""
    spread = np.std(series, axis=0) if len(series) > 1 else np.zeros_like(series[0])
    return np.concatenate([series[0], series[len(series) // 2], series[-1], np.mean(series, axis=0), spread])


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, required=True, help="folder with *_ts.npy files (searched recursively)")
    parser.add_argument("--output", type=Path, required=True, help="folder for the *_pooled.npy files (same sub-folders as the input)")
    arguments = parser.parse_args()
    files = sorted(arguments.input.rglob("*_ts.npy"))
    for path in files:
        target = arguments.output / path.relative_to(arguments.input).with_name(path.name[:-len("_ts.npy")] + "_pooled.npy")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, pooled_vector(np.load(path)))
    print(f"pool_vectors: read {len(files)} matrices from {arguments.input}, wrote {len(files)} pooled vectors to {arguments.output}")


if __name__ == "__main__":
    main()
