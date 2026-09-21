"""Command-line options shared by the selection entry points."""
import argparse
import os
from pathlib import Path

import config

# One numerical thread per worker process: the workers already run in parallel (set before numpy is imported).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def parse_positions(text):
    """"0-19" or "3,7,12" -> list of distribution positions."""
    if "-" in text:
        first, last = text.split("-")
        return list(range(int(first), int(last) + 1))
    return [int(part) for part in text.split(",")]


def build_parser(description):
    """Argument parser with the options of every selection stage."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", required=True, choices=sorted(config.DATASETS))
    parser.add_argument("--windows", default=None, help="window sizes, e.g. 1,2,3 (default: all, 1 to runs-1)")
    parser.add_argument("--subset-sizes", default=None, help="subset fractions in percent, e.g. 5,10,20,30 (default: all four)")
    parser.add_argument("--seeds", type=int, default=config.N_SEEDS, help="random draws per stochastic method and split")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker processes")
    parser.add_argument("--input", type=Path, default=None, help="vector folder with pooled/ and timeseries/ (default: data/<dataset>/vectors)")
    parser.add_argument("--output", type=Path, default=None, help="result folder (default: data/<dataset>/selection)")
    parser.add_argument("--methods", default=None, help="evaluate only these methods, e.g. centroid_pooled,fl_pooled or "
                        "shortlist:centroid_pooled_strata_15x (default: every method of the family)")
    parser.add_argument("--distributions", default=None,
                        help="synthetic-distributions protocol only: distributions to evaluate, e.g. 0-99 or 3,7 (default: all)")
    parser.add_argument("--distributions-file", type=Path, default=None,
                        help="synthetic-distributions protocol only (default: data/<dataset>/distributions/synthetic_distributions.json)")
    parser.add_argument("--run-groups", default=None, help="run-groups protocol only: run-group sizes to evaluate, e.g. 10 or 5,6")
    parser.add_argument("--repository-map", type=Path, default=None,
                        help="run-groups protocol only: repository_map.json (default: data/<dataset>/repository_map.json)")
    return parser


def resolve(arguments):
    """Fill defaults from the dataset definition, convert list options, and add ``vectors`` (the input folder) and ``base_seed``."""
    name = arguments.dataset
    arguments.windows = [int(w) for w in arguments.windows.split(",")] if arguments.windows else None
    arguments.subset_sizes = tuple(int(s) / 100 for s in arguments.subset_sizes.split(",")) if arguments.subset_sizes else config.SUBSET_SIZES
    arguments.vectors = arguments.input or config.stage_dir("vectors", name)
    arguments.output = arguments.output or config.stage_dir("selection", name)
    arguments.base_seed = config.SEED
    arguments.methods = arguments.methods.split(",") if arguments.methods else None
    if arguments.run_groups:
        arguments.run_groups = [int(g) for g in arguments.run_groups.split(",")]
    if arguments.repository_map is None:
        arguments.repository_map = config.repository_map(name)
    if arguments.distributions:
        arguments.distributions = parse_positions(arguments.distributions)
    if arguments.distributions_file is None:
        arguments.distributions_file = config.stage_dir("distributions", name) / "synthetic_distributions.json"
    return arguments
