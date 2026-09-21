"""Determinism validation stage (Finding 3.4): independent per-seed baseline draws and their comparison
with every non-baseline method.

    python pipeline/determinism.py --dataset multi_model [--windows 2] [--results RESULT_DIR] [--output RESULT_DIR]
                                   [--seeds 500] [--jobs N] [--input VECTOR_DIR] [--distributions 0-99]

Reads the family result files of ``--results`` (default: the output folder) and writes
``determinism_w<W>.npz`` (per-distribution x per-seed RMSE and MaxErr of every baseline) and
``determinism_w<W>.json`` (comparisons).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import protocols  # noqa: E402
from selection import options  # noqa: E402


def main():
    """Command-line entry point."""
    parser = options.build_parser(__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=None, help="folder with the family result files (default: --output)")
    parser.add_argument("--draws-only", action="store_true", help="synthetic-distributions protocol only: write the per-seed draws (npz) without the comparisons")
    parser.add_argument("--keep-float64", action="store_true", help="synthetic-distributions protocol only: also store the RMSE draws in float64 (<name>.float64.npz)")
    arguments = options.resolve(parser.parse_args())
    arguments.results = arguments.results or arguments.output
    protocols.load(config.dataset(arguments.dataset)["protocol"]).run_determinism(arguments.dataset, arguments)


if __name__ == "__main__":
    main()
