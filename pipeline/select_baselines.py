"""Selection stage: the six baselines and the Stability-Stratified control.

    python pipeline/select_baselines.py --dataset multi_model [--windows 1,2] [--subset-sizes 5,10,20,30] [--seeds 500]
                             [--jobs N] [--input VECTOR_DIR] [--output RESULT_DIR] [--distributions 0-99]

The evaluation protocol (synthetic distributions or run groups) is taken from the dataset definition in
config.py. Evaluating only some windows or distributions does not change any random stream.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import protocols  # noqa: E402
from selection import options  # noqa: E402


def main():
    """Command-line entry point."""
    arguments = options.resolve(options.build_parser(__doc__.split("\n")[0]).parse_args())
    protocol = protocols.load(config.dataset(arguments.dataset)["protocol"])
    protocol.run_family(arguments.dataset, "baselines", arguments)


if __name__ == "__main__":
    main()
