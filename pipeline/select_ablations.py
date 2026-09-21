"""Selection stage: the ablation families of RQ3 (Pure Embedding, Clustering, Shortlist).

    python pipeline/select_ablations.py --dataset multi_model [--families pure_embedding,clustering,shortlist]
                             [--windows 1,2] [--subset-sizes 5,10,20,30] [--seeds 500] [--jobs N]
                             [--input VECTOR_DIR] [--output RESULT_DIR] [--distributions 0-99]
                             [--features FILE.npz --features-meta FILE.json]

Clustering needs the trajectory features (``--features``, ``--features-meta``; default under
data/<dataset>/features/).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import protocols  # noqa: E402
from selection import options  # noqa: E402

ABLATION_FAMILIES = ("pure_embedding", "clustering", "shortlist")


def main():
    """Command-line entry point."""
    parser = options.build_parser(__doc__.split("\n")[0])
    parser.add_argument("--families", default=",".join(ABLATION_FAMILIES), help="comma-separated (default: all; clustering only with synthetic distributions)")
    parser.add_argument("--features", type=Path, default=None, help="clustering: trajectory_features.npz (default: data/<dataset>/features/)")
    parser.add_argument("--features-meta", type=Path, default=None, help="clustering: trajectory_features_meta.json (default: next to --features)")
    arguments = options.resolve(parser.parse_args())
    feature_dir = config.stage_dir("features", arguments.dataset)
    arguments.features = arguments.features or feature_dir / "trajectory_features.npz"
    arguments.features_meta = arguments.features_meta or feature_dir / "trajectory_features_meta.json"
    protocol = protocols.load(config.dataset(arguments.dataset)["protocol"])
    families = arguments.families.split(",")
    if config.dataset(arguments.dataset)["protocol"] == "run_groups" and arguments.families == ",".join(ABLATION_FAMILIES):
        families = [f for f in families if f != "clustering"]   # this protocol has no clustering family
    for family in families:
        if family not in ABLATION_FAMILIES:
            raise SystemExit(f"Unknown family '{family}'. Available: {', '.join(ABLATION_FAMILIES)}")
        protocol.run_family(arguments.dataset, family, arguments)


if __name__ == "__main__":
    main()
