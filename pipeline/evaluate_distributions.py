"""Evaluation stage: per-distribution errors of selected methods for every window size
(``per_distribution_mean_rmse.json``; Finding 3.4, second test, and the per-level sign test of Section 8.4).

    python pipeline/evaluate_distributions.py --dataset multi_model [--input SELECTION_DIR] [--output RESULT_DIR]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evaluation import inputs  # noqa: E402

METHODS = {"embedding_strata:centroid_pooled": "centroid_pooled", "embedding_strata:core_edge_pooled": "core_edge_pooled",
           "baselines:consistency_stratified": "consist_strat", "baselines:difficulty_stratified": "diff_strat",
           "baselines:stability_stratified": "stability_stratified", "baselines:random": "random"}
FIELDS = ("dist_means", "dist_maxerr", "dist_p95", "dist_maxerr_mean")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True,   # the file exists only for datasets with synthetic distributions
                        choices=sorted(n for n, d in config.DATASETS.items() if d["protocol"] == "synthetic_distributions"))
    parser.add_argument("--input", type=Path, default=None, help="selection results (default: data/<dataset>/selection)")
    parser.add_argument("--output", type=Path, default=None, help="default: data/<dataset>/results")
    arguments = parser.parse_args()
    source = arguments.input or config.stage_dir("selection", arguments.dataset)
    target = arguments.output or config.stage_dir("results", arguments.dataset)
    extract = {"windows": {}, "methods": {}}
    for window in inputs.available_windows(source):
        table = inputs.load_window(source, window, with_splits=False)
        present = {name: tag for name, tag in METHODS.items() if name in table["methods"]}
        present = {name: present[name] for name in table["methods"] if name in present}
        extract["windows"][str(window)] = {"n_distributions": table["n_distributions"], "buckets": table["levels"], "sizes": table["sizes"],
                                           "splits": table["splits"], "run_names": table["run_names"], "source_methods": present}
        for name, tag in present.items():
            for key, methods in table["results"].items():
                extract["methods"].setdefault(tag, {}).setdefault(key, {})[str(window)] = {f: methods[name][f].tolist() for f in FIELDS}
    target.mkdir(parents=True, exist_ok=True)
    (target / "per_distribution_mean_rmse.json").write_text(json.dumps(extract))
    print(f"{arguments.dataset}: wrote per_distribution_mean_rmse.json ({', '.join(extract['methods'])}; windows {', '.join(extract['windows'])})")


if __name__ == "__main__":
    main()
