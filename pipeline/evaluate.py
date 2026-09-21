"""Evaluation stage: aggregated results of a dataset (``aggregated_results.json``, read by the research-question scripts).

    python pipeline/evaluate.py --dataset multi_model [--input SELECTION_DIR] [--output RESULT_DIR]
                                [--windows 1,2] [--subset-sizes 5,10,20,30] [--n-top 30] [--gzip]

Reads the selection result files and the determinism files of ``--input`` and writes
``aggregated_results.json`` and ``evaluation_manifest.json`` to ``--output``. The manifest lists, per method
family, which configurations were evaluated and whether the family is complete; statistics taken over a whole
family (its best configuration) are meaningful only for complete families. The stage draws no random numbers.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evaluation import aggregated, inputs  # noqa: E402
from selection.families import FAMILIES  # noqa: E402

META = {"version": "3.0", "tie_tolerance": aggregated.TIE_TOLERANCE, "effect_size": "Cliff delta paired (Romano et al. 2006)",
        "thresholds": {"N": "<0.147", "S": "<0.33", "M": "<0.474", "L": ">=0.474"},
        "ps": "Vargha & Delaney (2000): neg<0.56, S<0.64, M<0.71, L>=0.71", "significance": "Wilcoxon + Holm-Bonferroni a=0.05",
        "determinism": "500-seed Monte Carlo reference distribution test", "ci": "Clopper-Pearson exact"}


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy scalars and arrays."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def family_coverage(tables):
    """{family: evaluated configurations, all configurations, complete?} over the loaded windows."""
    coverage = {}
    for family in FAMILIES:
        evaluated = sorted({m for t in tables.values() for m in t["families"].get(family, [])})
        every = sorted({m for w in tables for m in FAMILIES[family][0](w)})
        complete = bool(evaluated) and all(t["families"].get(family) == FAMILIES[family][0](w) for w, t in tables.items())
        coverage[family] = {"evaluated": evaluated, "n_evaluated": len(evaluated), "n_configurations": len(every), "complete": complete,
                            "best_of_family": "computable from these results" if complete else
                            f"not computed: only {len(evaluated)} of {len(every)} configurations were evaluated; the best configuration of this "
                            "family rests on the study's stored results" if evaluated else "family not evaluated"}
    return coverage


def evaluate_run_groups(arguments, source, target):
    """Run-groups protocol: one aggregated file per run group that has selection results."""
    from evaluation import run_groups
    definition = config.dataset(arguments.dataset)
    fractions = tuple(int(s) / 100 for s in arguments.subset_sizes.split(",")) if arguments.subset_sizes else config.SUBSET_SIZES
    keys = [f"{int(f * 100)}pct" for f in fractions]
    n_top = arguments.n_top if arguments.n_top is not None else 5
    sizes = [int(g) for g in arguments.run_groups.split(",")] if arguments.run_groups else \
        [g for g in definition["run_groups"] if any(source.glob(f"*_{g}runs.json")) or any(source.glob(f"*_{g}runs_w*.json"))]
    manifest = {"dataset": arguments.dataset, "subset_sizes": keys, "n_top": n_top, "run_groups": {},
                "statistics_across_methods": "every ranking block lists the methods of its Holm corrections and dominance ranks under 'methods_covered'"}
    for size in sizes:
        result, families = run_groups.evaluate_run_group(source, size, keys, n_top)
        folder = target if size == definition["reported_run_group"] else target / f"run_group_{size}"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "aggregated_results.json", "w") as handle:
            json.dump(result, handle, cls=NumpyEncoder, indent=1)
        manifest["run_groups"][str(size)] = {"file": str((folder / "aggregated_results.json").relative_to(target)),
                                             "methods_per_family": {family: len(methods) for family, methods in families.items()}}
        print(f"  {size}-run group: {sum(len(m) for m in families.values())} methods -> {folder / 'aggregated_results.json'}")
    (target / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(config.DATASETS))
    parser.add_argument("--input", type=Path, default=None, help="selection results (default: data/<dataset>/selection)")
    parser.add_argument("--output", type=Path, default=None, help="default: data/<dataset>/results")
    parser.add_argument("--windows", default=None, help="synthetic-distributions protocol only: window sizes, e.g. 1,2,3 (default: all with results)")
    parser.add_argument("--subset-sizes", default=None, help="subset fractions in percent, e.g. 5,10,20,30 (default: all four)")
    parser.add_argument("--n-top", type=int, default=None,
                        help="number of methods compared in the ranking blocks (default: 30; 5 for the run-groups protocol, as in the study)")
    parser.add_argument("--run-groups", default=None, help="run-groups protocol only: run-group sizes to evaluate, e.g. 10 or 5,6 (default: all with results)")
    parser.add_argument("--gzip", action="store_true", help="synthetic-distributions protocol only: write aggregated_results.json.gz")
    arguments = parser.parse_args()
    source = arguments.input or config.stage_dir("selection", arguments.dataset)
    target = arguments.output or config.stage_dir("results", arguments.dataset)
    if config.dataset(arguments.dataset)["protocol"] == "run_groups":
        evaluate_run_groups(arguments, source, target)
        return
    windows = [int(w) for w in arguments.windows.split(",")] if arguments.windows else inputs.available_windows(source)
    fractions = tuple(int(s) / 100 for s in arguments.subset_sizes.split(",")) if arguments.subset_sizes else config.SUBSET_SIZES
    keys = [f"{int(f * 100)}pct" for f in fractions]

    arguments.n_top = arguments.n_top if arguments.n_top is not None else 30
    tables = {w: inputs.load_window(source, w) for w in windows}
    draws = {w: d for w in windows if (d := inputs.load_draws(source, w)) is not None}
    comparisons = {w: c for w in windows if (c := inputs.load_comparisons(source, w)) is not None}
    print(f"{arguments.dataset}: windows {windows}; {tables[windows[0]]['n_distributions']} distributions; "
          f"{ {w: len(t['methods']) for w, t in tables.items()} } methods per window; draws for windows {sorted(draws)}")

    overall = aggregated.aggregate_windows(tables)
    result = {"_meta": {**META, "n_top": arguments.n_top}, "analyses": {}}
    result["analyses"]["overall"] = aggregated.analysis_block(overall, keys, arguments.n_top, aggregated.overall_comparisons(comparisons, keys), draws)
    for window in windows:
        result["analyses"][f"w{window}"] = aggregated.analysis_block(tables[window], keys, arguments.n_top, comparisons.get(window), draws)
    result["sensitivity"] = aggregated.sensitivity(tables, keys)

    target.mkdir(parents=True, exist_ok=True)
    path = target / ("aggregated_results.json.gz" if arguments.gzip else "aggregated_results.json")
    with (gzip.open(path, "wt", encoding="utf-8") if arguments.gzip else open(path, "w")) as handle:
        json.dump(result, handle, cls=NumpyEncoder, indent=1)
    coverage = family_coverage(tables)
    manifest = {"dataset": arguments.dataset, "input": str(source), "windows": windows, "subset_sizes": keys, "n_top": arguments.n_top,
                "n_distributions": tables[windows[0]]["n_distributions"], "families": coverage,
                "statistics_across_methods": "every ranking block lists the methods of its Holm corrections and dominance ranks under 'methods_covered'"}
    (target / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {path} and evaluation_manifest.json; complete families: {[f for f, c in coverage.items() if c['complete']]}")


if __name__ == "__main__":
    main()
