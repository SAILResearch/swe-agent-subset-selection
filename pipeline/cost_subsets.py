"""Cost stage 3: token cost of the subsets selected by one method (paper Finding 4.2).

    python pipeline/cost_subsets.py --dataset multi_model [--selection SELECTION_DIR] [--distributions-file FILE]
                                    [--input RESULT_DIR] [--output RESULT_DIR]

Reads ``cost_per_trajectory.json`` from ``RESULT_DIR``, the selection results
``SELECTION_DIR/embedding_strata_w<W>.json`` and the distributions file the selection was run on (defaults
data/<dataset>/results, data/<dataset>/selection and data/<dataset>/distributions/synthetic_distributions.json),
and writes ``cost_centroid_pooled_10pct.json``.

For every distribution, temporal split and test run, the cost of evaluating the run on the selected subset is
compared with the cost of evaluating it on the whole distribution. Selected subsets are positions inside a
distribution, so the distributions file must be the one the selection used; when the selection folder holds the
``manifest.json`` of run_selection.py, the md5 recorded there is checked. ``summary`` and ``records`` describe
window size 2, the window reported in the paper; ``per_window`` and ``all_windows`` give the same totals for every
available window size.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evaluation import inputs  # noqa: E402
from selection import data  # noqa: E402

FAMILY = "embedding_strata"
METHOD = "centroid_pooled"
SUBSET = "10pct"
REPORTED_WINDOW = 2


def check_distributions_file(path, selection_dir):
    """Stop if run_selection.py recorded a different distributions file for the selection results."""
    manifest = selection_dir / "manifest.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text())["distributions_file_md5"]
        if hashlib.md5(path.read_bytes()).hexdigest() != recorded:
            raise SystemExit(f"{path} is not the distributions file of the selection results in {selection_dir} (md5 {recorded})")


def savings_records(selection, distributions, costs):
    """One record per (distribution, temporal split, test run): cost of the population, of the subset, and the saving."""
    instance_ids, run_names = selection["instance_ids"], selection["run_names"]
    subsets = selection["results"][SUBSET][METHOD]["selected_subsets"]
    records = []
    for index, position in enumerate(selection["distribution_positions"]):
        _, level, members = distributions[position]
        population = [instance_ids[i] for i in members]
        for split_index, split in enumerate(selection["splits"]):
            selected = [population[i] for i in subsets[index][split_index]]
            for run_index in split["test"]:
                run_costs = costs[run_names[run_index]]
                full_cost = sum(run_costs[i] for i in population if i in run_costs)
                subset_cost = sum(run_costs[i] for i in selected if i in run_costs)
                saved = full_cost - subset_cost
                saved_pct = (saved / full_cost * 100) if full_cost > 0 else 0.0
                records.append({"dist_idx": index, "bucket": level, "fold_idx": split_index, "test_agent": run_names[run_index],
                                "pop_size": len(population), "subset_size": len(selected), "full_cost_tokens": full_cost,
                                "subset_cost_tokens": subset_cost, "saved_tokens": saved, "saved_pct": round(saved_pct, 2),
                                "missing_full": sum(i not in run_costs for i in population),
                                "missing_subset": sum(i not in run_costs for i in selected)})
    return records


def savings_summary(records):
    """Savings over a list of records: statistics of the per-record percentage and the token-weighted aggregate."""
    saved_pcts = [r["saved_pct"] for r in records]
    full = sum(r["full_cost_tokens"] for r in records)
    saved = sum(r["saved_tokens"] for r in records)
    return {"mean_saved_pct": round(float(np.mean(saved_pcts)), 2), "median_saved_pct": round(float(np.median(saved_pcts)), 2),
            "min_saved_pct": round(float(np.min(saved_pcts)), 2), "max_saved_pct": round(float(np.max(saved_pcts)), 2),
            "std_saved_pct": round(float(np.std(saved_pcts)), 2), "total_full_tokens": int(full),
            "total_subset_tokens": int(sum(r["subset_cost_tokens"] for r in records)), "total_saved_tokens": int(saved),
            "aggregate_saved_pct": round(saved / full * 100, 2) if full > 0 else 0}


def grouped_means(records, key):
    """Mean saving and record count per value of one record field."""
    groups = defaultdict(list)
    for record in records:
        groups[record[key]].append(record["saved_pct"])
    return {str(value): {"mean_saved_pct": round(float(np.mean(pcts)), 2), "n_records": len(pcts)}
            for value, pcts in sorted(groups.items())}


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=["multi_model"])   # the dataset with a cost per trajectory
    parser.add_argument("--selection", type=Path, default=None, help="selection results (default: data/<dataset>/selection)")
    parser.add_argument("--distributions-file", type=Path, default=None, help="default: data/<dataset>/distributions/synthetic_distributions.json")
    parser.add_argument("--input", type=Path, default=None, help="folder of cost_per_trajectory.json (default: data/<dataset>/results)")
    parser.add_argument("--output", type=Path, default=None, help="default: data/<dataset>/results")
    arguments = parser.parse_args()
    selection_dir = arguments.selection or config.stage_dir("selection", arguments.dataset)
    source = arguments.input or config.stage_dir("results", arguments.dataset)
    target = arguments.output or config.stage_dir("results", arguments.dataset)
    distributions_file = arguments.distributions_file or config.stage_dir("distributions", arguments.dataset) / "synthetic_distributions.json"
    check_distributions_file(distributions_file, selection_dir)
    stored = json.loads((source / "cost_per_trajectory.json").read_text(encoding="utf-8"))
    costs = {run: {instance: cost["triangular_total_tokens"] for instance, cost in agent["trajectories"].items()}
             for run, agent in stored["agents"].items()}
    per_window, everything, reported = {}, [], None
    for window in inputs.available_windows(selection_dir):
        path = inputs.single_file(selection_dir, f"{FAMILY}_w{window}", ".json")
        if path is None:
            continue
        selection = json.loads(path.read_text(encoding="utf-8"))
        if METHOD not in selection["results"].get(SUBSET, {}):
            raise SystemExit(f"{path.name} has no results of {METHOD} at {SUBSET}")
        records = savings_records(selection, data.load_distributions(distributions_file, selection["instance_ids"]), costs)
        per_window[str(window)] = savings_summary(records)
        everything.extend(records)
        if window == REPORTED_WINDOW:
            reported = records
    if reported is None:
        raise SystemExit(f"No {FAMILY}_w{REPORTED_WINDOW}.json in {selection_dir}: the summary describes window size {REPORTED_WINDOW}")
    summary = {"method": METHOD, "subset_pct": SUBSET, "window": REPORTED_WINDOW, "n_records": len(reported),
               "overall": savings_summary(reported), "per_agent": grouped_means(reported, "test_agent"),
               "per_bucket": grouped_means(reported, "bucket")}
    output = {"metadata": {"cost_estimates_file": "cost_per_trajectory.json", "selection_file": f"{FAMILY}_w{REPORTED_WINDOW}.json",
                           "method": METHOD, "subset_pct": SUBSET, "window": REPORTED_WINDOW},
              "summary": summary, "per_window": per_window, "all_windows": savings_summary(everything), "records": reported}
    target.mkdir(parents=True, exist_ok=True)
    (target / "cost_centroid_pooled_10pct.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"{arguments.dataset}: wrote cost_centroid_pooled_10pct.json; subset cost share "
          f"{100 - summary['overall']['aggregate_saved_pct']:.2f}% at W = {REPORTED_WINDOW}, "
          f"{100 - output['all_windows']['aggregate_saved_pct']:.2f}% over windows {', '.join(per_window)}")


if __name__ == "__main__":
    main()
