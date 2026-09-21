"""Stage 0b (single_setup): group the trajectories of ``trajectories.parquet`` into run groups and runs.

    python pipeline/group_runs.py --dataset single_setup [--input RAW_DIR] [--output GROUPED_DIR]
                                  [--run-groups 5,6,7,8,9,10] [--counts-only]

The dataset holds several trajectories (reruns) per instance. Instances are grouped by their number of
trajectories: all instances with exactly N trajectories form the run group ``<N>_runs``; groups with 50 instances or
fewer are dropped. Within a group, run ``i`` consists of the i-th trajectory of every instance, in the row order of
the parquet file. Nothing is random. Each trajectory is written as ``<N>_runs/run_<i>/<instance>.json``
("/" in instance ids replaced by "__"), with a ``stats.txt`` per run group. ``repository_names.json`` lists the
repositories of all run groups of the file (4 to 20 reruns); the sanitize stage removes the parts of these names, so
that the sanitized text of a run group does not depend on which other run groups were written.

The file is read in two streaming passes (first the instance ids, then the rows), one row group at a time (peak memory about 7 GB).
``--counts-only`` prints the instances and trajectories per run group without writing anything.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

REPOSITORY_FILE = "repository_names.json"   # repositories of all run groups of the file; read by the sanitize stage
MIN_INSTANCES_PER_GROUP = 50   # a run group needs more instances than this


def is_resolved(value):
    """True for the dataset's ways of marking a resolved trajectory."""
    return (isinstance(value, int) and value == 1) or (isinstance(value, str) and value.lower() in ("1", "true", "success"))


def group_sizes(parquet):
    """({instance id: number of trajectories}, {group size: number of instances}, repository names) for groups that are kept."""
    per_instance, repository_of = Counter(), {}
    for index in range(parquet.num_row_groups):
        columns = parquet.read_row_group(index, columns=["instance_id", "repo"])
        instances = columns.column("instance_id").to_pylist()
        per_instance.update(instances)
        repository_of.update(zip(instances, columns.column("repo").to_pylist()))
    per_group = Counter(per_instance.values())
    kept = {size: n for size, n in sorted(per_group.items()) if n > MIN_INSTANCES_PER_GROUP}
    repositories = sorted({repository_of[i] or "unknown" for i, size in per_instance.items() if size in kept})
    return per_instance, kept, repositories


def write_groups(parquet, per_instance, sizes, wanted, output_dir):
    """Write the trajectories of the wanted run groups and a ``stats.txt`` per group."""
    seen = Counter()
    resolved = {size: [0] * size for size in wanted}
    repositories = {size: [set() for _ in range(size)] for size in wanted}
    for index in range(parquet.num_row_groups):
        for record in parquet.read_row_group(index).to_pylist():
            instance, size = record["instance_id"], per_instance[record["instance_id"]]
            if size not in wanted:
                continue
            run = seen[instance]
            seen[instance] += 1
            run_dir = output_dir / f"{size}_runs" / f"run_{run + 1}"
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / f"{instance.replace('/', '__')}.json", "w") as handle:
                json.dump(record, handle, indent=2)
            resolved[size][run] += is_resolved(record.get("resolved", 0))
            repositories[size][run].add(record.get("repo", "unknown"))
    for size in wanted:
        n_instances, total = sizes[size], sizes[size] * size
        lines = [f"=== Stats for Group: {size}_runs ===\n", f"Unique Instances: {n_instances}", f"Total Trajectories: {total}",
                 f"Aggregate Resolve Rate: {sum(resolved[size]) / total * 100:.2f}% ({sum(resolved[size])}/{total})", "-" * 40]
        for run in range(size):
            lines += [f"\n[Run {run + 1}]", f"  - Resolve Rate: {resolved[size][run] / n_instances * 100:.2f}% ({resolved[size][run]}/{n_instances})",
                      f"  - Unique Repos: {len(repositories[size][run])}"]
        (output_dir / f"{size}_runs" / "stats.txt").write_text("\n".join(lines) + "\n")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(n for n, d in config.DATASETS.items() if "run_groups" in d))
    parser.add_argument("--input", type=Path, default=None, help="raw folder with trajectories.parquet (default: data/<dataset>/raw)")
    parser.add_argument("--output", type=Path, default=None, help="grouped folder (default: data/<dataset>/grouped)")
    parser.add_argument("--run-groups", default=None, help="group sizes to write, e.g. 10 or 5,6 (default: the dataset's run groups)")
    parser.add_argument("--counts-only", action="store_true", help="print instances and trajectories per run group; write nothing")
    arguments = parser.parse_args()
    definition = config.dataset(arguments.dataset)
    parquet = pq.ParquetFile((arguments.input or config.stage_dir("raw", arguments.dataset)) / "trajectories.parquet")
    per_instance, sizes, repositories = group_sizes(parquet)
    wanted = [int(g) for g in arguments.run_groups.split(",")] if arguments.run_groups else list(definition["run_groups"])
    for size, n_instances in sizes.items():
        print(f"  {size}_runs: {n_instances} instances, {n_instances * size} trajectories{'' if size in definition['run_groups'] else '  (not used in the study)'}")
    used = [s for s in sizes if s in definition["run_groups"]]
    print(f"group_runs: run groups {used}: {sum(sizes[s] for s in used)} instances, {sum(sizes[s] * s for s in used)} trajectories")
    if arguments.counts_only:
        return
    missing = [g for g in wanted if g not in sizes]
    if missing:
        raise SystemExit(f"No run group of size {missing} with more than {MIN_INSTANCES_PER_GROUP} instances")
    output_dir = arguments.output or config.stage_dir("grouped", arguments.dataset)
    write_groups(parquet, per_instance, sizes, wanted, output_dir)
    (output_dir / REPOSITORY_FILE).write_text(json.dumps(repositories, indent=1) + "\n")
    print(f"group_runs: wrote run groups {wanted} to {output_dir}")


if __name__ == "__main__":
    main()
