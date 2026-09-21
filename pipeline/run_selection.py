"""Resumable driver of the selection stage: runs it in blocks of distributions and assembles the result files.

    python pipeline/run_selection.py --dataset multi_model --distributions-file FILE --output RESULT_DIR
                 [--families baselines,embedding_strata,determinism,pure_embedding,shortlist,clustering]
                 [--methods shortlist:centroid_pooled_strata_15x,clustering:hdbscan_pooled_strata]
                 [--windows 1,2] [--jobs N] [--block-size 100] [--distributions 0-199] [--keep-going]

One unit of work is (family, window, block of distributions); ``determinism`` stands for the per-seed baseline
draws. Families run in the order given. A family named in ``--methods`` (as ``family:method``) evaluates only those
methods; the other families are evaluated in full. Each unit runs the public entry point of its family in a
separate process, writes its result under ``<output>/blocks/`` and then a completion record
(``<unit>.done.json``). Units with a completion record that covers the requested methods are skipped, so an
interrupted run is resumed by repeating the same command; a unit that covers fewer methods is run again. When all
blocks of a family and window are complete they are joined into ``<output>/<family>_w<W>.json``; at the end the comparisons of the determinism stage are computed from the joined
files. ``<output>/manifest.json`` lists every unit and joined file: md5 of the output, wall-clock time, peak
memory, md5 of the distributions file and the code version (md5 over the source files of the selection stage).
Random draws are seeded by distribution position, split and draw number, so the blocks give the values of a
single run.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))
import config  # noqa: E402
import protocols  # noqa: E402
from selection.families import FAMILIES, chosen_methods  # noqa: E402
from selection.options import parse_positions  # noqa: E402

DRAWS = "determinism"
ENTRY_POINTS = {"baselines": "select_baselines.py", "embedding_strata": "select_embedding.py", "pure_embedding": "select_ablations.py",
                "clustering": "select_ablations.py", "shortlist": "select_ablations.py", DRAWS: "determinism.py"}
SOURCE_PATTERNS = ("config.py", "select_*.py", "determinism.py", "run_selection.py", "selection/*.py", "protocols/*.py")


def md5_of(path):
    """md5 of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version():
    """md5 over the source files of the selection stage (names and contents, in sorted order)."""
    digest = hashlib.md5()
    for path in sorted(p for pattern in SOURCE_PATTERNS for p in PIPELINE_DIR.glob(pattern)):
        digest.update(path.relative_to(PIPELINE_DIR).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, payload):
    """Write a JSON file atomically."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def parse_arguments():
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True,   # units are blocks of synthetic distributions
                        choices=sorted(n for n, d in config.DATASETS.items() if d["protocol"] == "synthetic_distributions"))
    parser.add_argument("--families", default=",".join([*FAMILIES, DRAWS]), help=f"comma-separated, run in this order; '{DRAWS}' = per-seed draws")
    parser.add_argument("--methods", default=None, help="family:method,... (families not named here run in full)")
    parser.add_argument("--windows", default=None, help="window sizes, e.g. 1,2,3 (default: all, 1 to runs-1)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker processes")
    parser.add_argument("--distributions-file", type=Path, required=True, help="synthetic distributions of the dataset")
    parser.add_argument("--output", type=Path, required=True, help="result folder; its blocks/ sub-folder holds the units")
    parser.add_argument("--block-size", type=int, default=100, help="distributions per unit (default: 100)")
    parser.add_argument("--distributions", default=None, help="positions to evaluate, e.g. 0-199 (default: all)")
    parser.add_argument("--seeds", type=int, default=config.N_SEEDS, help="random draws per stochastic method and split")
    parser.add_argument("--input", type=Path, default=None, help="vector folder with pooled/ and timeseries/ (default: data/<dataset>/vectors)")
    parser.add_argument("--features", type=Path, default=None, help="clustering: trajectory_features.npz (default: data/<dataset>/features/)")
    parser.add_argument("--features-meta", type=Path, default=None, help="clustering: trajectory_features_meta.json (default: next to --features)")
    parser.add_argument("--keep-going", action="store_true", help="after a failed unit continue with the remaining units")
    arguments = parser.parse_args()
    arguments.families = arguments.families.split(",")
    unknown = [f for f in arguments.families if f not in ENTRY_POINTS]
    if unknown:
        raise SystemExit(f"Unknown families: {', '.join(unknown)}. Available: {', '.join(ENTRY_POINTS)}")
    arguments.methods = arguments.methods.split(",") if arguments.methods else []
    for name in arguments.methods:
        family, _, method = name.partition(":")
        if family not in FAMILIES or not any(method in FAMILIES[family][0](w) for w in (1, 2)):
            raise SystemExit(f"--methods takes family:method names of existing methods; got '{name}'")
    return arguments


def unit_command(arguments, family, window, first, last, blocks_dir):
    """Command line of one unit."""
    command = [sys.executable, "-B", str(PIPELINE_DIR / ENTRY_POINTS[family]), "--dataset", arguments.dataset, "--windows", str(window),
               "--distributions", f"{first}-{last}", "--distributions-file", str(arguments.distributions_file), "--output", str(blocks_dir),
               "--seeds", str(arguments.seeds), "--jobs", str(arguments.jobs)]
    if arguments.input:
        command += ["--input", str(arguments.input)]
    if family == DRAWS:
        return command + ["--draws-only", "--keep-float64"]
    if ENTRY_POINTS[family] == "select_ablations.py":
        command += ["--families", family]
        for option, value in (("--features", arguments.features), ("--features-meta", arguments.features_meta)):
            if value:
                command += [option, str(value)]
    methods = [name for name in arguments.methods if name.startswith(f"{family}:")]
    return command + (["--methods", ",".join(methods)] if methods else [])


def run_unit(arguments, unit, blocks_dir, logs_dir, constants):
    """Run one unit unless its completion record exists; return its completion record (None when it failed)."""
    family, window, first, last = unit
    protocol = protocols.load(config.dataset(arguments.dataset)["protocol"])
    stem = f"{family}_w{window}{protocol.file_suffix(list(range(first, last + 1)), constants['n_distributions'])}"
    output = blocks_dir / (stem + (".npz" if family == DRAWS else ".json"))
    record_path = blocks_dir / f"{stem}.done.json"
    methods = sorted(name for name in arguments.methods if name.startswith(f"{family}:")) or "all"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        for field, value in (("distributions_file_md5", constants["distributions_file_md5"]), ("seeds", arguments.seeds)):
            if record[field] != value:
                raise SystemExit(f"{record_path.name} was produced with another '{field}' ({record[field]} vs {value}); use another --output")
        covered = record["methods"] == "all" or (methods != "all" and set(methods) <= set(record["methods"]))
        if covered and output.exists() and output.stat().st_size == record["output_bytes"]:
            if record["code_version"] != constants["code_version"]:
                print(f"  {stem}: reused; it was produced by code version {record['code_version']}")
            return record
    started = time.time()
    with open(logs_dir / f"{stem}.log", "w") as log:
        status = subprocess.run(unit_command(arguments, family, window, first, last, blocks_dir), stdout=log, stderr=subprocess.STDOUT).returncode
    if status != 0 or not output.exists():
        print(f"  FAILED {stem} (exit status {status}); see {logs_dir / (stem + '.log')}", flush=True)
        return None
    runlog = json.loads(output.with_suffix(".runlog.json").read_text())
    record = {"unit": stem, "family": family, "methods": methods, "window": window, "first_distribution": first, "last_distribution": last,
              "seeds": arguments.seeds, "output": output.name, "output_bytes": output.stat().st_size, "output_md5": md5_of(output),
              "wall_clock_seconds": round(time.time() - started, 1), "peak_memory_mb_main_process": runlog["peak_memory_mb_main_process"],
              "peak_memory_mb_largest_worker": runlog["peak_memory_mb_largest_worker"], "jobs": arguments.jobs,
              "distributions_file_md5": constants["distributions_file_md5"], "code_version": constants["code_version"],
              "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(record_path, record)
    print(f"  done {stem}: {record['wall_clock_seconds']} s, {record['output_bytes'] / 1e6:.1f} MB", flush=True)
    return record


def join_blocks(arguments, family, window, records, positions, constants, joined):
    """Join the blocks of one family and window; skipped when the joined file is newer than its blocks."""
    protocol = protocols.load(config.dataset(arguments.dataset)["protocol"])
    blocks_dir = arguments.output / "blocks"
    stem = f"{family}_w{window}{protocol.file_suffix(positions, constants['n_distributions'])}"
    target = arguments.output / (stem + (".npz" if family == DRAWS else ".json"))
    sources = [blocks_dir / r["output"] for r in records]
    if not target.exists() or target.stat().st_mtime < max(s.stat().st_mtime for s in sources):
        if family == DRAWS:
            protocol.assemble_draw_blocks(sources, target)
            protocol.assemble_draw_blocks([s.with_suffix(".float64.npz") for s in sources], target.with_suffix(".float64.npz"))
        else:
            protocol.assemble_blocks(sources, target)
        print(f"  joined {target.name} from {len(sources)} blocks", flush=True)
    joined.append({"file": target.name, "family": family, "window": window, "methods": records[0]["methods"], "blocks": [r["unit"] for r in records],
                   "distributions": len(positions), "md5": md5_of(target), "bytes": target.stat().st_size})


def write_comparisons(arguments, windows, positions, constants, joined):
    """Comparisons of the determinism stage from the joined draws and the joined family files of each window."""
    import numpy as np
    protocol = protocols.load(config.dataset(arguments.dataset)["protocol"])
    suffix = protocol.file_suffix(positions, constants["n_distributions"])
    for window in windows:
        draws_file = arguments.output / f"{DRAWS}_w{window}{suffix}.float64.npz"
        if not draws_file.exists():
            continue
        stored = np.load(arguments.output / f"{DRAWS}_w{window}{suffix}.npz")
        payload = protocol.comparisons_payload(arguments.output, window, config.SUBSET_SIZES, arguments.seeds, positions,
                                               [int(level) for level in stored["distribution_levels"]], dict(np.load(draws_file)), suffix)
        target = arguments.output / f"{DRAWS}_w{window}{suffix}.json"
        target.write_text(json.dumps(payload))
        joined.append({"file": target.name, "family": DRAWS, "window": window, "methods_compared": payload["methods_compared"],
                       "md5": md5_of(target), "bytes": target.stat().st_size})
        print(f"  wrote {target.name}: {len(payload['methods_compared'])} methods compared with the per-seed draws", flush=True)


def main():
    """Run all missing units, join complete families, write the manifest."""
    arguments = parse_arguments()
    blocks_dir, logs_dir = arguments.output / "blocks", arguments.output / "logs"
    blocks_dir.mkdir(parents=True, exist_ok=True), logs_dir.mkdir(parents=True, exist_ok=True)
    n_distributions = len(json.loads(arguments.distributions_file.read_text()))
    positions = parse_positions(arguments.distributions) if arguments.distributions else list(range(n_distributions))
    n_runs = len(config.dataset(arguments.dataset)["runs"])
    windows = [int(w) for w in arguments.windows.split(",")] if arguments.windows else list(range(1, n_runs))
    constants = {"n_distributions": n_distributions, "distributions_file_md5": md5_of(arguments.distributions_file), "code_version": code_version()}
    blocks = [(chunk[0], chunk[-1]) for chunk in (positions[i:i + arguments.block_size] for i in range(0, len(positions), arguments.block_size))]
    print(f"{arguments.dataset}: families {arguments.families}; windows {windows}; {len(positions)} distributions in {len(blocks)} blocks; "
          f"distributions file md5 {constants['distributions_file_md5']}; code version {constants['code_version']}", flush=True)
    units, joined, failed = [], [], []
    manifest = {"dataset": arguments.dataset, "command": " ".join(sys.argv), **constants, "units": units, "joined_files": joined, "failed_units": failed}
    for family in arguments.families:
        for window in windows:
            if family != DRAWS and not chosen_methods(family, window, arguments.methods or None):
                continue   # the requested methods of this family do not exist for this window size
            records = []
            for first, last in blocks:
                record = run_unit(arguments, (family, window, first, last), blocks_dir, logs_dir, constants)
                if record is None:
                    failed.append(f"{family}_w{window}_d{first}-{last}")
                    if not arguments.keep_going:
                        write_json(arguments.output / "manifest.json", manifest)
                        raise SystemExit(1)
                    continue
                records.append(record), units.append(record)
                write_json(arguments.output / "manifest.json", manifest)
            if len(records) == len(blocks):
                join_blocks(arguments, family, window, records, positions, constants, joined)
                write_json(arguments.output / "manifest.json", manifest)
    write_comparisons(arguments, windows, positions, constants, joined)
    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(arguments.output / "manifest.json", manifest)
    print(f"finished: {len(units)} units complete, {len(failed)} failed; manifest: {arguments.output / 'manifest.json'}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
