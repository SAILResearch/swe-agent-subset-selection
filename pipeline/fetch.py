"""Stage 0: download the published vectors or the raw trajectories of a dataset.

    python pipeline/fetch.py --dataset multi_model --vectors [--source FOLDER] [--output DATASET_DIR]
    python pipeline/fetch.py --dataset single_setup [--output RAW_DIR]
    python pipeline/fetch.py --dataset multi_model [--output RAW_DIR] [--repository CLONE_DIR] [--runs RUN,RUN] [--use-cli]

``--vectors``: downloads ``<dataset>_vectors.tar.gz`` and its manifest ``<dataset>_vectors.md5`` from the Hugging Face
dataset repository named in ``config.py`` (or takes them from ``--source``, a local folder with the same files),
unpacks the archive into ``DATASET_DIR`` (default data/<dataset>), which gives ``vectors/`` and, for multi_model and
multi_agent, ``features/``, and compares the md5 of every unpacked file with the manifest. For single_setup it also
places ``single_setup_repository_map.json`` as ``repository_map.json``. A dataset folder that already holds ``vectors/``
is left untouched.

single_setup: downloads the Hugging Face dataset named in ``config.py`` at its pinned revision into ``RAW_DIR``
(default data/single_setup/raw); the trajectories are in ``trajectories.parquet`` (about 2 GB). A folder that
already holds files is left untouched.

multi_model, multi_agent: the runs are submissions to the SWE-bench experiments repository
(https://github.com/SWE-bench/experiments). The repository holds each submission's ``results/``; the trajectories
are downloaded with the repository's own tool, ``python -m analysis.download_logs evaluation/verified/<run>
--only_trajs``. This script clones the repository without file contents into ``CLONE_DIR`` (default
data/swe_bench_experiments) when it is absent, checks out ``analysis/`` and the submissions of the dataset, runs the
download tool for every run, and copies ``results/`` and ``trajs/`` of each run to ``RAW_DIR/<run>/`` (default
data/<dataset>/raw). The number of trajectories of every run is compared with the number used in the study. Runs
already complete in ``RAW_DIR`` are skipped. Requirements: ``git``; the Python packages ``boto3`` and ``pyyaml``
(imported by the download tool, which reads the public storage bucket without an account); with ``--use-cli`` the
download tool calls the AWS command-line tool (``aws s3 cp``) instead, which must then be installed and configured
with an account.
"""
import argparse
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

IGNORED_FILES = (".DS_Store", "desktop.ini", ".gitignore")
EXPERIMENTS_REPOSITORY = "https://github.com/SWE-bench/experiments.git"
SPLIT = "evaluation/verified"


def fetch_hub_dataset(definition, target):
    """Download a Hugging Face dataset repository at its pinned revision, unless the target already holds files."""
    target.mkdir(parents=True, exist_ok=True)
    present = [p.name for p in target.iterdir() if p.name not in IGNORED_FILES]
    if present:
        print(f"fetch: {target} already holds {len(present)} entries ({present[:3]} ...); nothing downloaded")
        return
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=definition["hub_dataset"], repo_type="dataset", revision=definition["hub_revision"], local_dir=target)
    print(f"fetch: downloaded {definition['hub_dataset']} @ {definition['hub_revision']} to {target}")


def published_file(name, source):
    """Path of a file of the vectors repository: taken from the local folder ``source``, or downloaded."""
    if source is not None:
        if not (source / name).is_file():
            raise SystemExit(f"fetch: {name} not found in {source}")
        return source / name
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=config.VECTORS_REPOSITORY, filename=name, repo_type="dataset"))


def fetch_vectors(dataset_name, target, source):
    """Unpack the published vector archive of a dataset into ``target`` and check every file against the md5 manifest."""
    if (target / "vectors").exists():
        print(f"fetch: {target / 'vectors'} already exists; nothing downloaded")
        return
    manifest = published_file(f"{dataset_name}_vectors.md5", source).read_text().splitlines()
    archive = published_file(f"{dataset_name}_vectors.tar.gz", source)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        handle.extractall(target, filter="data")
    wrong = []
    for line in manifest:
        digest, name = line.split("  ", 1)
        path = target / name
        if not path.is_file() or hashlib.md5(path.read_bytes()).hexdigest() != digest:
            wrong.append(name)
    if wrong:
        raise SystemExit(f"fetch: {len(wrong)} of {len(manifest)} files differ from {dataset_name}_vectors.md5, e.g. {wrong[0]}")
    print(f"fetch: unpacked {archive.name} to {target}; {len(manifest)} files match the md5 manifest")
    if "run_groups" in config.dataset(dataset_name):
        shutil.copyfile(published_file(f"{dataset_name}_repository_map.json", source), target / "repository_map.json")
        print(f"fetch: placed {target / 'repository_map.json'}")


def trajectory_count(run_dir):
    """Number of entries in a run's ``trajs`` folder (files, or one folder per instance, depending on the agent)."""
    folder = run_dir / "trajs"
    return sum(1 for p in folder.iterdir() if p.name not in IGNORED_FILES) if folder.is_dir() else 0


def call(command, cwd=None):
    """Run a command and stop with its name when it fails."""
    print("  $ " + " ".join(str(part) for part in command), flush=True)
    if subprocess.run([str(part) for part in command], cwd=cwd).returncode != 0:
        raise SystemExit(f"fetch: command failed: {' '.join(str(part) for part in command)}")


def check_requirements(use_cli):
    """Stop with a clear message when a tool needed for the download is missing."""
    if shutil.which("git") is None:
        raise SystemExit("fetch: git is needed to clone the SWE-bench experiments repository")
    missing = [name for name, module in (("boto3", "boto3"), ("pyyaml", "yaml")) if importlib.util.find_spec(module) is None]
    if missing:
        raise SystemExit(f"fetch: the download tool of the SWE-bench experiments repository needs: pip install {' '.join(missing)}")
    if use_cli and shutil.which("aws") is None:
        raise SystemExit("fetch: --use-cli needs the AWS command-line tool (https://aws.amazon.com/cli/), configured with an account "
                         "(aws configure); without --use-cli the download needs no account")


def prepare_repository(repository, runs):
    """Clone the experiments repository without file contents if absent, and check out the download tool and the runs."""
    if not (repository / ".git").is_dir():
        repository.parent.mkdir(parents=True, exist_ok=True)
        call(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", EXPERIMENTS_REPOSITORY, repository])
    call(["git", "sparse-checkout", "set", "analysis", *[f"{SPLIT}/{run}" for run in runs]], cwd=repository)
    absent = [run for run in runs if not (repository / SPLIT / run).is_dir()]
    if absent:
        raise SystemExit(f"fetch: not found in {EXPERIMENTS_REPOSITORY} under {SPLIT}/: {', '.join(absent)}")


def fetch_experiment_runs(dataset_name, target, repository, runs, use_cli):
    """Download the trajectories of the given runs and copy every run to the raw folder; return the runs with a wrong count."""
    expected = config.dataset(dataset_name).get("run_trajectory_counts", {})
    pending = [run for run in runs if trajectory_count(target / run) != expected.get(run, config.DEFAULT_RUN_TRAJECTORY_COUNT)]
    for run in runs:
        if run not in pending:
            print(f"fetch: {run} is complete in {target}; skipped")
    if pending:
        check_requirements(use_cli)
        prepare_repository(repository, pending)
    wrong = []
    for run in pending:
        call([sys.executable, "-m", "analysis.download_logs", f"{SPLIT}/{run}", "--only_trajs"] + (["--use_cli"] if use_cli else []),
             cwd=repository)
        for part in ("results", "trajs"):
            destination = target / run / part
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(repository / SPLIT / run / part, destination)
        found, wanted = trajectory_count(target / run), expected.get(run, config.DEFAULT_RUN_TRAJECTORY_COUNT)
        print(f"fetch: {run}: {found} trajectories (study: {wanted}){'' if found == wanted else '  <-- differs'}")
        if found != wanted:
            wrong.append(run)
    return wrong


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(config.DATASETS))
    parser.add_argument("--vectors", action="store_true", help="download the published vectors instead of the raw trajectories")
    parser.add_argument("--source", type=Path, default=None, help="with --vectors: local folder holding the files of the vectors repository")
    parser.add_argument("--output", type=Path, default=None, help="raw folder (default: data/<dataset>/raw); with --vectors: data/<dataset>")
    parser.add_argument("--repository", type=Path, default=config.DATA_DIR / "swe_bench_experiments",
                        help="clone of the SWE-bench experiments repository (created if absent)")
    parser.add_argument("--runs", default=None, help="only these runs, comma-separated (default: all runs of the dataset)")
    parser.add_argument("--use-cli", action="store_true", help="let the download tool use the AWS command-line tool")
    arguments = parser.parse_args()
    definition = config.dataset(arguments.dataset)
    if arguments.vectors:
        fetch_vectors(arguments.dataset, arguments.output or config.DATA_DIR / arguments.dataset, arguments.source)
        return
    target = arguments.output or config.stage_dir("raw", arguments.dataset)
    if "hub_dataset" in definition:
        fetch_hub_dataset(definition, target)
        return
    runs = arguments.runs.split(",") if arguments.runs else config.runs_of(arguments.dataset)
    unknown = [run for run in runs if run not in definition["runs"]]
    if unknown:
        raise SystemExit(f"fetch: not a run of {arguments.dataset}: {', '.join(unknown)}")
    wrong = fetch_experiment_runs(arguments.dataset, target, arguments.repository.resolve(), runs, arguments.use_cli)
    if wrong:
        raise SystemExit(f"fetch: trajectory count differs from the study for: {', '.join(wrong)}")


if __name__ == "__main__":
    main()
