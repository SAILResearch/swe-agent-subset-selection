"""Dataset definitions of the pipeline.

Each dataset names its trajectory format (parser), its sanitization rules module and its runs in
chronological order. Stage scripts take ``--dataset`` and read everything else from here.
Default working folders are ``data/<dataset>/<stage>/`` under the package root (not tracked).
"""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"

SEED = 42            # base seed of every random draw
N_SEEDS = 500        # independent draws per stochastic method and temporal split
SUBSET_SIZES = (0.05, 0.10, 0.20, 0.30)
# Hugging Face dataset repository with the published vectors: <dataset>_vectors.tar.gz, <dataset>_vectors.md5 (md5 of
# every file in the archive) and single_setup_repository_map.json.
VECTORS_REPOSITORY = "Mahmoud-queens/swe-agent-subset-selection-vectors"
DEFAULT_RUN_TRAJECTORY_COUNT = 500   # raw trajectories of a SWE-Bench Verified run, unless the dataset lists another number

DATASETS = {
    "multi_model": {
        # Paper name: Multi-model. Source: SWE-bench experiments repository, evaluation/verified/<run>.
        "parser": "openhands",
        "protocol": "synthetic_distributions",
        "sanitization_rules": "multi_model",
        # Runs in chronological (submission) order; the first is the source run of the synthetic distributions.
        "runs": (
            "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
            "20250415_openhands",
            "20250520_openhands_devstral_small",
            "20250524_openhands_claude_4_sonnet",
            "20250716_openhands_kimi_k2",
            "20250807_openhands_gpt5",
        ),
    },
    "multi_agent": {
        # Paper name: Multi-agent. Source: SWE-bench experiments repository, evaluation/verified/<run>.
        "parser": "openhands",
        # Trajectory format of the runs that are not OpenHands runs.
        "run_parsers": {
            "20250519_trae": "trae",
            "20250603_Refact_Agent_claude-4-sonnet": "refact",
            "20250611_moatless_claude-4-sonnet-20250514": "moatless",
            "20250612_trae": "trae",
            "20250720_Lingxi-v1.5_claude-4-sonnet-20250514": "lingxi",
        },
        "protocol": "synthetic_distributions",
        "sanitization_rules": "multi_agent",
        # The source run is the same run as in the multi-model dataset and is sanitized with that dataset's rules,
        # so that both datasets use the same sanitized trajectories and vectors of this run.
        "run_sanitization_rules": {"20241029_OpenHands-CodeAct-2.1-sonnet-20241022": "multi_model"},
        # Raw trajectories of the runs that do not hold exactly one trajectory per instance (checked by fetch.py).
        "run_trajectory_counts": {"20250611_moatless_claude-4-sonnet-20250514": 501, "20250612_trae": 499},
        "runs": (
            "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
            "20250519_trae",
            "20250603_Refact_Agent_claude-4-sonnet",
            "20250611_moatless_claude-4-sonnet-20250514",
            "20250612_trae",
            "20250716_openhands_kimi_k2",
            "20250720_Lingxi-v1.5_claude-4-sonnet-20250514",
        ),
    },
    "single_setup": {
        # Paper name: Single-setup. Source: the Hugging Face dataset below, at the pinned revision.
        "hub_dataset": "nebius/SWE-rebench-openhands-trajectories",
        "hub_revision": "35455389ab51bf5e2306bfd436ef72d0f98bf882",
        "parser": "rebench_openhands",
        "protocol": "run_groups",
        "sanitization_rules": "single_setup",
        # Run groups used in the study: instances with exactly N reruns, N = 5..10 (see group_runs.py).
        "run_groups": (5, 6, 7, 8, 9, 10),
        # The paper reports the 10-run group; its results are written to the top of the result folder.
        "reported_run_group": 10,
    },
}


def dataset(name):
    """Definition of a dataset; stops with the list of available datasets when the name is unknown."""
    if name not in DATASETS:
        raise SystemExit(f"Unknown dataset '{name}'. Available: {', '.join(sorted(DATASETS))}")
    return DATASETS[name]


def runs_of(name, run_groups=None):
    """Run folders of a dataset relative to its stage folder: the runs, or ``<N>_runs/run_<i>`` for run groups."""
    definition = dataset(name)
    if "run_groups" not in definition:
        return list(definition["runs"])
    return [f"{size}_runs/run_{run}" for size in (run_groups or definition["run_groups"]) for run in range(1, size + 1)]


def repository_map(name):
    """Default location of a run-group dataset's repository map: data/<dataset>/repository_map.json."""
    return DATA_DIR / name / "repository_map.json"


def stage_dir(stage, name):
    """Default folder of a stage's data for a dataset, e.g. data/multi_model/parsed."""
    return DATA_DIR / name / stage
