"""Entry point of the replication package.

    python reproduce.py results [--rq 1|2|3|4|supporting] [--dry-run]
    python reproduce.py pipeline --dataset NAME (--rq 1|2|3|4|supporting | --all) [--from vectors|raw] [--embed]
                                 [--configurations reported|all] [--windows 1,2] [--distributions 0-99]
                                 [--run-groups 5,6] [--jobs N] [--workdir regenerated] [--dry-run]

``results`` rebuilds tables, figures and reproduced values from the shipped result files in ``results/``.

``pipeline`` runs the pipeline stages that a research question needs for one dataset, starting from the published
vectors (``data/<dataset>/vectors/``) or, with ``--from raw``, from the raw trajectories. Regenerated result files
go to ``<workdir>/results/<dataset>/``; files that the chosen stages do not produce, and the other datasets, are
linked from ``results/``, so that the research-question scripts run on a complete result folder. Their outputs go
to ``<workdir>/outputs/``. Nothing is written into ``results/``. Every mode first prints its plan and an estimated
run time; ``--dry-run`` stops there. Completed selection work in the work folder is reused.

Research question 4 (cost) exists for multi_model only; its cost per trajectory is recomputed when the raw
trajectories are in ``data/multi_model/raw/`` and is otherwise taken from ``results/``. The supporting numbers use
multi_model and single_setup; for multi_model they need the parsed trajectories (``--from raw`` or an existing
``data/multi_model/parsed/``).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / "pipeline"))
import config  # noqa: E402

MACHINE = "Apple M1 Pro, 10 cores, 16 GB, 10 worker processes"

# Research questions: folder, scripts, paper items, the result files the scripts read, and what they need from the
# pipeline: families of the selection stage ("restricted": the one configuration needed of a family), the per-seed
# baseline draws, the evaluation scripts, the cost stages, the distribution-validity stages, and the datasets for which
# pipeline mode is available.
AGGREGATED = ("aggregated_results.json", "per_seed_maxerr_all_baselines.json", "per_seed_maxerr_extract.npz")
RESEARCH_QUESTIONS = {
    "1": {"folder": "rq1_baselines", "scripts": ("reproduce.py",), "families": ("baselines", "embedding_strata"), "draws": True,
          "evaluation": True, "paper": "Tables 3, 4; Figure 5", "reads": AGGREGATED,
          # The per-seed result files report each baseline next to Centroid Pooled, so this one configuration is run too.
          "restricted": {"embedding_strata": "embedding_strata:centroid_pooled"}},
    "2": {"folder": "rq2_trajectory_aware", "scripts": ("reproduce.py",), "families": ("baselines", "embedding_strata"), "draws": True,
          "evaluation": True, "paper": "Tables 5, 6; Figure 6",
          "reads": AGGREGATED + ("per_seed_maxerr_vs_consistency_stratified.json",)},
    "3": {"folder": "rq3_ablation", "scripts": ("reproduce.py",),
          "families": ("baselines", "embedding_strata", "pure_embedding", "shortlist", "clustering"), "draws": True,
          "evaluation": True, "paper": "Table 7; Finding 3.4",
          "reads": ("aggregated_results.json", "determinism_band.json", "per_distribution_mean_rmse.json", "per_seed_maxerr_extract.npz")},
    "4": {"folder": "rq4_cost", "scripts": ("reproduce.py",), "families": ("embedding_strata",), "draws": False,
          "evaluation": False, "cost": True, "datasets": ("multi_model",), "paper": "Tables 8, 9",
          "reads": ("cost_profile.json", "cost_per_trajectory.json", "cost_centroid_pooled_10pct.json"),
          "restricted": {"embedding_strata": "embedding_strata:centroid_pooled"}},
    "supporting": {"folder": "supporting", "scripts": ("reproduce.py", "single_setup_consistency.py"),
                   "families": ("baselines", "embedding_strata"), "draws": False, "evaluation": True, "validity": True,
                   "datasets": ("multi_model", "single_setup"),
                   "paper": "Sections 5.1, 5.2.1, 8.3, 8.4; single-setup across run groups",
                   "reads": ("synthetic_distributions.json", "cross_run_resolve_rates.json", "difficulty_level_validity_per_distribution.csv",
                             "instance_churn_per_distribution.csv", "per_distribution_mean_rmse.json", "cost_profile.json",
                             "aggregated_results.json", "run_groups.json")},
}

# Result files written by each stage that follows the selection stage (run-group datasets: the per-seed stage writes
# the two per-seed files only, and the evaluation also writes run_group_<N>/aggregated_results.json).
STAGE_OUTPUTS = {
    "evaluate.py": ("aggregated_results.json",),
    "evaluate_per_seed.py": ("per_seed_maxerr_all_baselines.json", "per_seed_maxerr_vs_consistency_stratified.json",
                             "per_seed_maxerr_extract.npz", "determinism_band.json", "determinism_band_maxerr.json"),
    "evaluate_distributions.py": ("per_distribution_mean_rmse.json",),
    "summarize_run_groups.py": ("run_groups.json",),
    "cost_per_trajectory.py": ("cost_per_trajectory.json",),
    "cost_profile.py": ("cost_profile.json",),
    "cost_subsets.py": ("cost_centroid_pooled_10pct.json",),
    "distribution_validity.py": ("cross_run_resolve_rates.json", "cross_run_resolve_rates.csv",
                                 "difficulty_level_validity_per_distribution.csv", "instance_churn_per_distribution.csv"),
}
RUN_GROUP_PER_SEED_OUTPUTS = STAGE_OUTPUTS["evaluate_per_seed.py"][:2]

# Configurations reported in the paper for the two ablation families that are not reported in full (Table 7).
REPORTED_CONFIGURATIONS = {"shortlist": "shortlist:centroid_pooled_strata_15x", "clustering": "clustering:hdbscan_pooled_strata"}

# Hours of the selection stage per dataset and family on MACHINE. "measured": a full run; the others are estimates
# scaled from measured parts (see pipeline/README.md). Keys: a family, "draws" (the per-seed baseline draws), or
# (family, "reported" | "all" | "restricted").
SELECTION_HOURS = {
    "multi_model": {"baselines": (1.1, "measured"), "embedding_strata": (1.1, "measured"), "draws": (0.8, "measured"),
                    "pure_embedding": (1.9, "measured"), ("shortlist", "reported"): (0.3, "measured"), ("clustering", "reported"): (0.5, "measured"),
                    ("shortlist", "all"): (4.7, "estimate"), ("clustering", "all"): (18.0, "estimate"),
                    ("embedding_strata", "restricted"): (0.2, "measured")},
    "multi_agent": {"baselines": (2.3, "estimate"), "embedding_strata": (1.9, "estimate"), "draws": (1.7, "estimate"),
                    "pure_embedding": (3.5, "estimate"), ("shortlist", "reported"): (0.6, "estimate"), ("clustering", "reported"): (1.0, "estimate"),
                    ("shortlist", "all"): (9.0, "estimate"), ("clustering", "all"): (33.0, "estimate")},
    "single_setup": {"baselines": (0.2, "estimate"), "embedding_strata": (0.4, "estimate"), "draws": (0.3, "estimate"),
                     "pure_embedding": (0.6, "estimate"), ("shortlist", "reported"): (0.5, "estimate"), ("shortlist", "all"): (0.5, "estimate")},
}
COST_HOURS = (0.01, "measured")       # the three cost stages together
VALIDITY_HOURS = (0.01, "measured")   # distributions generator and validity checks together
RAW_STAGE_HOURS = {"multi_model": 0.1, "multi_agent": 0.1, "single_setup": 0.7}
EMBED_HOURS = {"multi_model": "7 to 35", "multi_agent": "7 to 39", "single_setup": "53 to 300"}


def run(command, dry_run=False):
    """Print and run one command of the plan; stop at the first failure."""
    print("  $ " + " ".join(str(part) for part in command), flush=True)
    if not dry_run and subprocess.run([str(part) for part in command]).returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(str(part) for part in command)}")


def question_commands(questions, results_dir=None, outputs_dir=None):
    """Commands of the research-question scripts."""
    commands = []
    for question in questions:
        entry = RESEARCH_QUESTIONS[question]
        for script in entry["scripts"]:
            command = [sys.executable, "-B", PACKAGE_ROOT / entry["folder"] / script]
            if results_dir is not None:
                target = outputs_dir / entry["folder"] / ("single_setup_consistency" if script == "single_setup_consistency.py" else "")
                command += ["--results", results_dir, "--outputs", target]
            commands.append(command)
    return commands


def results_mode(options):
    """Rebuild outputs from the shipped result files."""
    questions = [options.rq] if options.rq else list(RESEARCH_QUESTIONS)
    print(f"Plan: rebuild the outputs of research question(s) {', '.join(questions)} from results/ "
          f"({'; '.join(RESEARCH_QUESTIONS[q]['paper'] for q in questions)}).")
    print(f"Estimated time: under 1 minute in total ({MACHINE}). Nothing needs to be downloaded.")
    for command in question_commands(questions):
        run(command, options.dry_run)


def families_of(questions, dataset, configurations):
    """Selection families (in stage order) and the methods restriction for a set of research questions."""
    protocol = config.dataset(dataset)["protocol"]
    wanted = [f for f in ("baselines", "embedding_strata", "pure_embedding", "shortlist", "clustering")
              if any(f in RESEARCH_QUESTIONS[q]["families"] for q in questions) and not (f == "clustering" and protocol == "run_groups")]
    methods = [REPORTED_CONFIGURATIONS[f] for f in wanted if configurations == "reported" and f in REPORTED_CONFIGURATIONS]
    for family in wanted:
        using = [RESEARCH_QUESTIONS[q] for q in questions if family in RESEARCH_QUESTIONS[q]["families"]]
        if all(family in entry.get("restricted", {}) for entry in using):
            methods.append(using[0]["restricted"][family])
    return wanted, methods


# Window sizes a research-question script cannot do without, checked when --windows restricts a run.
REQUIRED_WINDOWS = {"1": ("W = 1 and at least one W >= 2 (Consist-Strat needs two training runs)", lambda w: 1 in w and max(w) >= 2),
                    "4": ("W = 2 (the window of the reported cost share)", lambda w: 2 in w)}


def check_windows(questions, windows):
    """Stop before any work when a --windows restriction leaves out window sizes a chosen research question reads."""
    chosen = [int(w) for w in windows.split(",")]
    for question in questions:
        need, satisfied = REQUIRED_WINDOWS.get(question, ("", lambda w: True))
        if not satisfied(chosen):
            raise SystemExit(f"Research question {question} reads {need}; --windows {windows} does not contain them")


def pipeline_questions(dataset):
    """Research questions for which pipeline mode is available on a dataset."""
    return [q for q, entry in RESEARCH_QUESTIONS.items() if dataset in entry.get("datasets", config.DATASETS)]


def estimate(dataset, questions, configurations, from_raw):
    """(hours, note) of a plan from the run-time table."""
    families, methods = families_of(questions, dataset, configurations)
    draws = any(RESEARCH_QUESTIONS[q]["draws"] for q in questions)
    table, hours, kinds = SELECTION_HOURS[dataset], 0.0, set()
    for family in list(families) + (["draws"] if draws else []):
        restricted = any(m.startswith(family + ":") for m in methods) and (family, "restricted") in table
        value, kind = table[(family, "restricted")] if restricted else table.get(family) or table[(family, configurations)]
        hours, kinds = hours + value, kinds | {kind}
    extras = [("cost", COST_HOURS)] + ([("validity", VALIDITY_HOURS)] if "run_groups" not in config.dataset(dataset) else [])
    for key, extra in extras:
        if any(RESEARCH_QUESTIONS[q].get(key) for q in questions):
            hours, kinds = hours + extra[0], kinds | {extra[1]}
    hours += 0.05 + (RAW_STAGE_HOURS[dataset] if from_raw else 0.0)
    return hours, "measured" if kinds == {"measured"} else "partly estimated" if "measured" in kinds else "estimated"


def selection_commands(options, families, methods, draws, selection_dir):
    """Commands of the selection stage for the dataset's protocol."""
    dataset, pipeline = options.dataset, PACKAGE_ROOT / "pipeline"
    vectors = PACKAGE_ROOT / "data" / dataset / "vectors"
    common = ["--dataset", dataset, "--jobs", options.jobs, "--input", vectors] + (["--windows", options.windows] if options.windows else [])
    if config.dataset(dataset)["protocol"] == "synthetic_distributions":
        command = [sys.executable, "-B", pipeline / "run_selection.py", *common, "--output", selection_dir,
                   "--distributions-file", PACKAGE_ROOT / "results" / dataset / "synthetic_distributions.json",
                   # "determinism" is the selection family that produces the per-seed baseline draws
                   "--families", ",".join(list(families) + (["determinism"] if draws else []))]
        command += ["--methods", ",".join(methods)] if methods else []
        command += ["--distributions", options.distributions, "--block-size", "100"] if options.distributions else []
        return [command]
    common += ["--output", selection_dir, "--repository-map", vectors.parent / "repository_map.json"]
    common += ["--run-groups", options.run_groups] if options.run_groups else []
    scripts = {"baselines": "select_baselines.py", "embedding_strata": "select_embedding.py"}
    commands = [[sys.executable, "-B", pipeline / scripts[f], *common] for f in families if f in scripts]
    ablations = [f for f in families if f not in scripts]
    if ablations:
        commands.append([sys.executable, "-B", pipeline / "select_ablations.py", *common, "--families", ",".join(ablations)]
                        + (["--methods", ",".join(methods)] if methods else []))
    if draws:
        commands.append([sys.executable, "-B", pipeline / "determinism.py", *common])
    return commands


def raw_commands(options):
    """Commands from the raw trajectories to the sanitized ones (and, with --embed, the vectors)."""
    dataset, pipeline = options.dataset, PACKAGE_ROOT / "pipeline"
    commands = [[sys.executable, "-B", pipeline / "fetch.py", "--dataset", dataset]]
    if "run_groups" in config.dataset(dataset):
        commands += [[sys.executable, "-B", pipeline / "group_runs.py", "--dataset", dataset] + (["--run-groups", options.run_groups] if options.run_groups else []),
                     [sys.executable, "-B", pipeline / "parse.py", "--dataset", dataset, "--input", config.stage_dir("grouped", dataset), "--validate"]
                     + (["--run-groups", options.run_groups] if options.run_groups else [])]
    else:
        commands.append([sys.executable, "-B", pipeline / "parse.py", "--dataset", dataset, "--validate"])
    commands.append([sys.executable, "-B", pipeline / "sanitize.py", "--dataset", dataset])
    if "run_groups" in config.dataset(dataset):
        commands.append([sys.executable, "-B", pipeline / "build_repository_map.py", "--dataset", dataset])
    else:
        commands.append([sys.executable, "-B", pipeline / "extract_features.py", "--dataset", dataset])
    if options.embed:
        commands.append([sys.executable, "-B", pipeline / "embed.py", "--dataset", dataset])
    return commands


def cost_commands(options, selection_dir, regenerated):
    """Commands of the cost stages; the cost per trajectory is recomputed only when the raw trajectories are present."""
    dataset, pipeline = options.dataset, PACKAGE_ROOT / "pipeline"
    commands, costs = [], PACKAGE_ROOT / "results" / dataset
    if options.source == "raw" or config.stage_dir("raw", dataset).is_dir():
        commands.append([sys.executable, "-B", pipeline / "cost_per_trajectory.py", "--dataset", dataset, "--output", regenerated])
        costs = regenerated
    commands.append([sys.executable, "-B", pipeline / "cost_profile.py", "--dataset", dataset, "--input", costs, "--output", regenerated])
    commands.append([sys.executable, "-B", pipeline / "cost_subsets.py", "--dataset", dataset, "--selection", selection_dir, "--input", costs,
                     "--distributions-file", PACKAGE_ROOT / "results" / dataset / "synthetic_distributions.json", "--output", regenerated])
    return commands


def validity_commands(options, regenerated):
    """Commands of the distributions generator and the validity checks (datasets with synthetic distributions)."""
    dataset, pipeline = options.dataset, PACKAGE_ROOT / "pipeline"
    if options.source != "raw" and not config.stage_dir("parsed", dataset).is_dir():
        raise SystemExit(f"The supporting numbers of {dataset} need the parsed trajectories: use --from raw, or place them in "
                         f"{config.stage_dir('parsed', dataset)}")
    return [[sys.executable, "-B", pipeline / "generate_distributions.py", "--dataset", dataset,
             "--output", regenerated.parent / f"{dataset}_generated_distributions.json"],
            [sys.executable, "-B", pipeline / "distribution_validity.py", "--dataset", dataset, "--output", regenerated,
             "--distributions-file", PACKAGE_ROOT / "results" / dataset / "synthetic_distributions.json"]]


def check_stage_outputs(commands, regenerated, synthetic, complete):
    """Stop when a stage did not write one of its result files.

    Run-group datasets write the top-level aggregated results only when the reported run group with all its window
    sizes was evaluated (``complete``).
    """
    for command in commands:
        script = Path(command[2]).name
        expected = STAGE_OUTPUTS.get(script, ())
        if not synthetic:
            expected = RUN_GROUP_PER_SEED_OUTPUTS if script == "evaluate_per_seed.py" else expected
            expected = () if script == "evaluate.py" and not complete else expected
        missing = [name for name in expected if not (regenerated / name).exists()]
        if missing:
            raise SystemExit(f"{script} did not write {', '.join(missing)} to {regenerated}")


def link_results(dataset, regenerated_dir, target_dir):
    """Complete result folder: regenerated files of the dataset, everything else linked from results/.

    Run-group datasets: the files at the top of the dataset folder describe the reported run group and the pooled
    per-seed statistics; when the reported run group was not regenerated, only the ``run_group_<N>/`` files are taken
    from the regenerated ones.
    """
    if target_dir.exists():
        shutil.rmtree(target_dir)
    reported = config.dataset(dataset).get("reported_run_group")
    if reported is not None and not (regenerated_dir / "aggregated_results.json").exists():
        print(f"  note: the {reported}-run group was not regenerated; its files and the per-seed files are taken from results/")
        kept = regenerated_dir.parent / f"{regenerated_dir.name}_run_groups"
        if kept.exists():
            shutil.rmtree(kept)
        for folder in sorted(regenerated_dir.glob("run_group_*")):
            shutil.copytree(folder, kept / folder.name)
        regenerated_dir = kept
    for source in sorted((PACKAGE_ROOT / "results").rglob("*")):
        relative = source.relative_to(PACKAGE_ROOT / "results")
        if source.is_dir():
            (target_dir / relative).mkdir(parents=True, exist_ok=True)
            continue
        own = regenerated_dir / relative.relative_to(dataset) if relative.parts[0] == dataset else None
        gz_of_own = own is not None and relative.name.endswith(".json.gz") and (own.parent / relative.name[:-3]).exists()
        if gz_of_own:
            continue
        (target_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        (target_dir / relative).symlink_to(own.resolve() if own is not None and own.exists() else source)
    for extra in sorted(regenerated_dir.rglob("*.json")):
        link = target_dir / dataset / extra.relative_to(regenerated_dir)
        if not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(extra.resolve())


def pipeline_mode(options):
    """Run the pipeline stages of the chosen research question(s) for one dataset, then the research-question scripts."""
    dataset, workdir = options.dataset, options.workdir.resolve()
    available = pipeline_questions(dataset)
    if not options.all and options.rq not in available:
        raise SystemExit(f"Research question {options.rq} has no pipeline stages for {dataset}; "
                         f"it is available for: {', '.join(RESEARCH_QUESTIONS[options.rq]['datasets'])}")
    questions = available if options.all else [options.rq]
    if options.windows:
        check_windows(questions, options.windows)
    synthetic = "run_groups" not in config.dataset(dataset)
    families, methods = families_of(questions, dataset, options.configurations)
    draws = any(RESEARCH_QUESTIONS[q]["draws"] for q in questions)
    evaluate = any(RESEARCH_QUESTIONS[q]["evaluation"] for q in questions)
    cost = any(RESEARCH_QUESTIONS[q].get("cost") for q in questions)
    validity = synthetic and any(RESEARCH_QUESTIONS[q].get("validity") for q in questions)
    selection_dir, regenerated = workdir / "selection" / dataset, workdir / "stage_results" / dataset
    hours, kind = estimate(dataset, questions, options.configurations, options.source == "raw")
    evaluation = (["evaluate.py"] + (["evaluate_per_seed.py"] if draws else [])
                  + (["evaluate_distributions.py"] if synthetic else [])) if evaluate else []
    print(f"Plan for research question(s) {', '.join(questions)} on {dataset} ({options.configurations} configurations):")
    print(f"  start from: {'raw trajectories (fetch, group runs, parse, sanitize)' if options.source == 'raw' else 'published vectors in data/' + dataset + '/vectors/ (downloaded by fetch.py --vectors if absent)'}")
    if options.source == "raw" and not options.embed:
        print(f"  embedding is NOT run (it takes {EMBED_HOURS[dataset]} hours for this dataset); the selection stage uses the published vectors. "
              f"Pass --embed to run it.")
    print(f"  selection families: {', '.join(families)}{' + per-seed baseline draws' if draws else ''}"
          + (f"; restricted to {', '.join(methods)}" if methods else ""))
    if evaluation:
        print(f"  evaluation: {', '.join(evaluation + ([] if synthetic or options.run_groups else ['summarize_run_groups.py']))}")
    if cost:
        print(f"  cost: cost_per_trajectory.py (when data/{dataset}/raw/ exists), cost_profile.py, cost_subsets.py")
    if validity:
        print("  distributions: generate_distributions.py, distribution_validity.py")
    print(f"  result files read by the scripts: {', '.join(sorted({f for q in questions for f in RESEARCH_QUESTIONS[q]['reads']}))}")
    print(f"  result files -> {regenerated}; complete result folder -> {workdir / 'results'}; outputs -> {workdir / 'outputs'}")
    print(f"Estimated time: about {hours:.1f} hours ({kind}; {MACHINE})"
          + ("; a restricted run (--windows, --distributions, --run-groups) takes proportionally less." if options.windows or options.distributions or options.run_groups else "."))
    if options.configurations == "reported" and evaluate:
        print("  Statistics computed across methods (Holm-adjusted p-values, dominance ranks, the best configuration of a family) cover the methods "
              "that were run; each ranking block of the regenerated results lists them, and evaluation_manifest.json lists the families.")
    pipeline = PACKAGE_ROOT / "pipeline"
    vectors = [] if options.embed else [[sys.executable, "-B", pipeline / "fetch.py", "--dataset", dataset, "--vectors"]]
    commands = vectors + (raw_commands(options) if options.source == "raw" else []) + selection_commands(options, families, methods, draws, selection_dir)
    commands += [[sys.executable, "-B", pipeline / script, "--dataset", dataset, "--input", selection_dir, "--output", regenerated] for script in evaluation]
    if evaluate and not synthetic and not options.run_groups:   # run_groups.json describes all run groups
        vectors = PACKAGE_ROOT / "data" / dataset / "vectors"
        commands.append([sys.executable, "-B", pipeline / "summarize_run_groups.py", "--dataset", dataset, "--input", vectors,
                         "--repository-map", vectors.parent / "repository_map.json", "--output", regenerated])
    commands += cost_commands(options, selection_dir, regenerated) if cost else []
    commands += validity_commands(options, regenerated) if validity else []
    for command in commands:
        run(command, options.dry_run)
    if not options.dry_run:
        check_stage_outputs(commands, regenerated, synthetic, complete=not (options.windows or options.run_groups))
        link_results(dataset, regenerated, workdir / "results")
    for command in question_commands(questions, workdir / "results", workdir / "outputs"):
        run(command, options.dry_run)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    results = modes.add_parser("results", help="rebuild tables, figures and numbers from results/")
    results.add_argument("--rq", choices=list(RESEARCH_QUESTIONS), default=None, help="one research question (default: all)")
    results.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    pipe = modes.add_parser("pipeline", help="re-run the pipeline for one dataset")
    pipe.add_argument("--dataset", required=True, choices=sorted(config.DATASETS), help="dataset whose result files are regenerated")
    which = pipe.add_mutually_exclusive_group(required=True)
    which.add_argument("--rq", choices=list(RESEARCH_QUESTIONS), help="research question whose inputs are regenerated")
    which.add_argument("--all", action="store_true", help="every research question available for the dataset")
    pipe.add_argument("--from", dest="source", choices=("vectors", "raw"), default="vectors",
                      help="start from the published vectors or from the raw trajectories (default: vectors)")
    pipe.add_argument("--embed", action="store_true", help="with --from raw: also embed the sanitized trajectories (many hours)")
    pipe.add_argument("--configurations", choices=("reported", "all"), default="reported",
                      help="ablation configurations: those reported in the paper, or all (default: reported)")
    pipe.add_argument("--windows", default=None, help="window sizes, e.g. 2 (default: all)")
    pipe.add_argument("--distributions", default=None, help="multi_model, multi_agent: positions of synthetic distributions, e.g. 0-4 (default: all)")
    pipe.add_argument("--run-groups", default=None, help="single_setup: run-group sizes, e.g. 5 (default: 5 to 10)")
    pipe.add_argument("--jobs", default="10", help="worker processes (default: 10)")
    pipe.add_argument("--workdir", type=Path, default=PACKAGE_ROOT / "regenerated", help="folder for regenerated files (default: regenerated/)")
    pipe.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    options = parser.parse_args()
    (results_mode if options.mode == "results" else pipeline_mode)(options)


if __name__ == "__main__":
    main()
