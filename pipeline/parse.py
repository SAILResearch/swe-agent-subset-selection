"""Stage 1: raw trajectories -> unified schema (paper Section 4.1).

    python pipeline/parse.py --dataset multi_model [--input RAW_DIR] [--output PARSED_DIR] [--validate]

``RAW_DIR`` holds one folder per run (defaults to data/<dataset>/raw); the parsed files are written to
``PARSED_DIR/<run>/unified_<instance>.json`` with a ``_parser_log.txt`` per run. The parser is chosen by
the dataset definition in config.py.

``--validate`` runs the parsing validation of Section 4.1 after parsing: for every parsed trajectory,
each step's tool name, action payload and observation must appear verbatim in the raw trajectory, and the
recorded error must agree with the exit code reported in the observation. What "verbatim" can mean depends on
the format: JSON message formats are compared with the decoded strings of the raw file (tool arguments included);
Moatless actions are stored as the text form of the raw action object and are compared with it; Lingxi logs are
plain text, so fields are searched in the text, and action payloads, which the parser cuts to 100 characters,
are compared without the trailing "..." and also with the escape sequences of quoted arguments decoded. A summary with one entry per run is written to
``PARSED_DIR/parsing_validation.json``.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import parsers  # noqa: E402

DERIVED_TOOL_NAMES = ("think", "unknown", "Conclusion", "bash")   # set by a parser when a step has no tool call of that name
DERIVED_PAYLOADS = ("None", "N/A", "think", "finish", "project", "[See final patch / summary]")   # placeholders written by the parsers
EXIT_CODE_PATTERN = re.compile(r"(?:exit code|finished with exit code)\s*(\d+)")


def parse_dataset(dataset_name, input_dir, output_dir, run_groups=None):
    """Parse every run of a dataset; return the number of trajectories written."""
    definition = config.dataset(dataset_name)
    total = 0
    for run in config.runs_of(dataset_name, run_groups):
        parser = parsers.load(definition.get("run_parsers", {}).get(run, definition["parser"]))
        run_dir = input_dir / run
        if not run_dir.is_dir():
            raise SystemExit(f"Run folder not found: {run_dir}")
        log = parser.parse_run(run_dir, output_dir / run)
        (output_dir / run / "_parser_log.txt").write_text(f"--- Parsing Log for {run} ---\n" + "\n".join(log), encoding="utf-8")
        print(f"  {run}: read {len(log)} raw trajectories from {run_dir}, wrote {len(log)} files to {output_dir / run}")
        total += len(log)
    names = input_dir / "repository_names.json"   # written by group_runs.py; passed on to the sanitize stage
    if names.exists():
        (output_dir / names.name).write_text(names.read_text())
    return total


def raw_strings(node, found):
    """Collect every string of a raw trajectory, including the values inside JSON-encoded tool arguments."""
    if isinstance(node, str):
        found.add(node)
        found.add(node.strip())
        try:
            decoded = json.loads(node)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(decoded, (dict, list)):
            raw_strings(decoded, found)
    elif isinstance(node, dict):
        if "action_args_class" in node:
            found.add(str(node))   # Moatless actions are recorded as the text form of the action object
        for value in node.values():
            raw_strings(value, found)
    elif isinstance(node, list):
        for value in node:
            raw_strings(value, found)


ESCAPE_SEQUENCE = re.compile(r"\\(.)", re.DOTALL)   # decoded in one pass, so that an escaped backslash is not read twice
ESCAPED_CHARACTERS = {"n": "\n", "t": "\t", "r": "\r", "'": "'", '"': '"', "\\": "\\", "\n": ""}
_UNESCAPED = {}


def unescaped(raw_text):
    """Plain-text log with the escape sequences of quoted tool arguments decoded (cached for the current file)."""
    if _UNESCAPED.get("source") is not raw_text:
        text = ESCAPE_SEQUENCE.sub(lambda match: ESCAPED_CHARACTERS.get(match.group(1), match.group(0)), raw_text)
        _UNESCAPED.update(source=raw_text, text=text)
    return _UNESCAPED["text"]


def step_problems(step, strings, raw_text):
    """Names of the checks a parsed step does not pass."""
    problems = []
    payload = step["action"].split(": ", 1)[1] if ": " in step["action"] else step["action"]
    if not strings and payload.endswith("..."):
        payload = payload[:-3]
    if step["action_header"] not in raw_text and step["action_header"] not in DERIVED_TOOL_NAMES:
        problems.append("tool name")
    if payload and payload not in DERIVED_PAYLOADS and payload not in strings and not any(payload in s for s in strings) \
            and not (not strings and (payload in raw_text or payload in unescaped(raw_text))):
        problems.append("action payload")
    if step["observation"] and step["observation"] not in strings and not (not strings and step["observation"] in raw_text):
        problems.append("observation")
    exit_code = EXIT_CODE_PATTERN.search(step["observation"])
    if exit_code and int(exit_code.group(1)) == 0 and step.get("error_message"):
        problems.append("exit code 0 with an error recorded")
    if exit_code and int(exit_code.group(1)) != 0 and not step.get("error_message"):
        problems.append("non-zero exit code without an error recorded")
    return problems


def raw_file(trajectory_dir, instance):
    """Raw trajectory of an instance in a run folder: trajs/<instance>.json, trajs/<instance>.traj,
    trajs/<instance>/trajectory.json, or <instance>.json (run groups)."""
    candidates = (trajectory_dir / "trajs" / f"{instance}.json", trajectory_dir / "trajs" / f"{instance}.traj",
                  trajectory_dir / "trajs" / instance / "trajectory.json", trajectory_dir / f"{instance}.json")
    found = next((path for path in candidates if path.exists()), None)
    if found is None:
        raise SystemExit(f"parse: no raw trajectory of {instance} in {trajectory_dir}")
    return found


def validate_dataset(dataset_name, input_dir, output_dir, run_groups=None):
    """Check every parsed trajectory against its raw trajectory; write and return the summary."""
    summary = {"dataset": dataset_name, "trajectories": 0, "steps": 0, "trajectories_with_problems": 0,
               "steps_with_problems": 0, "problems_by_check": {}, "runs": {}, "examples": []}
    for run in config.runs_of(dataset_name, run_groups):
        per_run = summary["runs"][run] = {"trajectories": 0, "steps": 0, "steps_with_problems": 0, "problems_by_check": {}}
        for parsed_path in sorted((output_dir / run).glob("unified_*.json")):
            raw_path = raw_file(input_dir / run, parsed_path.stem[len("unified_"):])
            raw_text = raw_path.read_text(encoding="utf-8")
            strings = set()
            if raw_path.suffix == ".json":
                raw_strings(json.loads(raw_text), strings)
            record = json.loads(parsed_path.read_text(encoding="utf-8"))
            bad_steps = 0
            for step in record["steps"]:
                problems = step_problems(step, strings, raw_text)
                for problem in problems:
                    summary["problems_by_check"][problem] = summary["problems_by_check"].get(problem, 0) + 1
                    per_run["problems_by_check"][problem] = per_run["problems_by_check"].get(problem, 0) + 1
                if problems:
                    bad_steps += 1
                    if len(summary["examples"]) < 20:
                        summary["examples"].append({"run": run, "trajectory": record["trajectory_id"],
                                                    "step": step["step_index"], "problems": problems})
            summary["trajectories"] += 1
            summary["steps"] += len(record["steps"])
            summary["steps_with_problems"] += bad_steps
            per_run["trajectories"] += 1
            per_run["steps"] += len(record["steps"])
            per_run["steps_with_problems"] += bad_steps
            summary["trajectories_with_problems"] += bool(bad_steps)
    (output_dir / "parsing_validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(config.DATASETS))
    parser.add_argument("--input", type=Path, default=None, help="raw folder (default: data/<dataset>/raw; for datasets with run groups: data/<dataset>/grouped)")
    parser.add_argument("--output", type=Path, default=None, help="parsed folder (default: data/<dataset>/parsed)")
    parser.add_argument("--run-groups", default=None, help="run-group sizes to parse, e.g. 10 (datasets with run groups only)")
    parser.add_argument("--validate", action="store_true", help="check the parsed steps against the raw trajectories")
    arguments = parser.parse_args()
    input_dir = arguments.input or config.stage_dir("grouped" if "run_groups" in config.dataset(arguments.dataset) else "raw", arguments.dataset)
    output_dir = arguments.output or config.stage_dir("parsed", arguments.dataset)
    started = time.time()
    run_groups = [int(g) for g in arguments.run_groups.split(",")] if arguments.run_groups else None
    total = parse_dataset(arguments.dataset, input_dir, output_dir, run_groups)
    print(f"parse: {total} trajectories, {time.time() - started:.1f} s")
    if arguments.validate:
        summary = validate_dataset(arguments.dataset, input_dir, output_dir, run_groups)
        print(f"validation: {summary['trajectories']} trajectories, {summary['steps']} steps, "
              f"{summary['steps_with_problems']} steps with problems in {summary['trajectories_with_problems']} trajectories; "
              f"by check: {summary['problems_by_check']} -> {output_dir / 'parsing_validation.json'}")


if __name__ == "__main__":
    main()
