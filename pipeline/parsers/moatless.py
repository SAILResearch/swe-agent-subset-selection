"""Parser for Moatless trajectories (SWE-bench experiments repository).

``trajs/<instance>/trajectory.json`` holds a list of nodes; every entry of a node's ``action_steps`` (an action and its
observation) becomes one step. The outcome is the trajectory's own ``evaluation_result`` when present, otherwise the
run's list of resolved instances (``results/results.json``). One ``unified_<instance>.json`` per trajectory is written
to the output folder.
"""
import json
import os

from parsers.openhands import extract_error_message, load_resolved_instances

AGENT_NAME = "MoatlessClaude4Sonnet"
FILE_ACTIONS = ("ReadFile", "StringReplace", "CreatePythonFile", "CreateFile")


def parse_action_step(action_step):
    """One step of the unified schema from a Moatless action step."""
    action, observation = action_step.get("action", {}), action_step.get("observation", {})
    action_class = action.get("action_args_class", "")
    action_type = action_class.split(".")[-1].replace("Args", "") if action_class else "unknown"
    step = {"action": f"{action_type}: {action}", "action_header": action_type, "action_command": action_type,
            "files_touched": [], "tests_run": [], "tests_passed": [], "edit_size": 0, "observation": observation.get("message", "")}
    if action_type in FILE_ACTIONS:
        file_path = action.get("file_path") or action.get("path")
        if file_path:
            step["files_touched"].append(file_path)
    if action_type == "StringReplace":
        text_before, text_after = action.get("old_str", ""), action.get("new_str", "")
        if text_before or text_after:
            step["edit_size"] = abs(len(text_before.split("\n")) - len(text_after.split("\n")))
    if action_type == "RunTests":
        step["tests_run"] = action.get("test_files", [])
        if "passed" in observation.get("message", "").lower():
            step["tests_passed"] = ["some"]
    step["error_message"] = extract_error_message(step["observation"])
    return step


def parse_trajectory(trajectory, path_text, resolved):
    """Unified-schema record of one Moatless trajectory."""
    nodes = trajectory.get("nodes", [])
    trajectory_id = trajectory.get("trajectory_id", "")
    if not trajectory_id and path_text:
        trajectory_id = os.path.basename(os.path.dirname(path_text)) if "/" in path_text else path_text
    evaluation = trajectory.get("evaluation_result", {})
    steps = []
    for node in nodes:
        for action_step in node.get("action_steps", []):
            step = parse_action_step(action_step)
            step["step_index"] = len(steps) + 1
            step["node_id"] = node.get("node_id")
            steps.append(step)
    return {"trajectory_id": trajectory_id, "repo_name": trajectory_id.split("__")[0] if "__" in trajectory_id else trajectory_id,
            "agent_name": AGENT_NAME, "task_description": nodes[0].get("user_message", "").strip() if nodes else "",
            "final_outcome": evaluation.get("resolved", False) if evaluation else (trajectory_id in resolved), "steps": steps}


def parse_run(run_dir, output_dir):
    """Parse every ``trajs/<instance>/trajectory.json`` of a run into ``output_dir``; return one log line per trajectory."""
    trajectory_dir = run_dir / "trajs"
    if not trajectory_dir.is_dir():
        raise SystemExit(f"No 'trajs' folder in {run_dir}")
    resolved = load_resolved_instances(run_dir / "results" / "results.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for path in sorted(trajectory_dir.glob("*/trajectory.json")):
        with open(path, "r", encoding="utf-8") as handle:
            record = parse_trajectory(json.load(handle), str(path), resolved)
        with open(output_dir / f"unified_{path.parent.name}.json", "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        log.append(f"{path.parent.name}: PARSED ({'SUCCESS' if record['final_outcome'] else 'FAILURE'})")
    return log
