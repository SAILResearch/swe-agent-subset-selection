"""Parser for Trae Agent trajectories (SWE-bench experiments repository).

``trajs/<instance>.json`` is a list of chat messages; every assistant ``tool_calls`` entry becomes one step, and its
observation is the ``tool`` message with the same ``tool_call_id``. The outcome comes from
``results/results.json``; one ``unified_<instance>.json`` per trajectory is written to the output folder.
"""
import json
import os
import re

from parsers.openhands import edit_size, extract_error_message, load_resolved_instances, message_text, parse_shell_call

AGENT_NAME = "TRAE"


def task_description(messages):
    """Problem statement of the first user message (text after "[Problem statement]:", code fences removed)."""
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            match = re.search(r"\[Problem statement\]:\s*(.*)", content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip().replace("```python", "").replace("```julia", "").replace("```", "").strip()
            return content.strip()
    return ""


def parse_tool_call(tool_call, observation):
    """One step of the unified schema from a tool call and its observation."""
    name, arguments_text = tool_call["function"]["name"], tool_call["function"]["arguments"]
    step = {"action": f"{name}: {arguments_text}", "action_header": name, "action_command": name, "files_touched": [],
            "tests_run": [], "tests_passed": [], "edit_size": 0, "error_message": ""}
    try:
        arguments = json.loads(arguments_text)
        command = arguments.get("command", "")
        if name == "bash":
            step.update(parse_shell_call(command, observation))
            step["action"] = f"bash: {command}"
        elif name == "str_replace_editor":
            path = arguments.get("path")
            if path:
                step["files_touched"].append(path)
            step["action_command"] = command
            step["action"] = f"{command}: {path or 'N/A'}"
            if command == "create":
                file_text = arguments.get("file_text", "")
                step["edit_size"] = len(file_text.splitlines()) if file_text else 0
            elif command == "str_replace":
                step["edit_size"] = edit_size(arguments.get("old_str", ""), arguments.get("new_str", ""))
        elif name == "ckg_tool":
            step["action_command"] = command
            detail = arguments.get("function_name") or arguments.get("class_name") or arguments.get("file_name") or "project"
            step["action"] = f"{command}: {detail}"
        elif name == "sequential_thinking":
            step["action_command"] = "think"
            step["action"] = f"think: {arguments.get('thought', '')}"
        elif name == "task_done":
            step["action_command"] = "finish"
            step["action"] = "finish"
        else:
            step["action_command"] = name
            step["action"] = f"{name}: {command or arguments_text}"
        step["error_message"] = extract_error_message(observation)
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        step["action"] = f"{name}: {arguments_text}"
        step["action_command"] = name
        step["error_message"] = extract_error_message(observation)
    step["observation"] = observation
    return step


def parse_trajectory(messages, file_name, resolved, agent_name=AGENT_NAME, parse_call=parse_tool_call, describe=task_description):
    """Unified-schema record of one trajectory given as a list of chat messages."""
    trajectory_id = os.path.splitext(file_name)[0]
    observations = {m["tool_call_id"]: message_text(m) for m in messages if m.get("role") == "tool" and "tool_call_id" in m}
    steps = []
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                step = parse_call(tool_call, observations.get(tool_call.get("id"), ""))
                step["step_index"] = len(steps) + 1
                steps.append(step)
    return {"trajectory_id": trajectory_id, "repo_name": trajectory_id.split("__")[0] if "__" in trajectory_id else trajectory_id,
            "agent_name": agent_name, "task_description": describe(messages), "final_outcome": trajectory_id in resolved, "steps": steps}


def parse_message_files(run_dir, output_dir, parse_one):
    """Parse every ``trajs/*.json`` of a run with ``parse_one(messages, file name, resolved)``; return the log lines."""
    trajectory_dir = run_dir / "trajs"
    if not trajectory_dir.is_dir():
        raise SystemExit(f"No 'trajs' folder in {run_dir}")
    resolved = load_resolved_instances(run_dir / "results" / "results.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for path in sorted(trajectory_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            record = parse_one(json.load(handle), path.name, resolved)
        with open(output_dir / f"unified_{path.name}", "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        log.append(f"{path.name}: PARSED ({'SUCCESS' if record['final_outcome'] else 'FAILURE'})")
    return log


def parse_run(run_dir, output_dir):
    """Parse every trajectory of a run into ``output_dir``; return one log line per trajectory."""
    return parse_message_files(run_dir, output_dir, parse_trajectory)
