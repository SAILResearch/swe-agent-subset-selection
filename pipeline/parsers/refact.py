"""Parser for Refact Agent trajectories (SWE-bench experiments repository).

Same message layout as Trae (``trajs/<instance>.json``, assistant ``tool_calls`` answered by ``tool`` messages); the
tools differ. The outcome comes from
``results/results.json``; one ``unified_<instance>.json`` per trajectory is written to the output folder.
"""
import json

from parsers import trae
from parsers.openhands import edit_size, extract_error_message, parse_shell_call


def task_description(messages):
    """Text of the first user message, without surrounding back-ticks."""
    for message in messages:
        if message.get("role") == "user":
            return message.get("content", "").strip().strip("`").strip()
    return ""


def parse_tool_call(tool_call, observation):
    """One step of the unified schema from a tool call and its observation."""
    name, arguments_text = tool_call["function"]["name"], tool_call["function"]["arguments"]
    step = {"action": f"{name}: {arguments_text}", "action_header": name, "action_command": name, "files_touched": [],
            "tests_run": [], "tests_passed": [], "edit_size": 0}
    try:
        arguments = json.loads(arguments_text)
        if name in ("execute_bash", "shell"):
            step.update(parse_shell_call(arguments.get("command", ""), observation))
        elif name in ("create_textdoc", "update_textdoc"):
            path = arguments.get("path")
            if path:
                step["files_touched"].append(path)
            step["action_command"] = name
            if name == "create_textdoc":
                step["edit_size"] = len(arguments.get("content", "").split("\n"))
            else:
                step["edit_size"] = edit_size(arguments.get("old_str", ""), arguments.get("replacement", ""))
        elif "file" in name or "editor" in name or "replace" in name or "cat" in name:
            path = arguments.get("path", arguments.get("paths"))
            if isinstance(path, list):
                path_text = ", ".join(path)
                step["files_touched"].extend(path)
            else:
                path_text = path
                if path:
                    step["files_touched"].append(path)
            step["action"] = f"{name}: {path_text or arguments_text}"
        else:
            step["action_command"] = name
        step["error_message"] = extract_error_message(observation)
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        step["action"] = f"{name}: {arguments_text}"
        if observation:
            step["error_message"] = extract_error_message(observation)
    step["observation"] = observation
    return step


def parse_run(run_dir, output_dir):
    """Parse every trajectory of a run into ``output_dir``; return one log line per trajectory."""
    def parse_one(messages, file_name, resolved):
        return trae.parse_trajectory(messages, file_name, resolved, agent_name=run_dir.name,
                                     parse_call=parse_tool_call, describe=task_description)
    return trae.parse_message_files(run_dir, output_dir, parse_one)
