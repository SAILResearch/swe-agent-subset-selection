"""Parser for the SWE-rebench OpenHands trajectories (one JSON file per trajectory, written by ``group_runs.py``).

Each file carries its own metadata (``trajectory_id``, ``repo``, ``resolved``) and the message list under
``trajectory``. Tool calls and observations are located as in ``parsers/openhands.py``; the step fields follow this
dataset's conventions: the observation and its error message are set for every tool, editor steps are labelled
``str_replace_editor: <path>``, and test scripts are recognised by the patterns ``*test*.py``, ``*reproduce*.py`` and
``*verify*.py``.
"""
import json
import re

from parsers.openhands import (TEST_PATTERNS, command_name, edit_size, extract_error_message, find_observation, message_text,
                               task_description, xml_tool_calls)

TEST_SCRIPT_PATTERN = re.compile(r"\b(\w*test\w*\.py|\w*reproduce\w*\.py|\w*verify\w*\.py)\b")


def parse_shell_call(command_line, observation):
    """Step fields of a shell tool call, including test-run detection."""
    lowered, observation_lower = command_line.lower(), observation.lower()
    test_name = None
    if any(pattern in lowered for pattern in TEST_PATTERNS):
        script = re.search(r"\b(test_\w+\.py)\b", lowered)
        test_name = script.group(1) if script else "pytest"
    else:
        script = TEST_SCRIPT_PATTERN.search(lowered)
        if script and "python" in lowered:
            test_name = script.group(1)
    tests_run, tests_passed = [], []
    if test_name is not None:
        tests_run.append(test_name)
        looks_passed = "passed" in observation_lower or "✅" in observation or "success" in observation_lower
        if looks_passed and "failed" not in observation_lower:
            tests_passed.append("some")
    return {"action": f"execute_bash: {command_line}", "action_command": command_name(command_line),
            "tests_run": tests_run, "tests_passed": tests_passed}


def parse_tool_call(tool_call, observation):
    """One step of the unified schema from a tool call and its observation."""
    name, arguments_text = tool_call["function"]["name"], tool_call["function"]["arguments"]
    step = {"action": f"{name}: {arguments_text}", "action_header": name, "action_command": name, "files_touched": [], "tests_run": [],
            "tests_passed": [], "edit_size": 0, "observation": observation, "error_message": extract_error_message(observation)}
    try:
        arguments = json.loads(arguments_text)
        if name == "execute_bash":
            step.update(parse_shell_call(arguments.get("command", ""), observation))
        elif name == "str_replace_editor":
            path, command = arguments.get("path"), arguments.get("command", "view")
            step["action_command"] = command
            step["action"] = f"str_replace_editor: {path}"
            if path:
                step["files_touched"].append(path)
            if command == "create":
                file_text = arguments.get("file_text", "")
                if file_text:
                    step["edit_size"] = len(file_text.split("\n"))
            elif command == "str_replace":
                step["edit_size"] = edit_size(arguments.get("old_str", ""), arguments.get("new_str", ""))
        elif name == "think":
            step["action"] = f"think: {arguments.get('thought', '')}"
            step["action_command"] = "think"
    except Exception:   # arguments that are not a JSON object: the step keeps the tool name and the raw arguments
        pass
    return step


def parse_trajectory(record, agent_name):
    """Unified-schema record of one trajectory file."""
    messages = record.get("trajectory", [])
    steps, position = [], 0
    while position < len(messages):
        message = messages[position]
        tool_calls = []
        if message.get("role") == "assistant":
            text = message_text(message)
            if message.get("tool_calls"):
                tool_calls = message["tool_calls"]
            elif "<function=" in text:
                tool_calls = xml_tool_calls(text)
        for tool_call in tool_calls:
            observation, position = find_observation(messages, position, tool_call.get("id"))
            step = parse_tool_call(tool_call, observation)
            step["step_index"] = len(steps) + 1
            steps.append(step)
        position += 1
    return {"trajectory_id": record.get("trajectory_id", "unknown_id"), "repo_name": record.get("repo", "unknown_repo"),
            "agent_name": agent_name, "task_description": task_description(messages),
            "final_outcome": bool(record.get("resolved", 0)), "steps": steps}


def parse_run(run_dir, output_dir):
    """Parse every ``*.json`` of a run folder (``<N>_runs/run_<i>``) into ``output_dir``; return one log line per trajectory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_name = f"{run_dir.parent.name}/{run_dir.name}"
    log = []
    for path in sorted(run_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            record = parse_trajectory(json.load(handle), agent_name)
        with open(output_dir / f"unified_{path.name}", "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        log.append(f"{path.name}: PARSED ({'SUCCESS' if record['final_outcome'] else 'FAILURE'}) - {len(record['steps'])} steps")
    return log
