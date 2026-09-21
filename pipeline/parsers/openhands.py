"""Parser for OpenHands trajectories as published in the SWE-bench experiments repository.

A run folder holds ``trajs/<instance>.json`` (the message list of one trajectory) and
``results/results.json`` (the resolved instances). Each assistant tool call, either a structured
``tool_calls`` entry or an XML-style ``<function=...>`` block in the message text, becomes one step
of the unified schema together with the observation that answers it.

All rules are deterministic: shell commands are normalised by rule, test runs are detected by
pattern matching, and errors are extracted from exit codes and Python tracebacks.
"""
import json
import os
import re

ERROR_KEYWORDS = (
    "error", "exception", "traceback", "failed", "fatal", "syntaxerror",
    "no such file or directory", "command not found", "permission denied",
)
TEST_PATTERNS = (
    "pytest", "py.test", "python -m pytest", "python3 -m pytest",
    "python -m unittest", "python3 -m unittest", "unittest discover",
)
XML_TOOL_CALL_ID = "xml_parsed"
OBSERVATION_LOOKAHEAD = 3  # messages inspected after a tool call for its observation
FUNCTION_PATTERN = re.compile(r"<function=(.*?)>(.*?)</function>", re.DOTALL)
PARAMETER_PATTERN = re.compile(r"<parameter=(.*?)>(.*?)</parameter>", re.DOTALL)
TEST_SCRIPT_PATTERN = re.compile(r"\b(\w*test\w*\.py|\w*reproduce\w*\.py|\w*verify\w*\.py|\w*verification\w*\.py)\b")


def message_text(message):
    """Text of a message whose content is either a string or a list of {"type": "text", "text": ...} items."""
    content = message.get("content")
    if not content:
        return ""
    if isinstance(content, list):
        first = content[0]
        return first.get("text", "") if isinstance(first, dict) else str(content)
    return content if isinstance(content, str) else str(content)


def load_resolved_instances(results_file):
    """Set of resolved instance ids of a run (empty when the results file is missing or unreadable)."""
    try:
        with open(results_file, "r", encoding="utf-8") as handle:
            return set(json.load(handle).get("resolved", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def command_name(command_line):
    """Normalised program name of a shell command line (leading ``cd ... &&`` and VAR=value parts skipped)."""
    if command_line.startswith("cd "):
        parts = command_line.split(" && ", 1)
        command_line = parts[1] if len(parts) > 1 else parts[0]
    command_line = command_line.strip()
    if not command_line:
        return "empty_command"
    parts = command_line.split()
    name = next((part for part in parts if "=" not in part), "")
    if not name and parts:
        name = parts[0]
    if name.startswith("./"):
        name = name[2:]
    return name if name else "empty_command"


def edit_size(text_before, text_after):
    """Number of changed lines between two strings: differing lines plus the difference in length."""
    lines_before, lines_after = text_before.split("\n"), text_after.split("\n")
    changed = abs(len(lines_before) - len(lines_after))
    return changed + sum(1 for before, after in zip(lines_before, lines_after) if before != after)


def extract_error_message(observation):
    """Error message of an observation, based on its exit code and Python traceback.

    Exit code 0 means no error. Otherwise a traceback is returned when present; with a non-zero exit
    code and no traceback, the short lines containing an error keyword are returned.
    """
    if not observation:
        return ""
    match = re.search(r"(?:exit code|finished with exit code)\s*(\d+)", observation)
    exit_code = int(match.group(1)) if match else None
    if exit_code == 0:
        return ""
    traceback = re.search(r"Traceback \(most recent call last\):.*", observation, re.DOTALL)
    if traceback:
        return traceback.group(0).strip()
    if exit_code is None:
        return ""
    found = []
    for line in observation.split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) < 300 and any(k in stripped.lower() for k in ERROR_KEYWORDS) and stripped not in found:
            found.append(stripped)
    return "\n".join(found) if found else f"Command failed with exit code {exit_code}"


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
    """One unified-schema step from a tool call and its observation."""
    name = tool_call["function"]["name"]
    raw_arguments = tool_call["function"]["arguments"]
    step = {"action": f"{name}: {raw_arguments}", "action_header": name, "action_command": name,
            "files_touched": [], "tests_run": [], "tests_passed": [], "edit_size": 0}
    try:
        arguments = json.loads(raw_arguments)
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
        step["error_message"] = extract_error_message(observation)
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        step["action"] = f"{name}: {raw_arguments}"
        if observation:
            step["error_message"] = extract_error_message(observation)
    step["observation"] = observation
    return step


def xml_tool_calls(text):
    """Tool calls written as ``<function=NAME><parameter=KEY>VALUE</parameter></function>`` blocks."""
    calls = []
    for name, body in FUNCTION_PATTERN.findall(text):
        parameters = {key.strip(): value.strip() for key, value in PARAMETER_PATTERN.findall(body)}
        calls.append({"id": XML_TOOL_CALL_ID, "type": "function",
                      "function": {"name": name.strip(), "arguments": json.dumps(parameters)}})
    return calls


def task_description(messages):
    """Issue text of the first user message (the content of its <issue_description> block when present)."""
    for message in messages:
        if message.get("role") == "user":
            text = message_text(message)
            if "<issue_description>" in text:
                start = text.find("<issue_description>") + len("<issue_description>")
                end = text.find("</issue_description>")
                if end != -1:
                    return text[start:end].strip()
            return text
    return ""


def find_observation(messages, position, tool_call_id):
    """Observation answering a tool call and the index of its message (``position`` if none is found)."""
    for index in range(position + 1, min(position + 1 + OBSERVATION_LOOKAHEAD, len(messages))):
        candidate = messages[index]
        text = message_text(candidate)
        if candidate.get("role") == "tool" and candidate.get("tool_call_id") == tool_call_id:
            return text, index
        if tool_call_id == XML_TOOL_CALL_ID and candidate.get("role") == "user" and "EXECUTION RESULT" in text:
            return text, index
    return "", position


def parse_trajectory(messages, file_name, resolved, agent_name):
    """Unified-schema record of one raw trajectory."""
    trajectory_id = os.path.splitext(file_name)[0]
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
    return {"trajectory_id": trajectory_id,
            "repo_name": trajectory_id.split("__")[0] if "__" in trajectory_id else trajectory_id,
            "agent_name": agent_name, "task_description": task_description(messages),
            "final_outcome": trajectory_id in resolved, "steps": steps}


def parse_run(run_dir, output_dir):
    """Parse every ``trajs/*.json`` of a run into ``output_dir``; return one log line per trajectory."""
    trajectory_dir = run_dir / "trajs"
    if not trajectory_dir.is_dir():
        raise SystemExit(f"No 'trajs' folder in {run_dir}")
    resolved = load_resolved_instances(run_dir / "results" / "results.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for path in sorted(trajectory_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            messages = json.load(handle)
        record = parse_trajectory(messages, path.name, resolved, run_dir.name)
        with open(output_dir / f"unified_{path.name}", "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        outcome = "SUCCESS" if record["final_outcome"] else "FAILURE"
        log.append(f"{path.name}: PARSED ({outcome}) - {len(record['steps'])} steps")
    return log
