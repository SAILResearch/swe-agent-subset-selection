"""Parser for Lingxi trajectories (SWE-bench experiments repository).

``trajs/<instance>.traj`` is a text log. The steps are the ``## Step <n>`` sections of the ``<problem_solver 0>`` block;
a section holds an observation or a thinking part and an action (or the conclusion). The result of a tool call is
the observation of the following section. The outcome comes from
``results/results.json``; one ``unified_<instance>.json`` per trajectory is written to the output folder.
"""
import ast
import json
import os
import re

from parsers.openhands import edit_size, extract_error_message, load_resolved_instances, parse_shell_call

AGENT_NAME = "Lingxi-v1.5"
SECTION_END = r"(?=\n##|\n###|\Z)"
OBSERVATION = re.compile(r"### 👁️ Observation\n(.*?)" + SECTION_END, re.S)
THINKING = re.compile(r"### 🤔 Thinking\n(.*?)" + SECTION_END, re.S)
OBSERVATION_OR_THINKING = re.compile(r"### (?:👁️ Observation|🤔 Thinking)\n(.*?)" + SECTION_END, re.S)
ACTION = re.compile(r"### (?:🎯 Action|📋 Conclusion)\n(.*?)" + SECTION_END, re.S)
CONCLUSION_STARTS = ("<observation>", "diff --git", "## Summary", "The final patch")
DISPLAY_LENGTH = 100


def parse_arguments(text):
    """Arguments of a tool call written as a Python dict literal or as ``key='value', ...`` pairs."""
    if text.strip().startswith("{") and text.strip().endswith("}"):
        try:
            arguments = ast.literal_eval(text)
            if isinstance(arguments, dict):
                return arguments
        except Exception:   # not a literal: fall back to the key=value scan
            pass
    arguments, position = {}, 0
    while position < len(text):
        match = re.match(r"(\w+)\s*=\s*(['\"])", text[position:])
        if not match:
            position += 1
            continue
        key, quote = match.group(1), match.group(2)
        start = end = position + match.end()
        escaped = False
        while end < len(text):
            character = text[end]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                arguments[key] = text[start:end]
                position = end + 1
                break
            end += 1
        else:
            arguments[key] = text[start:]
            break
    return arguments or {"raw_args": text}


def shorten(text):
    """Text cut to ``DISPLAY_LENGTH`` characters (ellipsis included) for the action field."""
    return text if len(text) < DISPLAY_LENGTH else text[:DISPLAY_LENGTH - 3] + "..."


def following(sections, index, pattern):
    """Stripped first match of ``pattern`` in the section after ``index`` ("" when there is none)."""
    if index < len(sections) - 1:
        match = pattern.search(sections[index + 1])
        if match:
            return match.group(1).strip()
    return ""


def parse_section(sections, index):
    """One step of the unified schema from section ``index`` (1-based step number)."""
    section = sections[index]
    observation = OBSERVATION.search(section)
    thinking = THINKING.search(section)
    own_text = (observation.group(1).strip() if observation else "") or (thinking.group(1).strip() if thinking else "")
    step = {"step_index": index, "action": "", "action_header": "", "action_command": "", "files_touched": [], "tests_run": [],
            "tests_passed": [], "edit_size": 0, "observation": own_text, "error_message": ""}
    action = ACTION.search(section)
    if not action:
        step.update(action="think", action_header="think", action_command="think", error_message=extract_error_message(own_text))
        return step
    action_text = action.group(1).strip()
    call = re.match(r"(\w+)\((.*)\)", action_text, re.S)
    name, arguments_text, arguments = "unknown", action_text, {}
    if call:
        name, arguments_text = call.group(1), call.group(2).strip()
        arguments = parse_arguments(arguments_text)
    elif action_text.startswith(CONCLUSION_STARTS):
        name, arguments_text = "Conclusion", "[See final patch / summary]"
        step["observation"] = action_text
    step.update(action=f"{name}: {shorten(arguments_text)}", action_header=name, action_command=name)
    try:
        if name in ("bash", "execute_bash"):
            step["action_header"] = "bash"
            result = following(sections, index, OBSERVATION_OR_THINKING)
            step.update(parse_shell_call(arguments.get("command", arguments.get("raw_args", "")), result))
            step["observation"] = result
            step["error_message"] = extract_error_message(result)
        elif name in ("view_file_content", "view_directory", "search_files_by_keywords", "str_replace_based_edit_tool"):
            if name == "str_replace_based_edit_tool":
                path, command = arguments.get("path", arguments.get("raw_args")), arguments.get("command", "view")
                step["action_command"] = command
            else:
                path = arguments.get("file_name", arguments.get("dir_path", arguments.get("directory", arguments.get("raw_args"))))
                command = None
            if path and path != arguments.get("raw_args"):
                step["files_touched"].append(path)
            if command == "create":
                file_text = arguments.get("file_text", "")
                if isinstance(file_text, str) and file_text:
                    step["edit_size"] = len(file_text.split("\n"))
            elif command == "str_replace":
                text_before, text_after = arguments.get("old_str", ""), arguments.get("new_str", "")
                if isinstance(text_before, str) and isinstance(text_after, str):
                    step["edit_size"] = edit_size(text_before, text_after)
            result = following(sections, index, OBSERVATION)
            if result:
                step["observation"] = result
            step["error_message"] = extract_error_message(result or own_text)
        elif name == "think":
            step["action"] = f"think: {shorten(arguments.get('thought', arguments.get('raw_args', arguments_text)))}"
            step["error_message"] = extract_error_message(own_text)
        elif name == "ask_repository_agent":
            step["action"] = f"ask_repository_agent: {shorten(arguments.get('query', arguments.get('raw_args', arguments_text)))}"
            step["observation"] = following(sections, index, OBSERVATION_OR_THINKING)
        elif name == "Conclusion":
            step["action"] = "Conclusion: [See final patch / summary]"
        else:
            step["error_message"] = extract_error_message(own_text)
    except Exception:   # malformed arguments: keep the step with the tool name only
        step["action_command"] = name
        step["error_message"] = extract_error_message(own_text)
    return step


def parse_trajectory(text, file_name, resolved):
    """Unified-schema record of one Lingxi trajectory log."""
    trajectory_id = os.path.splitext(file_name)[0]
    description = re.search(r"<issue_description>(.*?)</issue_description>", text, re.S)
    solver = re.search(r"<problem_solver 0>(.*?)</problem_solver 0>", text, re.S)
    sections = re.split(r"\n## Step \d+\n", solver.group(1).strip()) if solver else []
    steps = [parse_section(sections, index) for index in range(1, len(sections))]
    return {"trajectory_id": trajectory_id, "repo_name": trajectory_id.split("__")[0] if "__" in trajectory_id else trajectory_id,
            "agent_name": AGENT_NAME, "task_description": description.group(1).strip() if description else "",
            "final_outcome": trajectory_id in resolved, "steps": steps}


def parse_run(run_dir, output_dir):
    """Parse every ``trajs/*.traj`` of a run into ``output_dir``; return one log line per trajectory."""
    trajectory_dir = run_dir / "trajs"
    if not trajectory_dir.is_dir():
        raise SystemExit(f"No 'trajs' folder in {run_dir}")
    resolved = load_resolved_instances(run_dir / "results" / "results.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for path in sorted(trajectory_dir.glob("*.traj")):
        record = parse_trajectory(path.read_text(encoding="utf-8"), path.name, resolved)
        with open(output_dir / f"unified_{path.stem}.json", "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        log.append(f"{path.name}: PARSED ({'SUCCESS' if record['final_outcome'] else 'FAILURE'})")
    return log
