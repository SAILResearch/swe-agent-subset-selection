"""Cost stage 1: token cost of every raw trajectory (paper Section 6.4).

    python pipeline/cost_per_trajectory.py --dataset multi_model [--input RAW_DIR] [--output RESULT_DIR]

Reads the raw message logs ``RAW_DIR/<run>/trajs/<instance>.json`` (default data/<dataset>/raw) and writes
``RESULT_DIR/cost_per_trajectory.json`` (default data/<dataset>/results).

Tokens are counted as characters / 4. Two costs are recorded per trajectory: the flat sum of all message tokens,
and the triangular cost, in which every assistant turn pays for the whole conversation before it as input and for
its own message as output. Tool outputs enter the conversation capped at 10,000 tokens, as agent frameworks
truncate long observations.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

CHARS_PER_TOKEN = 4.0
MAX_OBSERVATION_TOKENS = 10000


def count_tokens(text):
    """Approximate token count of a text; a non-empty text counts at least one token."""
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def message_text(message):
    """Text content of a message; list content is joined from the text of its parts."""
    content = message.get("content")
    if not content:
        return ""
    if isinstance(content, list):
        return "\n".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    return content if isinstance(content, str) else str(content)


def message_tokens(message):
    """Tokens of a message: its text plus the name and arguments of every tool call."""
    total = count_tokens(message_text(message))
    for call in message.get("tool_calls") or ():
        function = call.get("function", {})
        total += count_tokens(function.get("name", "")) + count_tokens(function.get("arguments", ""))
    return total


def trajectory_cost(messages):
    """Flat and triangular token cost of one conversation."""
    tokens = [message_tokens(message) for message in messages]
    triangular_input = triangular_output = context = steps = 0
    for message, count in zip(messages, tokens):
        role = message.get("role")
        if role == "assistant":
            triangular_input += context
            triangular_output += count
            steps += 1
            context += count
        elif role == "tool":
            context += min(count, MAX_OBSERVATION_TOKENS)
        else:
            context += count
    return {"num_steps": steps, "flat_tokens": sum(tokens), "triangular_input_tokens": triangular_input,
            "triangular_output_tokens": triangular_output, "triangular_total_tokens": triangular_input + triangular_output}


def summarize(costs):
    """Totals, mean and range over a list of trajectory costs."""
    flat = [c["flat_tokens"] for c in costs]
    triangular = [c["triangular_total_tokens"] for c in costs]
    return {"num_trajectories": len(costs), "total_flat_tokens": sum(flat),
            "total_triangular_input_tokens": sum(c["triangular_input_tokens"] for c in costs),
            "total_triangular_output_tokens": sum(c["triangular_output_tokens"] for c in costs),
            "total_triangular_tokens": sum(triangular), "mean_flat_tokens": round(sum(flat) / len(flat), 1),
            "mean_triangular_tokens": round(sum(triangular) / len(triangular), 1),
            "min_triangular_tokens": min(triangular), "max_triangular_tokens": max(triangular)}


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=["multi_model"])   # the runs whose raw logs are message lists
    parser.add_argument("--input", type=Path, default=None, help="raw trajectories (default: data/<dataset>/raw)")
    parser.add_argument("--output", type=Path, default=None, help="default: data/<dataset>/results")
    arguments = parser.parse_args()
    source = arguments.input or config.stage_dir("raw", arguments.dataset)
    target = arguments.output or config.stage_dir("results", arguments.dataset)
    agents = {}
    for run in config.runs_of(arguments.dataset):
        files = sorted((source / run / "trajs").glob("*.json"))
        if not files:
            raise SystemExit(f"No raw trajectories found in {source / run / 'trajs'}")
        trajectories = {path.stem: trajectory_cost(json.loads(path.read_text(encoding="utf-8"))) for path in files}
        agents[run] = {"summary": summarize(list(trajectories.values())), "trajectories": trajectories}
        print(f"  {run}: {len(trajectories)} trajectories, {agents[run]['summary']['total_triangular_tokens']:,} triangular tokens")
    everything = [cost for agent in agents.values() for cost in agent["trajectories"].values()]
    output = {"metadata": {"source": "raw_trajectory_logs", "tokenizer": f"char_ratio_{CHARS_PER_TOKEN}",
                           "max_observation_tokens": MAX_OBSERVATION_TOKENS, "num_agents": len(agents),
                           "global_summary": summarize(everything)},
              "agents": agents}
    target.mkdir(parents=True, exist_ok=True)
    (target / "cost_per_trajectory.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"{arguments.dataset}: wrote cost_per_trajectory.json ({len(everything)} trajectories)")


if __name__ == "__main__":
    main()
