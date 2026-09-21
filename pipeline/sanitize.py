"""Stage 2: unified schema -> sanitized trajectories (paper Section 4.2.3, "Sanitize & Mask").

    python pipeline/sanitize.py --dataset multi_model [--input PARSED_DIR] [--output SANITIZED_DIR]

The engine is shared; the term lists and patterns come from the dataset's module in
``sanitization_rules/``. For the task description and for every step's action and observation it

1. replaces words that contain a part of the trajectory's repository name by ``[REPO_VAR]``,
2. deletes test-runner progress bars (datasets that define a pattern for them) and the outcome-leakage terms,
3. replaces identifier shapes by typed placeholders,
4. tokenises, drops harness stop words, noise terms and repository-name parts,
5. maps error types to abstract tokens and collapses repeated placeholders.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import sanitization_rules  # noqa: E402

TOKEN_PATTERN = re.compile(r"\[[A-Z_]+\]|[\w]+")
NAME_SEPARATORS = re.compile(r"[/_-]")


def parsed_files(input_dir):
    """All unified-schema files below a folder, in sorted order."""
    return sorted(path for path in input_dir.rglob("*.json") if path.name.startswith("unified_"))


def repository_terms(files, rules, input_dir):
    """Parts of all repository names of the dataset (removed as tokens everywhere), including the names listed in
    ``repository_names.json`` of a run-group dataset."""
    terms = set()
    listed = input_dir / "repository_names.json"
    names = json.loads(listed.read_text()) if listed.exists() else []
    for repository in names + [json.loads(path.read_text(encoding="utf-8")).get("repo_name", "") for path in files]:
        for part in NAME_SEPARATORS.split(repository.lower()):
            if len(part) >= rules.REPOSITORY_TERM_MIN_LENGTH and part not in rules.PROTECTED_TERMS:
                terms.add(part)
    return terms


def repository_compound_pattern(repository, rules):
    """Pattern matching any word that contains a part of this repository's name, or None."""
    if not rules.MASK_REPOSITORY_COMPOUNDS or not repository:
        return None
    parts = [part for part in NAME_SEPARATORS.split(repository.lower())
             if len(part) >= rules.REPOSITORY_COMPOUND_MIN_LENGTH and part not in rules.PROTECTED_TERMS]
    if not parts:
        return None
    return re.compile(r"\b\w*(" + "|".join(map(re.escape, parts)) + r")\w*\b", re.IGNORECASE)


def leakage_pattern(rules):
    """One pattern for all outcome-leakage terms, longest first."""
    terms = sorted(rules.OUTCOME_LEAKAGE_TERMS, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(map(re.escape, terms)) + r")\b", re.IGNORECASE)


def sanitize_text(text, rules, leakage, removed_tokens, repository_pattern):
    """Sanitized version of one text field."""
    if not text:
        return ""
    if repository_pattern is not None:
        text = repository_pattern.sub("[REPO_VAR]", text)
    progress_bars = getattr(rules, "PROGRESS_BAR_PATTERN", None)   # only datasets whose corpus contains them define it
    if progress_bars is not None:
        text = progress_bars.sub(" ", text)
    text = leakage.sub(" ", text)
    for pattern, placeholder in rules.PLACEHOLDER_PATTERNS:
        text = pattern.sub(placeholder, text)
    kept = []
    for token in TOKEN_PATTERN.findall(text):
        lowered = token.lower()
        if lowered in rules.HARNESS_STOP_WORDS or lowered in removed_tokens:
            continue
        if lowered in rules.ERROR_TOKENS:
            kept.append(rules.ERROR_TOKENS[lowered])
        elif not (lowered.startswith("[") and kept and kept[-1] == token):
            kept.append(token)
    return " ".join(kept)


def sanitize_record(record, rules, leakage, removed_tokens):
    """Sanitized copy of a unified-schema record; only text fields change."""
    pattern = repository_compound_pattern(record.get("repo_name", ""), rules)
    clean = lambda text: sanitize_text(text, rules, leakage, removed_tokens, pattern)  # noqa: E731
    sanitized = {"trajectory_id": record.get("trajectory_id"), "repo_name": record.get("repo_name"),
                 "agent_name": record.get("agent_name"), "final_outcome": record.get("final_outcome"),
                 "task_description": clean(record.get("task_description", "")), "steps": []}
    for step in record.get("steps", []):
        step = dict(step)
        for field in ("action", "observation"):
            if field in step:
                step[field] = clean(step[field])
        sanitized["steps"].append(step)
    return sanitized


def sanitize_dataset(dataset_name, input_dir, output_dir):
    """Sanitize every parsed file of a dataset; return the number of files written.

    A run listed under ``run_sanitization_rules`` in the dataset definition uses the rules named there.
    """
    definition = config.dataset(dataset_name)
    files = parsed_files(input_dir)
    if not files:
        raise SystemExit(f"No parsed files found in {input_dir}")
    prepared = {}
    for name in {definition["sanitization_rules"], *definition.get("run_sanitization_rules", {}).values()}:
        rules = sanitization_rules.load(name)
        prepared[name] = (rules, leakage_pattern(rules), rules.NOISE_TERMS | repository_terms(files, rules, input_dir))
        print(f"  rules '{name}': {len(rules.OUTCOME_LEAKAGE_TERMS)} leakage terms, {len(prepared[name][2])} removed tokens "
              f"(noise terms and repository-name parts)")
    print(f"  read {len(files)} parsed files from {input_dir}")
    for path in files:
        relative = path.relative_to(input_dir)
        name = definition.get("run_sanitization_rules", {}).get(relative.parts[0], definition["sanitization_rules"])
        rules, leakage, removed_tokens = prepared[name]
        record = json.loads(path.read_text(encoding="utf-8"))
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(sanitize_record(record, rules, leakage, removed_tokens), handle, indent=2)
    print(f"  wrote {len(files)} sanitized files to {output_dir}")
    return len(files)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(config.DATASETS))
    parser.add_argument("--input", type=Path, default=None, help="parsed folder (default: data/<dataset>/parsed)")
    parser.add_argument("--output", type=Path, default=None, help="sanitized folder (default: data/<dataset>/sanitized)")
    arguments = parser.parse_args()
    started = time.time()
    total = sanitize_dataset(arguments.dataset, arguments.input or config.stage_dir("parsed", arguments.dataset),
                             arguments.output or config.stage_dir("sanitized", arguments.dataset))
    print(f"sanitize: {total} trajectories, {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
