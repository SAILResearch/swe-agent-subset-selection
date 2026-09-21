"""Sanitization Phase 2: outcome-leakage detection by log-likelihood keyness (paper Section 4.2.2).

    python pipeline/sanitize_leakage.py --dataset multi_model [--input PARSED_DIR] [--output REPORT_DIR]

Splits the parsed corpus into passing and failing trajectories, counts every token in both groups and
computes the log-likelihood ratio G2. Tokens with G2 above the chi-square critical value for p < 0.001 are
flagged for review. Output: ``sanitization_candidates.json`` (all tokens occurring at least 20 times,
highest G2 first). The flagged list was reviewed once per dataset; the masking stage does not read it.

Before counting, the identifier shapes and harness stop words known from Phase 1 are removed so that the
statistics concentrate on content words. These two lists are set here and are independent of ``sanitization_rules/``.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from sanitize import parsed_files  # noqa: E402
from sanitize_discovery import trajectory_text  # noqa: E402

G2_CRITICAL_VALUE = 10.83          # chi-square, 1 degree of freedom, p < 0.001
MIN_TOKEN_COUNT = 20
MIN_TOKEN_LENGTH = 3
STRIP_CHARACTERS = ".,;:()[]'\""

PHASE1_STOP_WORDS = {
    "_", "=", "#", "-", "[", "]", "&&", "+", "==", "0]", "1",
    "execute_bash:", "[The", "[Python", "interpreter:", "[Command", "[Current",
    "/opt/conda/envs/testbed/bin/python]", "directory:", "command", "working",
    "cd", "completed", "finished", "result", "File", "str_replace_editor:",
    "0.]", "exit", "code",
    "the", "in", "with", "to", "of", "if", "is", "for", "and", "a", "not", "as", "on", "or", "be",
}
PHASE1_PATTERNS = [
    (r"--randomly-seed=\d+", "[RANDOM_SEED]"),
    (r"\d{2}:\d{2}:\d{2}\.\d+", "[TIMESTAMP]"),
    (r"id='\d+'>", "[OBJ_ID]"),
    (r"#\d+", "[ISSUE_ID]"),
    (r"file_\d+\.txt", "[GEN_FILE]"),
    (r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "[UUID]"),
    (r"\/[\w\-\.\/]+", "[PATH]"),
    (r"\b\d+\.\d+\b", "[FLOAT]"),
    (r"\b\d+\b", "[NUM]"),
]


def content_tokens(text):
    """Whitespace tokens after placeholder substitution, stripped of punctuation, without stop words."""
    for pattern, placeholder in PHASE1_PATTERNS:
        text = re.sub(pattern, placeholder, text)
    tokens = (token.strip(STRIP_CHARACTERS) for token in text.split())
    return [token for token in tokens if token not in PHASE1_STOP_WORDS and len(token) >= MIN_TOKEN_LENGTH]


def log_likelihood(count_pass, total_pass, count_fail, total_fail):
    """G2 of a token's counts in the passing and failing corpora."""
    expected_pass = total_pass * (count_pass + count_fail) / (total_pass + total_fail)
    expected_fail = total_fail * (count_pass + count_fail) / (total_pass + total_fail)
    g2 = count_pass * math.log(count_pass / expected_pass) if count_pass else 0.0
    g2 += count_fail * math.log(count_fail / expected_fail) if count_fail else 0.0
    return 2.0 * g2


def keyness_report(files):
    """Keyness of every sufficiently frequent token, highest G2 first; also the corpus sizes."""
    counts = {True: Counter(), False: Counter()}
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        counts[bool(record.get("final_outcome", False))].update(content_tokens(trajectory_text(record)))
    total_pass, total_fail = sum(counts[True].values()), sum(counts[False].values())
    report = []
    for token in set(counts[True]) | set(counts[False]):
        count_pass, count_fail = counts[True][token], counts[False][token]
        if count_pass + count_fail < MIN_TOKEN_COUNT:
            continue
        g2 = log_likelihood(count_pass, total_pass, count_fail, total_fail)
        rate_pass, rate_fail = 10000.0 * count_pass / total_pass, 10000.0 * count_fail / total_fail
        report.append({"token": token, "category": "LEAKAGE_RISK" if g2 > G2_CRITICAL_VALUE else "NEUTRAL",
                       "keyness_score": round(g2, 2), "skew_direction": "FAIL" if rate_fail > rate_pass else "PASS",
                       "count_pass": count_pass, "count_fail": count_fail, "normalized_diff": round(abs(rate_pass - rate_fail), 4)})
    report.sort(key=lambda item: (-item["keyness_score"], item["token"]))
    return report, total_pass, total_fail


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(config.DATASETS))
    parser.add_argument("--input", type=Path, default=None, help="parsed folder (default: data/<dataset>/parsed)")
    parser.add_argument("--output", type=Path, default=None, help="report folder (default: data/<dataset>/sanitization_reports)")
    arguments = parser.parse_args()
    input_dir = arguments.input or config.stage_dir("parsed", arguments.dataset)
    output_dir = arguments.output or config.stage_dir("sanitization_reports", arguments.dataset)
    files = parsed_files(input_dir)
    if not files:
        raise SystemExit(f"No parsed files found in {input_dir}")
    report, total_pass, total_fail = keyness_report(files)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sanitization_candidates.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    flagged = [item for item in report if item["category"] == "LEAKAGE_RISK"]
    print(f"leakage: read {len(files)} parsed files from {input_dir}; {total_pass} passing and {total_fail} failing tokens; "
          f"{len(report)} tokens analysed, {len(flagged)} flagged; wrote {output_dir / 'sanitization_candidates.json'}")
    print("highest keyness:", ", ".join(f"{item['token']} ({item['keyness_score']}, {item['skew_direction']})" for item in flagged[:20]))


if __name__ == "__main__":
    main()
