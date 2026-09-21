"""Sanitization Phase 1: corpus discovery (paper Section 4.2.1).

    python pipeline/sanitize_discovery.py --dataset multi_model [--input PARSED_DIR] [--output REPORT_DIR]

Scans the parsed trajectories (task description, actions, observations) and writes three reports that
were reviewed once per dataset to set the stop-word list and the placeholder patterns:

    vocabulary_stats.csv       per token: total count, number of trajectories containing it
    discovery_frequency.json   whitespace tokens ranked by frequency (boilerplate candidates)
    discovery_shapes.json      character shapes (letters -> c/C, digits -> d) with the number of distinct
                               tokens and total occurrences per shape (identifier candidates)

The masking stage (sanitize.py) does not read these reports.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from sanitize import parsed_files  # noqa: E402

WORD_PATTERN = re.compile(r"[a-zA-Z0-9_\-\.]+")
MIN_SHAPE_TOKEN_LENGTH = 4
MIN_DISTINCT_PER_SHAPE = 6


def trajectory_text(record):
    """Task description, actions and observations of a record joined by spaces."""
    parts = [record.get("task_description", "")]
    for step in record.get("steps", []):
        parts += [step.get("action", ""), step.get("observation", "")]
    return " ".join(parts)


def token_shape(token):
    """Character shape of a token: lower-case letter -> c, upper-case -> C, digit -> d, others kept."""
    return "".join("c" if ch.islower() else "C" if ch.isupper() else "d" if ch.isdigit() else ch for ch in token)


def guess_token_type(token):
    """Rough token type used to sort the vocabulary report during review."""
    if re.match(r"^\d+$", token):
        return "number"
    if re.match(r"^[a-f0-9]{8,}$", token):
        return "hex/hash"
    if "." in token or "/" in token:
        return "path/file"
    return "snake_case_var" if "_" in token else "word"


def scan(files):
    """Count word tokens (total and per trajectory), whitespace tokens and token shapes."""
    term_count, document_count, whitespace_count = Counter(), Counter(), Counter()
    shapes = defaultdict(Counter)
    for path in files:
        text = trajectory_text(json.loads(path.read_text(encoding="utf-8")))
        words = WORD_PATTERN.findall(text.lower())
        term_count.update(words)
        document_count.update(set(words))
        tokens = text.split()
        whitespace_count.update(tokens)
        for token in tokens:
            if len(token) >= MIN_SHAPE_TOKEN_LENGTH:
                shapes[token_shape(token)][token] += 1
    return term_count, document_count, whitespace_count, shapes


def write_vocabulary(path, term_count, document_count, n_trajectories):
    """vocabulary_stats.csv, most widespread tokens first."""
    header = ["token", "type_guess", "total_count (TF)", "doc_count (DF)", "percent_trajectories", "signal_strength"]
    rows = []
    for token, documents in sorted(document_count.items(), key=lambda item: (-item[1], item[0])):
        percent = 100.0 * documents / n_trajectories
        rows.append([token, guess_token_type(token), term_count[token], documents, round(percent, 4),
                     "HIGH" if 10 < percent < 90 else "LOW (Stop/Rare)"])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def frequency_report(whitespace_count):
    """Whitespace tokens ranked by count, with the cumulative share of the corpus."""
    total, cumulative, report = sum(whitespace_count.values()), 0, []
    ranked = sorted(whitespace_count.items(), key=lambda item: (-item[1], item[0]))
    for rank, (token, count) in enumerate(ranked, start=1):
        cumulative += count
        report.append({"rank": rank, "token": token, "count": count, "cumulative_percent": round(100.0 * cumulative / total, 4)})
    return report


def shape_report(shapes):
    """Shapes produced by at least six distinct tokens, most distinct tokens first."""
    report = []
    for shape, examples in shapes.items():
        if len(examples) >= MIN_DISTINCT_PER_SHAPE:
            total = sum(examples.values())
            top = [token for token, _ in sorted(examples.items(), key=lambda item: (-item[1], item[0]))[:5]]
            report.append({"shape": shape, "unique_tokens_count": len(examples), "total_volume": total,
                           "variety_ratio": round(len(examples) / total, 4), "examples": top})
    return sorted(report, key=lambda item: (-item["unique_tokens_count"], item["shape"]))


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
    output_dir.mkdir(parents=True, exist_ok=True)
    term_count, document_count, whitespace_count, shapes = scan(files)
    write_vocabulary(output_dir / "vocabulary_stats.csv", term_count, document_count, len(files))
    frequency, shape_rows = frequency_report(whitespace_count), shape_report(shapes)
    (output_dir / "discovery_frequency.json").write_text(json.dumps(frequency, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "discovery_shapes.json").write_text(json.dumps(shape_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"discovery: read {len(files)} parsed files from {input_dir}; {len(whitespace_count)} distinct whitespace tokens, "
          f"{len(shape_rows)} shapes; wrote 3 reports to {output_dir}")
    print("most frequent tokens:", ", ".join(repr(item["token"]) for item in frequency[:25]))
    varied = [item for item in shape_rows if item["variety_ratio"] > 0.5][:10]
    print("high-variety shapes:", "; ".join(f"{item['shape'][:30]} ({item['unique_tokens_count']} distinct / {item['total_volume']})" for item in varied))


if __name__ == "__main__":
    main()
