"""Sanitization Phase 4: validation (paper Section 4.2.4).

    python pipeline/sanitize_validate.py --dataset multi_model [--parsed PARSED_DIR] [--sanitized SANITIZED_DIR]
                                         [--output REPORT_DIR] [--seed 42]

Two checks, both written to ``sanitization_validation.json``:

1. Outcome probe. A TF-IDF + logistic-regression (SGD) classifier predicts pass/fail from the text of at
   most 5,000 trajectories, once on the parsed corpus and once on the sanitized corpus (70/30 split), plus
   an ablation on the sanitized corpus with a list of task-related terms removed. The report holds the
   accuracies, the majority-class rate and the words with the largest positive and negative coefficients.
2. Term scan. A sample of 200 sanitized trajectories is scanned for outcome and repository terms that
   the masking stage must have removed.

File sampling is seeded (``--seed``), so the report is reproducible.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from sanitize import parsed_files  # noqa: E402

MAX_TRAJECTORIES = 5000
SCAN_SAMPLE_SIZE = 200
TOP_WORDS = 20
CLASSIFIER_SEED = 42
ABLATION_TERMS = [
    "test", "tests", "testing", "tested", "pytest", "unittest", "mock", "fixture", "fixtures",
    "assert", "assertion", "assertions", "hypothesis", "bdd", "asyncio", "setup", "teardown",
    "database", "db_field", "import",
]
SCANNED_TERMS = ("passed", "failed", "dvc", "stix2", "beetbox", "scrapy", "xarray", "pylint", "tox")


def load_corpus(folder, seed):
    """Texts and pass/fail labels of at most MAX_TRAJECTORIES trajectories, sampled with a fixed seed."""
    files = parsed_files(folder)
    random.Random(seed).shuffle(files)
    texts, labels = [], []
    for path in files[:MAX_TRAJECTORIES]:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("final_outcome") is None:
            continue
        body = record.get("task_description", "") + " "
        for step in record.get("steps", []):
            body += f"{step.get('action', '')} {step.get('observation', '')} "
        texts.append(body)
        labels.append(1 if record["final_outcome"] else 0)
    return texts, labels


def outcome_probe(texts, labels, removed_terms=()):
    """Train the probe and return its accuracy, the majority rate and the top-coefficient words."""
    stop_words = list(sklearn_text.ENGLISH_STOP_WORDS) + list(removed_terms)
    vectorizer = TfidfVectorizer(max_features=10000, stop_words=stop_words)
    features = vectorizer.fit_transform(texts)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.3, random_state=CLASSIFIER_SEED)
    classifier = SGDClassifier(loss="log_loss", max_iter=100, random_state=CLASSIFIER_SEED).fit(train_x, train_y)
    names, coefficients = vectorizer.get_feature_names_out(), classifier.coef_[0]
    order = coefficients.argsort()
    return {"trajectories": len(labels), "majority_class_rate": max(labels.count(0), labels.count(1)) / len(labels),
            "accuracy": float(accuracy_score(test_y, classifier.predict(test_x))),
            "top_words_predicting_pass": [[names[i], round(float(coefficients[i]), 2)] for i in order[-TOP_WORDS:][::-1]],
            "top_words_predicting_fail": [[names[i], round(float(coefficients[i]), 2)] for i in order[:TOP_WORDS]]}


def term_scan(sanitized_dir, seed):
    """Occurrences of the scanned terms (whole words) in a seeded sample of sanitized trajectories."""
    files = parsed_files(sanitized_dir)
    sample = random.Random(seed).sample(files, min(SCAN_SAMPLE_SIZE, len(files)))
    hits, files_with_hits = Counter(), 0
    for path in sample:
        record = json.loads(path.read_text(encoding="utf-8"))
        body = record.get("task_description", "") + " "
        for step in record.get("steps", []):
            body += step.get("action", "") + " " + step.get("observation", "")
        found = [term for term in SCANNED_TERMS if re.search(rf"\b{re.escape(term)}\b", body.lower())]
        hits.update(found)
        files_with_hits += bool(found)
    return {"sampled_trajectories": len(sample), "scanned_terms": list(SCANNED_TERMS),
            "trajectories_with_hits": files_with_hits, "hits_by_term": dict(hits)}


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(config.DATASETS))
    parser.add_argument("--parsed", type=Path, default=None, help="parsed folder (default: data/<dataset>/parsed)")
    parser.add_argument("--sanitized", type=Path, default=None, help="sanitized folder (default: data/<dataset>/sanitized)")
    parser.add_argument("--output", type=Path, default=None, help="report folder (default: data/<dataset>/sanitization_reports)")
    parser.add_argument("--seed", type=int, default=config.SEED, help="seed of the file sampling")
    arguments = parser.parse_args()
    parsed_dir = arguments.parsed or config.stage_dir("parsed", arguments.dataset)
    sanitized_dir = arguments.sanitized or config.stage_dir("sanitized", arguments.dataset)
    output_dir = arguments.output or config.stage_dir("sanitization_reports", arguments.dataset)
    parsed_corpus, sanitized_corpus = load_corpus(parsed_dir, arguments.seed), load_corpus(sanitized_dir, arguments.seed)
    report = {"dataset": arguments.dataset, "seed": arguments.seed,
              "probe_on_parsed_corpus": outcome_probe(*parsed_corpus),
              "probe_on_sanitized_corpus": outcome_probe(*sanitized_corpus),
              "probe_on_sanitized_corpus_without_task_terms": outcome_probe(*sanitized_corpus, removed_terms=ABLATION_TERMS),
              "term_scan": term_scan(sanitized_dir, arguments.seed)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sanitization_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name in ("probe_on_parsed_corpus", "probe_on_sanitized_corpus", "probe_on_sanitized_corpus_without_task_terms"):
        probe = report[name]
        print(f"{name}: accuracy {probe['accuracy']:.4f} (majority class {probe['majority_class_rate']:.4f}, {probe['trajectories']} trajectories)")
    scan = report["term_scan"]
    print(f"term scan: {scan['trajectories_with_hits']} of {scan['sampled_trajectories']} sampled trajectories contain a scanned term {scan['hits_by_term']}")
    print(f"read {parsed_dir} and {sanitized_dir}; wrote {output_dir / 'sanitization_validation.json'}")


if __name__ == "__main__":
    main()
