"""Sanitization rules of the multi-model dataset (OpenHands runs on SWE-Bench Verified).

The lists were fixed once, after reviewing this dataset's corpus-discovery and outcome-leakage reports
(``sanitize_discovery.py`` and ``sanitize_leakage.py``); the masking stage does not read those reports.
"""
import re

# Task and behaviour terms that are kept even when they are part of a repository name.
PROTECTED_TERMS = {
    "python", "java", "cpp", "asyncio", "test", "tests", "testing",
    "hypothesis", "fixture", "mock", "pytest", "unittest", "doc", "docs",
    "bug", "issue", "error", "fix", "patch", "code", "run", "profile", "loop",
    "creating", "cloning", "destroying", "default", "database", "alias",
}

# Boilerplate tokens of the evaluation harness and high-frequency function words (Phase 1).
# Tokens are compared in lower case, so entries with capitals or brackets are inert; they are kept as reviewed.
HARNESS_STOP_WORDS = {
    "_", "=", "#", "-", "[", "]", "&&", "+", "==", "0]", "1",
    "execute_bash:", "[The", "[Python", "interpreter:", "[Command", "[Current",
    "/opt/conda/envs/testbed/bin/python]", "directory:", "command", "working",
    "/opt/miniconda3/envs/testbed/bin/python]",
    "cd", "completed", "finished", "result", "File", "str_replace_editor:",
    "0.]", "exit", "code", "root",
    "the", "in", "with", "to", "of", "if", "is", "for", "and", "a", "not", "as", "on", "or", "be",
    "apr", "may", "jun", "jul", "aug",
}

# Identifier shapes replaced by typed placeholders (Phase 1), applied in this order.
PLACEHOLDER_PATTERNS = [
    (re.compile(r"\b\w+__\w+__\d+\b", re.IGNORECASE), "[REPO_ID]"),
    (re.compile(r"PYTHONHASHSEED=\d+", re.IGNORECASE), "[RANDOM_SEED]"),
    (re.compile(r"--randomly-seed=\d+"), "[RANDOM_SEED]"),
    (re.compile(r"\d{2}:\d{2}:\d{2}\.\d+"), "[TIMESTAMP]"),
    (re.compile(r"id='\d+'>"), "[OBJ_ID]"),
    (re.compile(r"#\d+"), "[ISSUE_ID]"),
    (re.compile(r"file_\d+\.txt"), "[GEN_FILE]"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "[UUID]"),
    (re.compile(r"\/[\w\-\.\/]+"), "[PATH]"),
    (re.compile(r"\b\d+\.\d+\b"), "[FLOAT]"),
    (re.compile(r"\b\d+\b"), "[NUM]"),
]

# Outcome-leakage and repository-tooling terms (Phase 2), deleted wherever they occur as whole words.
OUTCOME_LEAKAGE_TERMS = {
    "passed", "failed", "failures", "errors",
    "skipped", "xfailed", "xpassed", "fail", "pass",
    "success", "resolved", "successfully", "fixed",
    "tox", "pylint", "xarray", "dvc", "stix2", "scrapy", "beetbox",
    "execution", "result", "observation", "antlr",
    "w0201", "attribute-defined-outside-init", "noqa", "f811",
    "a*([num]*a",
}

# Error types mapped to abstract tokens.
ERROR_TOKENS = {
    "traceback": "[LOG_TRACE]",
    "assertionerror": "[ERR_ASSERT]",
    "keyerror": "[ERR_KEY]",
    "valueerror": "[ERR_VALUE]",
    "attributeerror": "[ERR_ATTR]",
    "modulenotfounderror": "[ERR_IMPORT]", "importerror": "[ERR_IMPORT]",
    "typeerror": "[ERR_TYPE]",
    "nameerror": "[ERR_NAME]",
    "indexerror": "[ERR_INDEX]",
    "syntaxerror": "[ERR_SYNTAX]",
    "indentationerror": "[ERR_INDENT]",
    "timeouterror": "[ERR_TIMEOUT]",
    "exception": "[ERR_GENERIC]", "fatal": "[ERR_FATAL]", "error": "[ERR_GENERIC]",
}

# Test-harness and benchmark-tool noise removed as whole tokens.
NOISE_TERMS = {
    "benchmark", "benchmarks", "timer", "calibration_precision", "min_rounds",
    "min_time", "max_time", "mean", "stddev", "rounds", "iterations",
    "warmup", "warmup_iterations", "disable_gc", "cpu", "msec",
    "pytestdeprecationwarning", "warnings", "warn", "fixture", "fixtures",
    "default_loop_scope", "asyncio_default_fixture_loop_scope",
    "collected", "items", "session", "starts", "most", "recent", "call", "last",
    "randomly", "random_seed", "random", "seed",
    "assertequal", "assertequals", "assert_equal",
    "permission", "auth", "azimuth_time",
}

# Words containing a part of the trajectory's own repository name are replaced by [REPO_VAR].
MASK_REPOSITORY_COMPOUNDS = True
REPOSITORY_COMPOUND_MIN_LENGTH = 4   # repository-name parts shorter than this are not masked
REPOSITORY_TERM_MIN_LENGTH = 3       # repository-name parts shorter than this are not removed as tokens
