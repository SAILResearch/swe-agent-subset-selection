"""Shared loading, naming and formatting helpers for the research-question scripts.

By default the scripts read from ``results/`` and write to their own ``outputs/`` folder; ``--results`` and
``--outputs`` (see ``configure``) change both. The defaults are resolved relative to the package root, so the
scripts can be run from any working directory.
"""
import argparse
import csv
import gzip
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_FOLDERS = {"results": PACKAGE_ROOT / "results", "outputs": None}

ALPHA = 0.05      # significance level after Holm correction
MINUS = "−"       # the paper's minus sign (U+2212)
PLOT_STYLE = {
    "font.family": "serif", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "axes.spines.top": False, "axes.spines.right": False,
}


def configure(description, default_outputs):
    """Read ``--results`` and ``--outputs`` from the command line of a research-question script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--results", type=Path, default=PACKAGE_ROOT / "results", help="folder with the result files (default: results/)")
    parser.add_argument("--outputs", type=Path, default=default_outputs, help="folder for the generated files (default: the script's outputs/)")
    options = parser.parse_args()
    _FOLDERS.update(results=options.results.resolve(), outputs=options.outputs.resolve())


def results_dir():
    """Folder the result files are read from."""
    return _FOLDERS["results"]


def output_dir():
    """Folder the generated files are written to."""
    return _FOLDERS["outputs"]


def shown(path):
    """Path relative to the package root when it lies inside the package."""
    return path.relative_to(PACKAGE_ROOT) if path.is_relative_to(PACKAGE_ROOT) else path

# Scenario identifiers (folder names under results/) and their paper names.
SCENARIOS = ("single_setup", "multi_model", "multi_agent")
SCENARIO_TITLES = {
    "single_setup": "Single-Setup",
    "multi_model": "Multi-Model",
    "multi_agent": "Multi-Agent",
}
SCENARIO_ABBREVIATIONS = {"single_setup": "SS", "multi_model": "MM", "multi_agent": "MA"}
SYNTHETIC_SCENARIOS = ("multi_model", "multi_agent")

SUBSET_SIZES = ("5pct", "10pct", "20pct", "30pct")
SUBSET_LABELS = {"5pct": "5%", "10pct": "10%", "20pct": "20%", "30pct": "30%"}

# Baselines in the row order of Tables 3 and 4, with the paper's display names.
BASELINES = (
    "random",
    "difficulty_stratified",
    "consistency_stratified",
    "repo_stratified",
    "repo_difficulty_stratified",
    "repo_consistency_stratified",
)
BASELINE_LABELS = {
    "random": "Random",
    "difficulty_stratified": "Diff-Strat",
    "consistency_stratified": "Consist-Strat",
    "repo_stratified": "Repo-Strat",
    "repo_difficulty_stratified": "Repo\u00d7Diff",
    "repo_consistency_stratified": "Repo\u00d7Consist",
}
STRONGEST_BASELINE = "consistency_stratified"

# The single-setup per-seed file aggregates temporal splits in two ways; the paper
# reports the mean over splits, which the stored schema names as below.
SINGLE_SETUP_SPLIT_AGGREGATION_KEY = "fold_mean"


def load_json(path):
    """Load a JSON file; if only a gzipped copy (``<name>.json.gz``) exists, read that instead."""
    path = Path(path)
    compressed = path.with_name(path.name + ".gz")
    if not path.exists() and compressed.exists():
        with gzip.open(compressed, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with open(path) as handle:
        return json.load(handle)


@lru_cache(maxsize=None)
def load_aggregated(scenario):
    """Load ``results/<scenario>/aggregated_results.json`` or its ``.json.gz`` copy (cached; the files are large)."""
    return load_json(results_dir() / scenario / "aggregated_results.json")


@lru_cache(maxsize=None)
def load_per_seed_maxerr(scenario):
    """Load the per-seed MaxErr statistics of all baselines for one scenario."""
    return load_json(results_dir() / scenario / "per_seed_maxerr_all_baselines.json")


def analysis(scenario, window="overall"):
    """Return one analysis block: ``"overall"`` or a window size such as ``2``."""
    tag = window if window == "overall" else f"w{window}"
    return load_aggregated(scenario)["analyses"][tag]


def window_sizes(scenario):
    """Return the window sizes (W) that have a stored analysis block."""
    tags = load_aggregated(scenario)["analyses"]
    return sorted(int(tag[1:]) for tag in tags if tag != "overall")


def method_key(scenario, group, method):
    """Return the stored key of a method; the separator differs between result files."""
    separator = "_" if scenario == "single_setup" else ":"
    return f"{group}{separator}{method}"


def method_stats(scenario, group, method, subset, window="overall"):
    """Return the stored statistics of one method, or ``None`` when it was not evaluated."""
    methods = analysis(scenario, window)["all_methods"][subset]
    return methods.get(method_key(scenario, group, method))


def comparison_row(scenario, subset, reference, method_name, window="overall"):
    """Return the stored paired comparison of ``method_name`` against a reference.

    ``reference`` is ``"vs_random"`` or ``"vs_best_baseline"``.
    """
    rows = analysis(scenario, window)["subset_sizes"][subset]["ranking"][reference]
    matches = [row for row in rows if row["name"] == method_name]
    return matches[0] if matches else None


def per_seed_maxerr(scenario, baseline, subset):
    """Return (median, 95th percentile, worst) per-seed MaxErr of a baseline, in percent.

    Multi-model and multi-agent: each statistic is taken over the 500 seeds within a
    distribution and then averaged over the 1,000 distributions. Single-setup: taken
    over seeds within a (run group, window) cell and then averaged over cells.
    """
    stored = load_per_seed_maxerr(scenario)
    if scenario == "single_setup":
        cell = stored["pooled"][baseline][subset][SINGLE_SETUP_SPLIT_AGGREGATION_KEY]
        return cell["baseline_seed_median"], cell["baseline_seed_p95"], cell["baseline_seed_max"]
    cell = stored["results"][baseline][subset]
    return (
        cell["baseline_seed_median_mean"],
        cell["baseline_seed_p95_mean"],
        cell["baseline_seed_max_mean"],
    )


def write_csv(path, header, rows):
    """Write a CSV file, creating parent folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_text(path, text):
    """Write a text file, creating parent folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def tex_escape(text):
    """Escape the characters of the labels that are special in LaTeX."""
    return text.replace("%", "\\%").replace("\u00d7", "$\\times$")


# Per-seed extract

def load_extract(scenario):
    """Per-seed MaxErr extract of a scenario: {(subset, window, name): array}."""
    arrays = {}
    with np.load(results_dir() / scenario / "per_seed_maxerr_extract.npz") as npz:
        for key in npz.files:
            subset, window, name = key.split("__")
            arrays[(subset, int(window[1:]), name)] = npz[key].astype(float)
    return arrays


def extract_windows(arrays):
    """Window sizes present in an extract."""
    return sorted({window for _, window, _ in arrays})


def seed_statistics(arrays, subset, baseline):
    """Per-distribution (median, P95, worst) MaxErr over the 500 seeds of a stochastic baseline.

    Each statistic is taken within a window and then averaged over windows with equal weight.
    """
    matrices = [arrays[(subset, w, baseline)] for w in extract_windows(arrays)]
    return (np.mean([np.median(m, axis=1) for m in matrices], axis=0),
            np.mean([np.percentile(m, 95, axis=1) for m in matrices], axis=0),
            np.mean([m.max(axis=1) for m in matrices], axis=0))


# Statistics

def holm_adjust(p_values):
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    order = np.argsort(p_values)
    adjusted, running_max = np.empty(len(p_values)), 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def paired_comparison(reference_errors, other_errors):
    """Two-sided Wilcoxon p, paired Cliff's delta and win/tie/loss of reference vs other.

    A win means the reference has the lower error on that evaluation unit; delta is
    (wins - losses) / n, positive when the reference is better.
    """
    wins = int(np.sum(reference_errors < other_errors))
    losses = int(np.sum(reference_errors > other_errors))
    ties = len(reference_errors) - wins - losses
    p_value = wilcoxon(reference_errors, other_errors).pvalue
    return p_value, (wins - losses) / len(reference_errors), (wins, ties, losses)


def unpaired_cliffs_delta(first_errors, second_errors):
    """Classical Cliff's delta over all pairs; positive when the first method has the lower errors."""
    difference = np.asarray(second_errors)[None, :] - np.asarray(first_errors)[:, None]
    return float((np.sum(difference > 0) - np.sum(difference < 0)) / difference.size)


def effect_size_letter(delta):
    """Magnitude letter of Cliff's delta: N, S, M or L (thresholds 0.147, 0.33, 0.474)."""
    size = abs(delta)
    return "N" if size < 0.147 else "S" if size < 0.33 else "M" if size < 0.474 else "L"


# Reproduced values

ALGORITHM_LABELS = {"centroid": "Centroid", "core_edge": "Core/Edge", "medoid": "Medoid", "fl": "FL", "ks": "KS"}


def signed(value, decimals):
    """Signed number with the paper's minus sign, e.g. +1.4 or −8.4."""
    return f"{value:+.{decimals}f}".replace("-", MINUS)


def paper_names(text):
    """Replace stored identifiers in a text by the names used in the paper."""
    for scenario, title in (("single_setup", "Single-setup"), ("multi_model", "Multi-model"), ("multi_agent", "Multi-agent")):
        text = text.replace(scenario, title)
    for algorithm, label in ALGORITHM_LABELS.items():
        text = re.sub(rf"\b{algorithm}_pooled\b", f"{label} Pooled", text)
        text = re.sub(rf"\b{algorithm}_ts\b", f"{label} TS", text)
    for baseline, label in BASELINE_LABELS.items():
        text = re.sub(rf"\b{baseline}\b", label, text)
    text = text.replace("stability_stratified", "Stability-Stratified")
    return re.sub(r"\b(\d+)pct\b", r"\1%", text)


def _as_number(cell):
    """Numeric value of a printed table cell (paper minus sign, thousands separators, %, x), or None."""
    text = cell.replace("\u2212", "-").replace(",", "").rstrip("%\u00d7")
    try:
        return float(text)
    except ValueError:
        return None


def _last_digit_step(cell):
    """Size of one unit in the last printed digit of a numeric cell."""
    digits = cell.rstrip("%\u00d7")
    return 10.0 ** -(len(digits) - digits.index(".") - 1) if "." in digits else 1.0


class ReproducedValues:
    """Collects one record per number stated in a section of the paper.

    Each record holds the paper location, the statement, the value in the paper, the value
    reproduced from ``results/`` and how it is computed. ``write`` produces
    ``reproduced_values.md`` and ``reproduced_values.csv``.
    """

    HEADER = ("Paper location", "Statement", "Value in the paper", "Reproduced value", "How it is computed")

    def __init__(self, title):
        """``title`` is the heading of the written files."""
        self.title = title
        self.records = []

    def _add(self, location, statement, paper, reproduced, how):
        """Store one record, writing scenario, subset and method identifiers with the paper's names.

        ``how="same"`` repeats the text of the previous record.
        """
        if how == "same" and self.records:
            how = self.records[-1]["how"]
        self.records.append({"location": location, "statement": paper_names(statement), "paper": paper,
                             "reproduced": paper_names(reproduced), "how": how})

    def number(self, location, statement, paper, reproduced, decimals, how):
        """A single printed number, reproduced with two more decimals than printed and at least four."""
        self._add(location, statement, f"{paper:.{decimals}f}", f"{reproduced:.{max(decimals + 2, 4)}f}", how)

    def number_range(self, location, statement, paper_low, paper_high, reproduced_values, how, decimals=0):
        """A printed range, reproduced as the minimum and maximum of the underlying values."""
        low, high = float(min(reproduced_values)), float(max(reproduced_values))
        paper = f"{paper_low}" if paper_low == paper_high else f"{paper_low}-{paper_high}"
        self._add(location, statement, paper, f"{low:.{decimals + 2}f}-{high:.{decimals + 2}f}", how)

    def statement(self, location, statement, paper, holds, reproduced, how):
        """A statement with a countable or listable value."""
        self._add(location, statement, paper, reproduced, how)

    def table(self, table_name, generated_rows, printed_rows, row_names=None, column_names=None,
              rows_label="All printed cells", note="", describe_gaps=True):
        """A whole table, or some of its rows: the generated cells compared with the printed cells.

        Differing cells are named when there are at most three (``row_names`` and ``column_names`` give
        the labels). With ``describe_gaps`` and numeric cells, the record also states how many cells differ
        only in the last printed digit and the largest absolute difference. ``note`` is appended to the
        reproduced-value text.
        """
        names = row_names or [f"row {r + 1}" for r in range(len(printed_rows))]
        columns = column_names or [f"column {c + 1}" for c in range(len(printed_rows[0]))]
        cells = [(names[r], columns[c], g, p) for r, (g_row, p_row) in enumerate(zip(generated_rows, printed_rows))
                 for c, (g, p) in enumerate(zip(g_row, p_row))]
        differing = [cell for cell in cells if cell[2] != cell[3]]
        text = f"{len(cells) - len(differing)} of {len(cells)} cells equal the printed value"
        gaps = [(abs(_as_number(g) - _as_number(p)), _last_digit_step(p)) for _, _, g, p in differing
                if _as_number(g) is not None and _as_number(p) is not None]
        if describe_gaps and differing and len(gaps) == len(differing):
            last_digit = sum(gap <= 1.5 * step for gap, step in gaps)
            text += f"; {last_digit} of the {len(differing)} differing cells differ only in the last printed digit"
            text += f"; largest absolute difference {max(gap for gap, _ in gaps):.2f}"
        if 0 < len(differing) <= 3:
            text += "; differing: " + "; ".join(f"{name}, {column}: printed '{p}', reproduced '{g}'" for name, column, g, p in differing)
        self._add(table_name, rows_label, f"{len(cells)} cells", text + note,
                  "generated table compared with common/paper_values.py")

    def write(self, output_dir):
        """Write reproduced_values.md and reproduced_values.csv into ``output_dir``; returns the number of records."""
        rows = [[r["location"], r["statement"], r["paper"], r["reproduced"], r["how"]] for r in self.records]
        write_csv(output_dir / "reproduced_values.csv", self.HEADER, rows)
        lines = [f"# {self.title}", "", "One row per number stated in the paper; values are recomputed from `results/`.", "",
                 "| " + " | ".join(self.HEADER) + " |", "|" + "---|" * len(self.HEADER)]
        lines += ["| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |" for row in rows]
        write_text(output_dir / "reproduced_values.md", "\n".join(lines) + "\n")
        return len(rows)
