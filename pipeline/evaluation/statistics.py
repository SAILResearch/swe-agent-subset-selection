"""Statistics of the evaluation (paper Section 5.4)."""
import numpy as np
from scipy import stats


def describe(values):
    """Descriptive statistics of an array."""
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return {}
    return {"mean": float(np.mean(a)), "median": float(np.median(a)), "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)), "p5": float(np.percentile(a, 5)),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)),
            "min": float(np.min(a)), "max": float(np.max(a)), "n": int(len(a))}


def paired_wilcoxon(differences):
    """Two-sided Wilcoxon signed-rank p-value of paired differences (1.0 when all differences are zero)."""
    if np.all(differences == 0):
        return 1.0
    try:
        return stats.wilcoxon(differences, alternative="two-sided")[1]
    except Exception:   # the test is undefined for degenerate inputs
        return 1.0


def paired_cliff_delta(x, y):
    """Paired dominance effect size (#(x > y) - #(x < y)) / n and its magnitude (Romano et al. 2006 thresholds)."""
    d = np.asarray(x) - np.asarray(y)
    if len(d) == 0:
        return 0.0, "negligible"
    delta = float((np.sum(d > 0) - np.sum(d < 0)) / len(d))
    size = abs(delta)
    return delta, "negligible" if size < 0.147 else "small" if size < 0.33 else "medium" if size < 0.474 else "large"


def holm_adjust(p_values):
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    m, adjusted, running = len(p_values), [None] * len(p_values), 0.0
    for rank, (position, p) in enumerate(sorted(enumerate(p_values), key=lambda item: item[1])):
        running = min(max(running, p * (m - rank)), 1.0)
        adjusted[position] = running
    return adjusted


def probability_of_superiority(x, y):
    """Paired P(x < y) + 0.5 P(x = y) and its magnitude (Vargha and Delaney 2000)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0:
        return 0.5, "negligible"
    ps = float((np.sum(x < y) + 0.5 * np.sum(x == y)) / len(x))
    distance = abs(ps - 0.5)
    return ps, "negligible" if distance < 0.06 else "small" if distance < 0.14 else "medium" if distance < 0.21 else "large"


def check_range(values, low, high, what):
    """Stop when values fall outside their scale (MaxErr in percentage points, RMSE as a fraction)."""
    if values.size == 0 or values.min() < low or values.max() > high:
        raise SystemExit(f"{what}: values outside [{low}, {high}]")


def relative_change(reference, value):
    """Change of ``value`` relative to ``reference`` in percent (negative: lower error)."""
    return (value - reference) / reference * 100.0 if reference > 0 else float("nan")
