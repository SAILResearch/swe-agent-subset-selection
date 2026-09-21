"""Determinism validation (paper Section 6.3, Finding 3.4, first test).

Independent per-seed draws of every baseline are generated for each distribution, and every non-baseline
method's per-distribution RMSE is compared with them: win rates, relative gaps and win/tie/loss counts against
the median draw. The draws use seeds offset by 2^31 from those of the baseline family, so the two sets of
draws never coincide.
"""
import numpy as np
from scipy import stats

from . import algorithms, groups, metrics, splits
from .families import STOCHASTIC_BASELINES, TWO_RUN_BASELINES, subset_key, subset_size

SEED_OFFSET = 1 << 31
TIE_TOLERANCE = 0.01   # relative to the median draw


def baselines_for(window):
    """Baselines that exist with ``window`` training runs, in stored order."""
    names = ["random", "difficulty_stratified", "consistency_stratified", "repo_stratified",
             "repo_difficulty_stratified", "repo_consistency_stratified"]
    available = set(STOCHASTIC_BASELINES + (TWO_RUN_BASELINES if window >= 2 else ()))
    return [name for name in names if name in available]


def baseline_groups(name, population, repositories, train_runs):
    """Groups of one baseline (None for Uniform Random)."""
    if name == "random":
        return None
    if name == "difficulty_stratified":
        return groups.difficulty_groups(population, train_runs)
    if name == "consistency_stratified":
        return groups.consistency_groups(population, train_runs)
    if name == "repo_stratified":
        return groups.repository_groups(repositories)
    labels = groups.difficulty_labels(population, train_runs) if name == "repo_difficulty_stratified" else groups.pass_counts(population, train_runs)
    return groups.crossed_groups(repositories, labels)


def per_seed_draws(population, repositories, window, subset_fractions, n_seeds, base_seed, distribution_index):
    """{subset key: {baseline: (mean RMSE over splits per seed, largest MaxErr over splits per seed)}}."""
    n = population.shape[0]
    all_splits = splits.temporal_splits(population.shape[1], window)
    names = baselines_for(window)
    rmse = {subset_key(f): {b: np.zeros(n_seeds) for b in names} for f in subset_fractions}
    maxerr = {subset_key(f): {b: np.zeros(n_seeds) for b in names} for f in subset_fractions}
    for split_index, (train_runs, test_runs) in enumerate(all_splits):
        test_outcomes = population[:, test_runs]
        rates = np.mean(test_outcomes, axis=0)
        strata = {b: baseline_groups(b, population, repositories, train_runs) for b in names}
        for fraction in subset_fractions:
            size, key = subset_size(n, fraction), subset_key(fraction)
            for name in names:
                quotas = groups.proportional_quotas(strata[name], size) if strata[name] else None
                for draw in range(n_seeds):
                    rng = np.random.default_rng(SEED_OFFSET + metrics.draw_seed(base_seed, distribution_index, split_index, draw))
                    selected = (rng.choice(n, size=size, replace=False) if strata[name] is None
                                else algorithms.stratified_random(rng, strata[name], quotas, size, n))
                    errors = metrics.run_errors(test_outcomes, selected, rates)
                    rmse[key][name][draw] += float(np.sqrt(np.mean(errors ** 2)))
                    maxerr[key][name][draw] = max(maxerr[key][name][draw], float(np.max(np.abs(errors)) * 100.0))
    for key in rmse:
        for name in names:
            rmse[key][name] /= len(all_splits)
    return rmse, maxerr


def win_rate_statistics(wins):
    """Summary of per-distribution win rates, with a t-interval and a one-sided Wilcoxon test against 0.5."""
    n, mean = len(wins), float(np.mean(wins))
    sem = float(stats.sem(wins)) if n > 1 else 0.0
    low, high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem) if n > 1 and sem > 0 else (mean, mean)
    shifted = (wins - 0.5)[(wins - 0.5) != 0]
    statistic, p_value = stats.wilcoxon(shifted, alternative="greater") if len(shifted) >= 10 else (np.nan, np.nan)
    return {"mean": round(mean, 4), "median": round(float(np.median(wins)), 4),
            "std": round(float(np.std(wins, ddof=1)), 4) if n > 1 else 0.0,
            "ci95_lo": round(float(low), 4), "ci95_hi": round(float(high), 4),
            "min": round(float(np.min(wins)), 4), "max": round(float(np.max(wins)), 4),
            "q25": round(float(np.percentile(wins, 25)), 4), "q75": round(float(np.percentile(wins, 75)), 4),
            "frac_win_majority": round(float(np.mean(wins > 0.5)), 4),
            "frac_win_supermajority": round(float(np.mean(wins > 0.8)), 4),
            "frac_lose_majority": round(float(np.mean(wins < 0.5)), 4),
            "wilcoxon_stat": None if np.isnan(statistic) else round(float(statistic), 2),
            "wilcoxon_pval": None if np.isnan(p_value) else float(p_value)}


def per_level_statistics(wins, levels):
    """Win-rate summary per difficulty level (levels with at least two distributions)."""
    levels, result = np.array(levels), {}
    for level in sorted(set(levels.tolist())):
        chosen = wins[levels == level]
        if len(chosen) >= 2:
            result[int(level)] = {"n": int(len(chosen)), "mean": round(float(np.mean(chosen)), 4),
                                  "median": round(float(np.median(chosen)), 4), "std": round(float(np.std(chosen, ddof=1)), 4),
                                  "frac_win_majority": round(float(np.mean(chosen > 0.5)), 4)}
    return result


def exact_binomial_interval(successes, n, alpha=0.05):
    """Clopper-Pearson interval."""
    if n == 0:
        return 0.0, 0.0
    low, high = stats.beta.ppf([alpha / 2, 1 - alpha / 2], [max(successes, 0.5), successes + 1], [n - successes + 1, max(n - successes, 0.5)])
    return float(low), float(high)


def compare_with_draws(method_rmse, draws, levels):
    """Comparison of one method's per-distribution RMSE with the per-seed draws of one baseline."""
    n = len(method_rmse)
    column = method_rmse[:, None]
    wins, ties = np.mean(column < draws, axis=1), np.mean(column == draws, axis=1)
    median_draw = np.median(draws, axis=1)
    tolerance = TIE_TOLERANCE * median_draw
    n_win, n_loss = int(np.sum(method_rmse < median_draw - tolerance)), int(np.sum(method_rmse > median_draw + tolerance))
    low, high = exact_binomial_interval(n_win, n)
    gaps = np.zeros(n)
    positive = median_draw > 0
    gaps[positive] = (median_draw[positive] - method_rmse[positive]) / median_draw[positive] * 100
    gap_mean = float(np.mean(gaps))
    gap_sem = float(stats.sem(gaps)) if n > 1 else 0.0
    gap_interval = stats.t.interval(0.95, df=n - 1, loc=gap_mean, scale=gap_sem) if n > 1 and gap_sem > 0 else (gap_mean, gap_mean)
    return {"win_rate": round(float(np.mean(column < draws)) * 100, 1),
            "percentile": round(float(np.mean(wins + 0.5 * ties)) * 100, 1),
            "emb_rmse": round(float(np.mean(method_rmse)), 6), "bl_median_rmse": round(float(np.median(draws)), 6),
            "gap_pct": round(gap_mean, 1), "gap_median_pct": round(float(np.median(gaps)), 1),
            "gap_ci95": [round(float(gap_interval[0]), 1), round(float(gap_interval[1]), 1)],
            "catastrophic_rate": round(float(np.mean(draws > 2 * column)) * 100, 1),
            "dist_wtl": {"win": n_win, "tie": n - n_win - n_loss, "loss": n_loss, "n": n,
                         "win_frac": round(n_win / n, 4), "win_ci95": [round(low, 4), round(high, 4)]},
            "per_dist": win_rate_statistics(wins), "per_bucket": per_level_statistics(wins, levels)}
