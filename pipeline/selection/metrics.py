"""Error metrics of a subset on the test runs of a temporal split (paper Section 5.3)."""
import numpy as np


def run_errors(test_outcomes, selected, population_rates):
    """Signed difference between the subset's and the population's resolve rate, per test run."""
    return np.mean(test_outcomes[selected], axis=0) - population_rates


def summarize(errors):
    """(RMSE, maximum absolute error in percentage points, 95th-percentile absolute error in points)."""
    absolute = np.abs(errors) * 100.0
    return float(np.sqrt(np.mean(errors ** 2))), float(np.max(absolute)), float(np.percentile(absolute, 95))


def evaluate_deterministic(test_outcomes, selected, population_rates):
    """Split-level record of a deterministic selection."""
    errors = run_errors(test_outcomes, selected, population_rates)
    rmse, maxerr, p95 = summarize(errors)
    return {"rmse": rmse, "maxerr": maxerr, "maxerr_mean": maxerr, "p95": p95,
            "run_errors": errors.tolist(), "subset": np.asarray(selected).tolist()}


def draw_seed(base_seed, distribution_index, split_index, draw):
    """Seed of one random draw: unique per (distribution, temporal split, draw)."""
    return base_seed + distribution_index * 100000 + split_index * 1000 + draw


def evaluate_stochastic(select, n_seeds, test_outcomes, population_rates, base_seed, split_index, distribution_index):
    """Split-level record of a stochastic method over ``n_seeds`` independent draws.

    RMSE and P95 are averaged over the draws; ``maxerr`` is the largest and ``maxerr_mean`` the mean MaxErr
    over the draws. The stored subset is the draw whose RMSE is closest to the mean RMSE.
    """
    rmses, maxerrs, p95s, subsets, errors_per_draw = [], [], [], [], []
    for draw in range(n_seeds):
        rng = np.random.default_rng(draw_seed(base_seed, distribution_index, split_index, draw))
        selected = select(rng)
        errors = run_errors(test_outcomes, selected, population_rates)
        rmse, maxerr, p95 = summarize(errors)
        rmses.append(rmse), maxerrs.append(maxerr), p95s.append(p95)
        subsets.append(selected.tolist()), errors_per_draw.append(errors.tolist())
    typical = int(np.argmin(np.abs(np.array(rmses) - np.mean(rmses))))
    return {"rmse": float(np.mean(rmses)), "maxerr": float(np.max(maxerrs)), "maxerr_mean": float(np.mean(maxerrs)),
            "p95": float(np.mean(p95s)), "run_errors": errors_per_draw[typical], "subset": subsets[typical]}
