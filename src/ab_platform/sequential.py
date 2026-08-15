"""
sequential.py
-------------
Guards against the single most common way A/B tests get misused in
practice: "peeking" at results daily and stopping as soon as p < 0.05.
Doing that inflates the true false-positive rate far above the nominal
alpha (can exceed 30-40% with frequent peeking instead of 5%).

This module implements an O'Brien-Fleming-style alpha-spending approach:
instead of one fixed critical value, each interim look gets a stricter
(smaller) significance threshold, spending a total error budget of alpha
across all looks, so the cumulative false-positive rate stays at alpha
even if you check results multiple times during the experiment.

This is a simplified, dependency-light implementation (closed-form
O'Brien-Fleming boundary approximation) intended to demonstrate the
concept correctly, not a full clinical-trials-grade sequential design
library (for production use at scale, consider `sequential` in R or a
vetted experimentation platform's built-in sequential testing).
"""

from typing import Dict, List

import numpy as np
from scipy import stats


def obrien_fleming_boundary(alpha: float, information_fraction: float) -> float:
    """
    Approximate O'Brien-Fleming spending-function boundary (z-scale) at a
    given information fraction t in (0, 1] (t = fraction of the planned
    total sample size collected so far). Early looks get a very strict
    (high) z threshold; the final look's threshold approaches the
    conventional fixed-sample z critical value.
    """
    t = max(information_fraction, 1e-6)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    # Classic O'Brien-Fleming closed-form approximation
    boundary_z = z_alpha / np.sqrt(t)
    return float(boundary_z)


def spending_schedule(alpha: float, n_looks: int) -> List[Dict]:
    """Boundary (z and p) at each of n_looks equally-spaced interim looks."""
    schedule = []
    for i in range(1, n_looks + 1):
        t = i / n_looks
        z_boundary = obrien_fleming_boundary(alpha, t)
        p_boundary = 2 * (1 - stats.norm.cdf(z_boundary))
        schedule.append({
            "look": i,
            "information_fraction": round(t, 3),
            "z_boundary": round(z_boundary, 4),
            "p_value_boundary": p_boundary,
        })
    return schedule


def evaluate_interim_look(z_stat: float, alpha: float, information_fraction: float) -> Dict:
    """Check whether an observed z-statistic crosses the O'Brien-Fleming
    stopping boundary at this interim look, i.e. whether it's safe to stop
    the experiment early and declare significance without inflating the
    false-positive rate."""
    boundary = obrien_fleming_boundary(alpha, information_fraction)
    crosses = abs(z_stat) >= boundary
    return {
        "observed_z": float(z_stat),
        "boundary_z": boundary,
        "information_fraction": information_fraction,
        "crosses_boundary": bool(crosses),
        "recommendation": (
            "Safe to stop early — significant at this look's adjusted threshold."
            if crosses else
            "Do not stop yet. Naive p<0.05 at this point in the test would be an "
            "inflated false positive; continue to plan and re-check at the next look."
        ),
    }


def naive_peeking_false_positive_rate(true_effect_pp: float, n_peeks: int,
                                       n_per_arm: int, baseline_rate: float,
                                       alpha: float = 0.05, n_sims: int = 2000,
                                       seed: int = 42) -> float:
    """
    Monte Carlo demonstration: under the NULL (true_effect_pp=0), what
    fraction of simulated experiments would a naive analyst incorrectly
    call 'significant' if they peek at the p-value n_peeks times during
    the experiment and stop as soon as p < alpha at any peek?

    This is the empirical justification for why sequential correction
    matters — run with true_effect_pp=0 to see the inflated false-positive
    rate directly.
    """
    rng = np.random.default_rng(seed)
    rate_a = baseline_rate
    rate_b = baseline_rate + true_effect_pp / 100
    peek_points = np.linspace(n_per_arm // n_peeks, n_per_arm, n_peeks).astype(int)

    false_positives = 0
    for _ in range(n_sims):
        outcomes_a = rng.binomial(1, rate_a, size=n_per_arm)
        outcomes_b = rng.binomial(1, rate_b, size=n_per_arm)
        stopped_significant = False
        for k in peek_points:
            a_k, b_k = outcomes_a[:k], outcomes_b[:k]
            n_a_k, n_b_k = len(a_k), len(b_k)
            x_a_k, x_b_k = a_k.sum(), b_k.sum()
            p_a_k, p_b_k = x_a_k / n_a_k, x_b_k / n_b_k
            se = np.sqrt(p_a_k * (1 - p_a_k) / n_a_k + p_b_k * (1 - p_b_k) / n_b_k)
            if se == 0:
                continue
            z = (p_b_k - p_a_k) / se
            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
            if p_val < alpha:
                stopped_significant = True
                break
        if stopped_significant:
            false_positives += 1

    return false_positives / n_sims
