"""
bayesian.py
-----------
Bayesian alternative/complement to the frequentist test. Uses a
Beta-Binomial conjugate model for the completion-rate metric:

    prior:      p_A, p_B ~ Beta(alpha0, beta0)
    likelihood: x_A ~ Binomial(n_A, p_A),  x_B ~ Binomial(n_B, p_B)
    posterior:  p_A | data ~ Beta(alpha0 + x_A, beta0 + n_A - x_A)
                p_B | data ~ Beta(alpha0 + x_B, beta0 + n_B - x_B)

We report:
  - P(B > A | data)            — probability treatment is better at all
  - Expected loss of choosing B (and of choosing A) — the decision-theoretic
    quantity that matters more than P(B > A) alone: a 51% chance of a tiny
    win isn't the same decision as a 51% chance of a huge win.
  - 95% credible interval on the difference

No p-values, no fixed alpha, and — unlike the frequentist test — this
does NOT inflate false-positive rate under continuous monitoring in the
same way, which is why it's often paired with a sequential testing setup.
"""

from typing import Dict

import numpy as np
import pandas as pd


def beta_binomial_posteriors(df: pd.DataFrame, outcome_col: str = "completed_onboarding",
                              group_col: str = "group",
                              prior_alpha: float = 1.0, prior_beta: float = 1.0):
    """Returns (alpha_a, beta_a, alpha_b, beta_b) posterior parameters."""
    a = df[df[group_col] == "A"][outcome_col]
    b = df[df[group_col] == "B"][outcome_col]

    x_a, n_a = int(a.sum()), len(a)
    x_b, n_b = int(b.sum()), len(b)

    post_a = (prior_alpha + x_a, prior_beta + n_a - x_a)
    post_b = (prior_alpha + x_b, prior_beta + n_b - x_b)
    return post_a, post_b


def bayesian_ab_test(df: pd.DataFrame, outcome_col: str = "completed_onboarding",
                      group_col: str = "group", prior_alpha: float = 1.0,
                      prior_beta: float = 1.0, n_samples: int = 200_000,
                      random_seed: int = 42) -> Dict:
    rng = np.random.default_rng(random_seed)
    (a_a, b_a), (a_b, b_b) = beta_binomial_posteriors(
        df, outcome_col, group_col, prior_alpha, prior_beta
    )

    samples_a = rng.beta(a_a, b_a, size=n_samples)
    samples_b = rng.beta(a_b, b_b, size=n_samples)
    diff = samples_b - samples_a

    prob_b_better = float((diff > 0).mean())

    # Expected loss: if we ship B but A was actually better, how much
    # completion rate do we lose on average (and vice versa)?
    expected_loss_choose_b = float(np.mean(np.maximum(samples_a - samples_b, 0)))
    expected_loss_choose_a = float(np.mean(np.maximum(samples_b - samples_a, 0)))

    ci_lo, ci_hi = np.percentile(diff, [2.5, 97.5])

    return {
        "posterior_a": {"alpha": a_a, "beta": b_a,
                         "mean": a_a / (a_a + b_a)},
        "posterior_b": {"alpha": a_b, "beta": b_b,
                         "mean": a_b / (a_b + b_b)},
        "prob_b_beats_a": prob_b_better,
        "expected_loss_if_ship_b": expected_loss_choose_b,
        "expected_loss_if_ship_a": expected_loss_choose_a,
        "credible_interval_diff_95": (float(ci_lo), float(ci_hi)),
        "mean_lift": float(diff.mean()),
        "n_samples": n_samples,
    }


def bayesian_recommendation(result: Dict, loss_threshold: float = 0.0005) -> Dict:
    """
    Decision rule based on expected loss (a common practical choice, e.g.
    used by Google Analytics / VWO / Optimizely's Bayesian engines):
    ship the variant whose expected loss, if it turns out to be the worse
    arm, is below a tolerable threshold (default 0.05 percentage points).
    """
    if result["expected_loss_if_ship_b"] <= loss_threshold:
        decision = "SHIP TREATMENT (B)"
        rationale = (
            f"P(B > A) = {result['prob_b_beats_a']:.1%}. Expected loss from "
            f"shipping B, if A were actually better, is only "
            f"{result['expected_loss_if_ship_b']*100:.3f}pp — below the "
            f"{loss_threshold*100:.3f}pp tolerance. Mean estimated lift: "
            f"{result['mean_lift']*100:+.2f}pp, 95% credible interval "
            f"[{result['credible_interval_diff_95'][0]*100:+.2f}, "
            f"{result['credible_interval_diff_95'][1]*100:+.2f}]pp."
        )
    elif result["expected_loss_if_ship_a"] <= loss_threshold:
        decision = "KEEP CONTROL (A)"
        rationale = (
            f"P(B > A) = {result['prob_b_beats_a']:.1%}, meaning A is "
            f"favored. Expected loss from keeping A is "
            f"{result['expected_loss_if_ship_a']*100:.3f}pp, below tolerance."
        )
    else:
        decision = "INCONCLUSIVE — collect more data"
        rationale = (
            f"Neither arm's expected loss is below the "
            f"{loss_threshold*100:.3f}pp tolerance yet "
            f"(loss(B)={result['expected_loss_if_ship_b']*100:.3f}pp, "
            f"loss(A)={result['expected_loss_if_ship_a']*100:.3f}pp). "
            f"The posteriors haven't separated enough to act confidently."
        )
    return {"decision": decision, "rationale": rationale}
