"""
cuped.py
--------
CUPED (Controlled-experiment Using Pre-Experiment Data) — Microsoft's
widely-used variance reduction technique (Deng et al., 2013).

Idea: if we have a covariate X measured BEFORE the experiment started
(so it can't be affected by treatment) that correlates with the outcome
Y, we can subtract out the part of Y explained by X to shrink variance
without touching the point estimate of the treatment effect:

    Y_cuped = Y - theta * (X - mean(X))
    theta   = Cov(X, Y) / Var(X)      (estimated pooled across both arms)

This doesn't change the estimated lift (in expectation) but can reduce
the variance of the estimator substantially when X is a decent predictor
of Y — which shortens the required experiment duration for the same
statistical power. In this project, `pre_experiment_engagement_score`
(simulated in simulate.py) plays that role.

Note: CUPED is most natural for continuous metrics. For a binary metric
(completion rate) we apply it on the 0/1 outcome directly, which is a
standard and valid application of the technique (it's just OLS residualization,
which doesn't require normality) — the resulting z-test still works because
we're comparing means of the adjusted variable, and with these sample sizes
the CLT applies.
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


def apply_cuped(df: pd.DataFrame, outcome_col: str, covariate_col: str,
                 group_col: str = "group") -> pd.DataFrame:
    """Returns a copy of df with an added `<outcome_col>_cuped` column."""
    df = df.copy()
    x = df[covariate_col].values
    y = df[outcome_col].values

    x_mean = x.mean()
    theta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)

    df[f"{outcome_col}_cuped"] = y - theta * (x - x_mean)
    return df, theta


def cuped_ttest(df: pd.DataFrame, outcome_col: str = "completed_onboarding",
                 covariate_col: str = "pre_experiment_engagement_score",
                 group_col: str = "group", alpha: float = 0.05) -> Dict:
    adj_df, theta = apply_cuped(df, outcome_col, covariate_col, group_col)
    adj_col = f"{outcome_col}_cuped"

    a = adj_df[adj_df[group_col] == "A"][adj_col]
    b = adj_df[adj_df[group_col] == "B"][adj_col]

    # Variance comparison: this is the headline number for CUPED —
    # "how much did we shrink the variance, i.e. how much faster could
    # we have called this experiment?"
    raw_a = df[df[group_col] == "A"][outcome_col]
    raw_b = df[df[group_col] == "B"][outcome_col]
    var_reduction_a = 1 - (a.var() / raw_a.var())
    var_reduction_b = 1 - (b.var() / raw_b.var())

    t_stat, p_value = stats.ttest_ind(b, a, equal_var=False)

    return {
        "theta": float(theta),
        "raw_var_a": float(raw_a.var()), "cuped_var_a": float(a.var()),
        "raw_var_b": float(raw_b.var()), "cuped_var_b": float(b.var()),
        "variance_reduction_pct_a": float(var_reduction_a * 100),
        "variance_reduction_pct_b": float(var_reduction_b * 100),
        "cuped_mean_a": float(a.mean()), "cuped_mean_b": float(b.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
        "effective_sample_size_multiplier": float(
            1 / (1 - np.mean([var_reduction_a, var_reduction_b]))
        ) if np.mean([var_reduction_a, var_reduction_b]) < 1 else float("inf"),
    }
