"""
frequentist.py
--------------
Classical hypothesis-testing primitives: two-proportion z-test, chi-square
independence check, Welch's t-test, per-segment checks, and power / MDE
calculations. Each function returns a plain dict (JSON-serializable) so
results are easy to log, test, and render.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    proportion_confint,
    proportion_effectsize,
    proportions_ztest,
)

_power_analysis = NormalIndPower()


def two_proportion_ztest(df: pd.DataFrame, outcome_col: str = "completed_onboarding",
                          group_col: str = "group", alpha: float = 0.05) -> Dict:
    a = df[df[group_col] == "A"][outcome_col]
    b = df[df[group_col] == "B"][outcome_col]

    n_a, n_b = len(a), len(b)
    x_a, x_b = int(a.sum()), int(b.sum())
    rate_a, rate_b = x_a / n_a, x_b / n_b

    count = np.array([x_b, x_a])
    nobs = np.array([n_b, n_a])
    z_stat, p_value = proportions_ztest(count, nobs, alternative="two-sided")

    ci_a = proportion_confint(x_a, n_a, alpha=alpha, method="wilson")
    ci_b = proportion_confint(x_b, n_b, alpha=alpha, method="wilson")

    se_diff = np.sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)
    diff = rate_b - rate_a
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_diff = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    return {
        "metric": outcome_col,
        "n_a": n_a, "n_b": n_b,
        "x_a": x_a, "x_b": x_b,
        "rate_a": rate_a, "rate_b": rate_b,
        "ci_a": ci_a, "ci_b": ci_b,
        "abs_lift": diff,
        "abs_lift_pp": diff * 100,
        "relative_lift_pct": (diff / rate_a) * 100 if rate_a > 0 else float("nan"),
        "ci_diff": ci_diff,
        "ci_diff_pp": (ci_diff[0] * 100, ci_diff[1] * 100),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
    }


def chi_square_independence(df: pd.DataFrame, outcome_col: str = "completed_onboarding",
                             group_col: str = "group") -> Dict:
    table = pd.crosstab(df[group_col], df[outcome_col])
    chi2, p, dof, _expected = stats.chi2_contingency(table, correction=False)
    return {"chi2_stat": float(chi2), "p_value": float(p), "dof": int(dof)}


def welch_ttest(df: pd.DataFrame, value_col: str = "days_to_complete",
                 group_col: str = "group", filter_col: str = "completed_onboarding",
                 alpha: float = 0.05) -> Dict:
    sub = df[df[filter_col] == 1] if filter_col else df
    a = sub[sub[group_col] == "A"][value_col].dropna()
    b = sub[sub[group_col] == "B"][value_col].dropna()

    t_stat, p_value = stats.ttest_ind(b, a, equal_var=False)
    return {
        "metric": value_col,
        "n_a": len(a), "n_b": len(b),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "median_a": float(a.median()), "median_b": float(b.median()),
        "std_a": float(a.std()), "std_b": float(b.std()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
    }


def mann_whitney_u(df: pd.DataFrame, value_col: str = "days_to_complete",
                    group_col: str = "group", filter_col: str = "completed_onboarding",
                    alpha: float = 0.05) -> Dict:
    """Non-parametric alternative to Welch's t-test — robust to the right
    skew typical of 'time to complete' data, doesn't assume normality."""
    sub = df[df[filter_col] == 1] if filter_col else df
    a = sub[sub[group_col] == "A"][value_col].dropna()
    b = sub[sub[group_col] == "B"][value_col].dropna()

    u_stat, p_value = stats.mannwhitneyu(b, a, alternative="two-sided")
    return {
        "metric": value_col, "test": "mann_whitney_u",
        "u_stat": float(u_stat), "p_value": float(p_value),
        "alpha": alpha, "significant": bool(p_value < alpha),
    }


def segment_check(df: pd.DataFrame, segment_col: str,
                   outcome_col: str = "completed_onboarding",
                   group_col: str = "group", min_n: int = 30) -> pd.DataFrame:
    rows = []
    for seg_val, sub in df.groupby(segment_col):
        a = sub[sub[group_col] == "A"][outcome_col]
        b = sub[sub[group_col] == "B"][outcome_col]
        if len(a) < min_n or len(b) < min_n:
            continue
        count = np.array([int(b.sum()), int(a.sum())])
        nobs = np.array([len(b), len(a)])
        _, p = proportions_ztest(count, nobs, alternative="two-sided")
        rows.append({
            segment_col: seg_val,
            "n_a": len(a), "n_b": len(b),
            "rate_a": a.mean(), "rate_b": b.mean(),
            "lift_pp": (b.mean() - a.mean()) * 100,
            "p_value": p,
        })
    cols = [segment_col, "n_a", "n_b", "rate_a", "rate_b", "lift_pp", "p_value"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("lift_pp", ascending=False).reset_index(drop=True)


def power_and_mde(n_per_arm: int, rate_a: float, rate_b: float,
                   alpha: float = 0.05, target_power: float = 0.80) -> Dict:
    effect_size = proportion_effectsize(rate_b, rate_a)
    achieved_power = _power_analysis.power(effect_size=effect_size, nobs1=n_per_arm,
                                            alpha=alpha, ratio=1.0)

    mde_effect_size = _power_analysis.solve_power(nobs1=n_per_arm, alpha=alpha,
                                                    power=target_power, ratio=1.0)

    def h_of_p2(p2):
        return proportion_effectsize(p2, rate_a) - mde_effect_size

    try:
        p2_mde = brentq(h_of_p2, 1e-4, 1 - 1e-4)
        mde_pp = (p2_mde - rate_a) * 100
    except Exception:
        mde_pp = float("nan")

    return {
        "n_per_arm": n_per_arm,
        "achieved_power": float(achieved_power),
        "min_detectable_lift_pp": float(mde_pp),
        "alpha": alpha,
        "target_power": target_power,
    }


def required_sample_size(baseline_rate: float, min_detectable_lift_pp: float,
                          alpha: float = 0.05, power: float = 0.80) -> int:
    """Pre-experiment: sample size needed per arm to detect a given lift."""
    target_rate = baseline_rate + min_detectable_lift_pp / 100
    effect_size = proportion_effectsize(target_rate, baseline_rate)
    n = _power_analysis.solve_power(effect_size=effect_size, alpha=alpha,
                                     power=power, ratio=1.0)
    return int(np.ceil(n))


def sample_size_table(baseline_rate: float, lifts_pp: List[float] = None,
                       alpha: float = 0.05, power: float = 0.80) -> pd.DataFrame:
    lifts_pp = lifts_pp or [1, 2, 3, 4, 5, 7, 10]
    rows = [
        {"min_lift_pp": lift,
         "required_n_per_arm": required_sample_size(baseline_rate, lift, alpha, power)}
        for lift in lifts_pp
    ]
    df = pd.DataFrame(rows)
    df["total_sample"] = df["required_n_per_arm"] * 2
    return df


def multiple_testing_correction(p_values: List[float], method: str = "holm",
                                 alpha: float = 0.05) -> pd.DataFrame:
    """Correct for testing multiple segments/metrics at once (family-wise
    error rate control). Without this, checking 6 segments at alpha=0.05
    gives a ~26% chance of a false positive somewhere by chance alone."""
    from statsmodels.stats.multitest import multipletests
    reject, p_adj, _, _ = multipletests(p_values, alpha=alpha, method=method)
    return pd.DataFrame({
        "p_value": p_values,
        "p_adjusted": p_adj,
        "significant_after_correction": reject,
    })
