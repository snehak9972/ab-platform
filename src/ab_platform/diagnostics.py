"""
diagnostics.py
--------------
Data-quality gates that should run BEFORE trusting any hypothesis test
result. A statistically "significant" result from a broken experiment
(bad randomization, bot traffic skewing one arm, logging bugs) is worse
than no result at all, because it's persuasive.

Includes:
  - Sample Ratio Mismatch (SRM) check — the single highest-value guardrail
    in experimentation. If the observed split isn't close to the intended
    split, something is wrong with randomization/logging and the whole
    experiment should be distrusted until fixed.
  - Missing data / null outcome check
  - Duplicate user_id check
  - Pre-experiment covariate balance check (are the arms comparable on
    variables that shouldn't have been affected by treatment?)
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def check_srm(df: pd.DataFrame, group_col: str = "group",
               expected_ratio: Dict[str, float] = None,
               alpha: float = 0.001) -> Dict:
    """
    Sample Ratio Mismatch check via chi-square goodness-of-fit test.
    Uses a strict alpha (0.001 is standard practice, e.g. per Kohavi et al.)
    because SRM checks should rarely false-alarm on healthy experiments,
    but must reliably catch real mismatches.
    """
    counts = df[group_col].value_counts().sort_index()
    groups = counts.index.tolist()
    expected_ratio = expected_ratio or {g: 1 / len(groups) for g in groups}

    observed = counts.values
    total = observed.sum()
    expected = np.array([expected_ratio[g] * total for g in groups])

    chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    return {
        "observed_counts": dict(zip(groups, observed.tolist())),
        "expected_counts": dict(zip(groups, expected.tolist())),
        "chi2_stat": float(chi2),
        "p_value": float(p_value),
        "alpha": alpha,
        "srm_detected": bool(p_value < alpha),
        "verdict": (
            "SRM DETECTED — do not trust this experiment's results until "
            "the randomization/logging issue causing the imbalance is found."
            if p_value < alpha else
            "No sample ratio mismatch detected. Split looks healthy."
        ),
    }


def check_missing_data(df: pd.DataFrame, required_cols: List[str] = None) -> Dict:
    required_cols = required_cols or list(df.columns)
    missing = df[required_cols].isna().sum()
    missing = missing[missing > 0]
    return {
        "columns_with_missing": missing.to_dict(),
        "any_missing": bool(len(missing) > 0),
    }


def check_duplicates(df: pd.DataFrame, id_col: str = "user_id") -> Dict:
    dup_count = int(df[id_col].duplicated().sum())
    return {
        "duplicate_count": dup_count,
        "has_duplicates": dup_count > 0,
    }


def check_covariate_balance(df: pd.DataFrame, covariate_col: str,
                             group_col: str = "group", alpha: float = 0.01) -> Dict:
    """
    Checks whether a pre-experiment covariate (that treatment cannot have
    caused) is balanced across arms. Significant imbalance suggests a
    randomization bug, not a real effect (since the covariate was measured
    before assignment).

    Default alpha is stricter than 0.05 (0.01) for the same reason SRM
    checks use a strict threshold: balance checks are often run on several
    covariates, and at alpha=0.05 roughly 1 in 20 will flag by chance alone
    even under perfectly healthy randomization. A single borderline flag
    (p just under 0.05) is not on its own evidence of a broken experiment —
    check whether it replicates on other covariates before escalating.
    """
    a = df[df[group_col] == "A"][covariate_col].dropna()
    b = df[df[group_col] == "B"][covariate_col].dropna()
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    return {
        "covariate": covariate_col,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "balanced": bool(p_value >= alpha),
        "verdict": (
            "Balanced — consistent with correct randomization."
            if p_value >= alpha else
            "IMBALANCED on a pre-experiment covariate — investigate "
            "randomization before trusting outcome differences."
        ),
    }


def run_all_checks(df: pd.DataFrame, group_col: str = "group",
                    id_col: str = "user_id",
                    covariate_col: str = "pre_experiment_engagement_score") -> Dict:
    return {
        "srm": check_srm(df, group_col),
        "missing_data": check_missing_data(df),
        "duplicates": check_duplicates(df, id_col),
        "covariate_balance": check_covariate_balance(df, covariate_col, group_col),
    }
