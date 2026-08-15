import pytest

from ab_platform import cuped


def test_cuped_reduces_variance_when_covariate_predictive(sim_df):
    result = cuped.cuped_ttest(sim_df)
    # Our simulated covariate is constructed to be predictive of outcome,
    # so CUPED should show a positive variance reduction.
    assert result["variance_reduction_pct_a"] > 0
    assert result["variance_reduction_pct_b"] > 0


def test_cuped_point_estimate_close_to_raw_mean(sim_df):
    result = cuped.cuped_ttest(sim_df)
    raw_a = sim_df[sim_df.group == "A"]["completed_onboarding"].mean()
    raw_b = sim_df[sim_df.group == "B"]["completed_onboarding"].mean()
    # CUPED preserves the mean in expectation (theta*mean(X-Xbar) ~ 0)
    assert result["cuped_mean_a"] == pytest.approx(raw_a, abs=0.02)
    assert result["cuped_mean_b"] == pytest.approx(raw_b, abs=0.02)


def test_effective_sample_size_multiplier_at_least_one(sim_df):
    result = cuped.cuped_ttest(sim_df)
    assert result["effective_sample_size_multiplier"] >= 1.0


def test_apply_cuped_adds_column(sim_df):
    adj_df, theta = cuped.apply_cuped(sim_df, "completed_onboarding",
                                       "pre_experiment_engagement_score")
    assert "completed_onboarding_cuped" in adj_df.columns
    assert isinstance(theta, float)
