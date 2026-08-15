import numpy as np
import pytest

from ab_platform import frequentist


def test_two_proportion_ztest_known_values(toy_df):
    # A: 60/100 = 0.60, B: 80/100 = 0.80
    result = frequentist.two_proportion_ztest(toy_df)
    assert result["n_a"] == 100
    assert result["n_b"] == 100
    assert result["rate_a"] == pytest.approx(0.60, abs=1e-9)
    assert result["rate_b"] == pytest.approx(0.80, abs=1e-9)
    assert result["abs_lift_pp"] == pytest.approx(20.0, abs=1e-9)
    assert result["significant"] is True
    assert result["p_value"] < 0.01


def test_ztest_detects_no_effect_under_null(no_effect_df):
    result = frequentist.two_proportion_ztest(no_effect_df, alpha=0.05)
    # Not guaranteed every run, but with equal true rates and n=3000/arm,
    # should almost always be non-significant. This seed is fixed upstream.
    assert result["p_value"] > 0.01


def test_ci_contains_point_estimate(sim_df):
    result = frequentist.two_proportion_ztest(sim_df)
    assert result["ci_a"][0] <= result["rate_a"] <= result["ci_a"][1]
    assert result["ci_b"][0] <= result["rate_b"] <= result["ci_b"][1]


def test_chi_square_agrees_with_ztest_direction(sim_df):
    z_result = frequentist.two_proportion_ztest(sim_df)
    chi_result = frequentist.chi_square_independence(sim_df)
    # For a 2x2 table, chi-square p-value should equal z-test p-value (both two-sided)
    assert chi_result["p_value"] == pytest.approx(z_result["p_value"], abs=1e-6)


def test_welch_ttest_runs_on_completers_only(sim_df):
    result = frequentist.welch_ttest(sim_df)
    n_completers = int(sim_df["completed_onboarding"].sum())
    assert result["n_a"] + result["n_b"] <= n_completers


def test_mann_whitney_u_returns_valid_pvalue(sim_df):
    result = frequentist.mann_whitney_u(sim_df)
    assert 0 <= result["p_value"] <= 1


def test_segment_check_returns_dataframe_with_expected_columns(sim_df):
    seg = frequentist.segment_check(sim_df, "signup_channel")
    assert {"signup_channel", "n_a", "n_b", "rate_a", "rate_b", "lift_pp", "p_value"} <= set(seg.columns)
    assert len(seg) > 0


def test_segment_check_respects_min_n(sim_df):
    seg = frequentist.segment_check(sim_df, "signup_channel", min_n=1_000_000)
    assert len(seg) == 0  # no segment should have 1M rows


def test_power_and_mde_sane_ranges(sim_df):
    result = frequentist.power_and_mde(n_per_arm=2000, rate_a=0.34, rate_b=0.385)
    assert 0 <= result["achieved_power"] <= 1
    assert not np.isnan(result["min_detectable_lift_pp"])


def test_required_sample_size_decreases_as_lift_increases():
    n_small_lift = frequentist.required_sample_size(0.34, 1)
    n_large_lift = frequentist.required_sample_size(0.34, 10)
    assert n_small_lift > n_large_lift


def test_sample_size_table_shape():
    table = frequentist.sample_size_table(0.34, lifts_pp=[2, 5])
    assert len(table) == 2
    assert "required_n_per_arm" in table.columns


def test_multiple_testing_correction_widens_or_matches_pvalues():
    p_values = [0.01, 0.03, 0.20, 0.04]
    corrected = frequentist.multiple_testing_correction(p_values)
    assert (corrected["p_adjusted"] >= corrected["p_value"] - 1e-9).all()
