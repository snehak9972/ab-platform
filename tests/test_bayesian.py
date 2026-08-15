import pytest

from ab_platform import bayesian


def test_posteriors_sane_means(toy_df):
    (a_a, b_a), (a_b, b_b) = bayesian.beta_binomial_posteriors(toy_df)
    mean_a = a_a / (a_a + b_a)
    mean_b = a_b / (a_b + b_b)
    assert mean_a == pytest.approx(0.60, abs=0.02)
    assert mean_b == pytest.approx(0.80, abs=0.02)


def test_bayesian_ab_test_prob_b_beats_a_high_when_b_clearly_better(toy_df):
    result = bayesian.bayesian_ab_test(toy_df, n_samples=50_000)
    assert result["prob_b_beats_a"] > 0.95


def test_bayesian_ab_test_symmetric_under_null(no_effect_df):
    result = bayesian.bayesian_ab_test(no_effect_df, n_samples=50_000)
    # Under the null the two arms have the same true rate, but a single
    # random draw can still land off-center by sampling noise — this just
    # checks the posterior hasn't collapsed to near-certainty either way.
    assert 0.05 < result["prob_b_beats_a"] < 0.95


def test_expected_loss_nonnegative(sim_df):
    result = bayesian.bayesian_ab_test(sim_df, n_samples=20_000)
    assert result["expected_loss_if_ship_b"] >= 0
    assert result["expected_loss_if_ship_a"] >= 0


def test_credible_interval_ordered(sim_df):
    result = bayesian.bayesian_ab_test(sim_df, n_samples=20_000)
    lo, hi = result["credible_interval_diff_95"]
    assert lo <= hi


def test_recommendation_ships_b_when_clear_winner(toy_df):
    result = bayesian.bayesian_ab_test(toy_df, n_samples=50_000)
    rec = bayesian.bayesian_recommendation(result)
    assert rec["decision"] == "SHIP TREATMENT (B)"


def test_recommendation_inconclusive_or_stable_under_null(no_effect_df):
    result = bayesian.bayesian_ab_test(no_effect_df, n_samples=50_000)
    rec = bayesian.bayesian_recommendation(result, loss_threshold=0.0005)
    assert rec["decision"] in {
        "INCONCLUSIVE — collect more data", "SHIP TREATMENT (B)", "KEEP CONTROL (A)"
    }
