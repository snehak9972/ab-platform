from ab_platform import diagnostics, simulate
from ab_platform.config import ExperimentConfig


def test_srm_not_detected_on_healthy_split(sim_df):
    result = diagnostics.check_srm(sim_df)
    assert result["srm_detected"] is False


def test_srm_detected_on_injected_mismatch():
    cfg = ExperimentConfig(n_per_group=3000, random_seed=1)
    df = simulate.generate(cfg, inject_srm=True)
    result = diagnostics.check_srm(df)
    assert result["srm_detected"] is True


def test_no_missing_data_in_clean_sim(sim_df):
    # completed_onboarding, group, channel etc should never be missing;
    # days_to_complete is expected to be missing for non-completers, so
    # exclude it from this check.
    cols = [c for c in sim_df.columns if c != "days_to_complete"]
    result = diagnostics.check_missing_data(sim_df, required_cols=cols)
    assert result["any_missing"] is False


def test_no_duplicate_user_ids(sim_df):
    result = diagnostics.check_duplicates(sim_df)
    assert result["has_duplicates"] is False


def test_covariate_balance_ok_on_healthy_randomization(sim_df):
    result = diagnostics.check_covariate_balance(sim_df, "pre_experiment_engagement_score")
    assert result["balanced"] is True


def test_run_all_checks_returns_all_keys(sim_df):
    result = diagnostics.run_all_checks(sim_df)
    assert {"srm", "missing_data", "duplicates", "covariate_balance"} <= set(result.keys())
