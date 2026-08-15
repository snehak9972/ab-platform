import pandas as pd

from ab_platform.config import ExperimentConfig
from ab_platform import simulate


def test_generate_returns_expected_row_count(small_config):
    df = simulate.generate(small_config)
    assert len(df) == small_config.n_per_group * 2


def test_generate_has_both_groups(sim_df):
    assert set(sim_df["group"].unique()) == {"A", "B"}


def test_generate_is_deterministic(small_config):
    df1 = simulate.generate(small_config)
    df2 = simulate.generate(small_config)
    pd.testing.assert_frame_equal(df1.sort_values("user_id").reset_index(drop=True),
                                   df2.sort_values("user_id").reset_index(drop=True))


def test_different_seeds_give_different_data():
    cfg1 = ExperimentConfig(n_per_group=500, random_seed=1)
    cfg2 = ExperimentConfig(n_per_group=500, random_seed=2)
    df1 = simulate.generate(cfg1)
    df2 = simulate.generate(cfg2)
    assert not df1["completed_onboarding"].equals(df2["completed_onboarding"])


def test_no_missing_user_ids(sim_df):
    assert sim_df["user_id"].isna().sum() == 0
    assert sim_df["user_id"].duplicated().sum() == 0


def test_completion_rate_roughly_matches_ground_truth(small_config):
    df = simulate.generate(small_config)
    rate_a = df[df.group == "A"]["completed_onboarding"].mean()
    rate_b = df[df.group == "B"]["completed_onboarding"].mean()
    # With n=2000/arm, sample rate should be within ~5pp of ground truth
    assert abs(rate_a - small_config.base_rate_a) < 0.05
    assert abs(rate_b - small_config.base_rate_b) < 0.05


def test_days_to_complete_only_set_for_completers(sim_df):
    completed = sim_df[sim_df.completed_onboarding == 1]
    not_completed = sim_df[sim_df.completed_onboarding == 0]
    assert completed["days_to_complete"].isna().sum() == 0
    assert not_completed["days_to_complete"].isna().all()


def test_inject_srm_breaks_balance(small_config):
    df_normal = simulate.generate(small_config)
    df_srm = simulate.generate(small_config, inject_srm=True)
    normal_counts = df_normal["group"].value_counts()
    srm_counts = df_srm["group"].value_counts()
    assert srm_counts["A"] < normal_counts["A"]


def test_persist_writes_csv_and_db(tmp_path, small_config):
    cfg = ExperimentConfig(n_per_group=200, random_seed=1,
                            csv_path=str(tmp_path / "out.csv"),
                            db_path=str(tmp_path / "out.db"))
    df = simulate.generate(cfg)
    simulate.persist(df, cfg)
    assert (tmp_path / "out.csv").exists()
    assert (tmp_path / "out.db").exists()
