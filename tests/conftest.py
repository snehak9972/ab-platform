import pytest
import numpy as np
import pandas as pd

from ab_platform.config import ExperimentConfig
from ab_platform import simulate


@pytest.fixture(scope="session")
def small_config():
    return ExperimentConfig(n_per_group=2000, random_seed=7,
                             csv_path="/tmp/ab_platform_test/data.csv",
                             db_path="/tmp/ab_platform_test/data.db",
                             reports_dir="/tmp/ab_platform_test/reports")


@pytest.fixture(scope="session")
def sim_df(small_config):
    return simulate.generate(small_config)


@pytest.fixture
def no_effect_df(small_config):
    """A dataset where A and B have identical true rates — used to test
    that the framework correctly does NOT find significance under the null."""
    cfg = ExperimentConfig(n_per_group=3000, base_rate_a=0.30, base_rate_b=0.30,
                            random_seed=123)
    return simulate.generate(cfg)


@pytest.fixture
def toy_df():
    """A tiny, hand-checkable dataset for exact-value assertions."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "user_id": [f"U{i}" for i in range(n)],
        "group": ["A"] * (n // 2) + ["B"] * (n // 2),
        "signup_channel": rng.choice(["organic", "paid_search"], size=n),
        "device_type": rng.choice(["mobile", "desktop"], size=n),
        "pre_experiment_engagement_score": rng.uniform(0, 1, size=n),
        "completed_onboarding": [1] * 60 + [0] * 40 + [1] * 80 + [0] * 20,
        "days_to_complete": np.where(
            np.array([1] * 60 + [0] * 40 + [1] * 80 + [0] * 20) == 1,
            rng.exponential(3, size=n), np.nan,
        ),
    })
