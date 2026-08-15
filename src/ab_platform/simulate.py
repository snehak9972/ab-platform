"""
simulate.py
-----------
Simulates a two-arm onboarding experiment with:
  - a known ground-truth treatment effect (for validating the framework)
  - realistic confounders (signup channel, device type)
  - a pre-experiment covariate (`pre_experiment_engagement_score`) used
    later for CUPED variance reduction
  - a small injected Sample Ratio Mismatch (SRM) knob, off by default,
    so the SRM check in diagnostics.py has something to catch when enabled

This module has no side effects at import time — call `run()` to generate
and persist data. Keeping generation deterministic (seeded) and pure makes
it unit-testable.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from ab_platform.config import ExperimentConfig, DEFAULT_CONFIG


def _simulate_group(cfg: ExperimentConfig, rng: np.random.Generator,
                     group_label: str, n: int, base_rate: float,
                     deposit_amount: int, reminder_cadence: str) -> pd.DataFrame:
    channel = rng.choice(cfg.channels, size=n, p=cfg.channel_probs)
    device = rng.choice(cfg.devices, size=n, p=cfg.device_probs)
    signup_day_offset = rng.integers(0, 30, size=n)
    signup_date = pd.Timestamp("2026-05-01") + pd.to_timedelta(signup_day_offset, unit="D")

    # Pre-experiment covariate: e.g. an engagement/propensity score computed
    # from the user's behavior BEFORE they ever saw either onboarding flow
    # (app installs, marketing site visits, etc). Correlated with outcome,
    # unaffected by treatment by construction -> valid for CUPED.
    pre_engagement = np.clip(rng.normal(0.5, 0.15, size=n), 0, 1)

    p = np.full(n, base_rate)
    p += np.array([cfg.channel_effect[c] for c in channel])
    p += np.array([cfg.device_effect[d] for d in device])
    p += (pre_engagement - 0.5) * 0.25   # engaged users more likely to complete
    p += rng.normal(0, 0.015, size=n)
    p = np.clip(p, 0.02, 0.98)

    completed = rng.binomial(1, p)

    mean_days = 4.5 if group_label == "A" else 3.1
    time_to_complete = np.where(
        completed == 1,
        np.clip(rng.exponential(mean_days, size=n), 0.1, 30),
        np.nan,
    )

    return pd.DataFrame({
        "user_id": [f"{group_label}_{i:06d}" for i in range(n)],
        "group": group_label,
        "signup_date": signup_date,
        "signup_channel": channel,
        "device_type": device,
        "pre_experiment_engagement_score": np.round(pre_engagement, 4),
        "suggested_deposit_amount": deposit_amount,
        "reminder_cadence": reminder_cadence,
        "completed_onboarding": completed,
        "days_to_complete": np.round(time_to_complete, 2),
    })


def generate(cfg: ExperimentConfig = DEFAULT_CONFIG,
             inject_srm: bool = False) -> pd.DataFrame:
    """Generate the full simulated experiment dataset.

    Parameters
    ----------
    inject_srm: if True, deliberately drop a chunk of group A rows to
        create a Sample Ratio Mismatch, for testing diagnostics.check_srm.
    """
    rng = np.random.default_rng(cfg.random_seed)

    df_a = _simulate_group(cfg, rng, "A", cfg.n_per_group, cfg.base_rate_a,
                            cfg.deposit_a, cfg.cadence_a)
    df_b = _simulate_group(cfg, rng, "B", cfg.n_per_group, cfg.base_rate_b,
                            cfg.deposit_b, cfg.cadence_b)

    if inject_srm:
        df_a = df_a.sample(frac=0.90, random_state=cfg.random_seed)

    df = pd.concat([df_a, df_b], ignore_index=True)
    df = df.sample(frac=1, random_state=cfg.random_seed).reset_index(drop=True)
    return df


def persist(df: pd.DataFrame, cfg: ExperimentConfig = DEFAULT_CONFIG) -> None:
    """Write the dataset to CSV and SQLite, creating parent dirs as needed."""
    csv_path = Path(cfg.csv_path)
    db_path = Path(cfg.db_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)

    conn = sqlite3.connect(db_path)
    try:
        df.to_sql("onboarding_events", conn, if_exists="replace", index=False)
    finally:
        conn.close()


def run(cfg: ExperimentConfig = DEFAULT_CONFIG, inject_srm: bool = False) -> pd.DataFrame:
    df = generate(cfg, inject_srm=inject_srm)
    persist(df, cfg)
    return df


if __name__ == "__main__":
    data = run()
    print(f"Generated {len(data):,} rows")
    print(data.groupby("group")["completed_onboarding"].agg(["count", "mean"]))
