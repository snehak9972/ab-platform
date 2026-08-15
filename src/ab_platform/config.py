"""
config.py
---------
Single source of truth for experiment parameters. In a real company this
would be a YAML/JSON config checked into the experiment repo, or pulled
from an experimentation platform (Statsig, GrowthBook, Optimizely, etc.)
Kept as a dataclass here so it's typed, importable, and testable.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ExperimentConfig:
    # --- Experiment identity ---
    experiment_name: str = "onboarding_deposit_reminder_v1"
    alpha: float = 0.05
    power: float = 0.80

    # --- Simulation ground truth (unknown to the "analyst") ---
    n_per_group: int = 5000
    base_rate_a: float = 0.34
    base_rate_b: float = 0.385
    random_seed: int = 42

    # --- Arm definitions ---
    deposit_a: int = 250
    deposit_b: int = 100
    cadence_a: str = "weekly_x1"
    cadence_b: str = "daily_x3"

    # --- Confounders used in simulation ---
    channels: tuple = ("paid_search", "organic", "referral", "social")
    channel_probs: tuple = (0.35, 0.30, 0.15, 0.20)
    channel_effect: Dict[str, float] = field(default_factory=lambda: {
        "paid_search": 0.00, "organic": 0.02, "referral": 0.05, "social": -0.03
    })
    devices: tuple = ("mobile", "desktop")
    device_probs: tuple = (0.65, 0.35)
    device_effect: Dict[str, float] = field(default_factory=lambda: {
        "mobile": -0.015, "desktop": 0.015
    })

    # --- Sequential testing / peeking guard ---
    interim_looks: int = 4          # how many times analysts might "peek"
    obf_spending_function: str = "obrien_fleming"

    # --- Paths ---
    db_path: str = "data/onboarding_ab.db"
    csv_path: str = "data/onboarding_events.csv"
    reports_dir: str = "reports"


DEFAULT_CONFIG = ExperimentConfig()
