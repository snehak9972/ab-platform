"""
viz.py
------
Matplotlib visualizations for the experiment report.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

COLOR_A = "#94a3b8"
COLOR_B = "#2563eb"


def plot_completion_and_speed(df: pd.DataFrame, out_path: str,
                               group_col: str = "group",
                               outcome_col: str = "completed_onboarding",
                               time_col: str = "days_to_complete") -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    summary = df.groupby(group_col)[outcome_col].agg(["sum", "count"])
    rates, lo, hi = [], [], []
    for _, row in summary.iterrows():
        ci = proportion_confint(row["sum"], row["count"], method="wilson")
        rate = row["sum"] / row["count"]
        rates.append(rate)
        lo.append(rate - ci[0])
        hi.append(ci[1] - rate)

    labels = ["A: Control", "B: Treatment"]
    axes[0].bar(labels, rates, yerr=[lo, hi], capsize=6, color=[COLOR_A, COLOR_B])
    axes[0].set_ylabel("Onboarding completion rate")
    axes[0].set_title("Primary metric: completion rate (95% CI)")
    for i, r in enumerate(rates):
        axes[0].text(i, r + hi[i] + 0.008, f"{r:.1%}", ha="center", fontweight="bold")
    axes[0].set_ylim(0, max(rates) + max(hi) + 0.06)

    a = df[(df[group_col] == "A") & (df[outcome_col] == 1)][time_col]
    b = df[(df[group_col] == "B") & (df[outcome_col] == 1)][time_col]
    axes[1].hist(a, bins=30, alpha=0.6, label="A: Control", color=COLOR_A, density=True)
    axes[1].hist(b, bins=30, alpha=0.6, label="B: Treatment", color=COLOR_B, density=True)
    axes[1].set_xlabel("Days to complete onboarding")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Secondary metric: speed of activation")
    axes[1].legend()

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_bayesian_posteriors(posterior_a, posterior_b, out_path: str,
                              n_points: int = 500) -> str:
    from scipy.stats import beta as beta_dist

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.linspace(0.28, 0.44, n_points)
    ax.plot(x, beta_dist.pdf(x, posterior_a["alpha"], posterior_a["beta"]),
            color=COLOR_A, label="A: Control posterior")
    ax.fill_between(x, beta_dist.pdf(x, posterior_a["alpha"], posterior_a["beta"]),
                     alpha=0.3, color=COLOR_A)
    ax.plot(x, beta_dist.pdf(x, posterior_b["alpha"], posterior_b["beta"]),
            color=COLOR_B, label="B: Treatment posterior")
    ax.fill_between(x, beta_dist.pdf(x, posterior_b["alpha"], posterior_b["beta"]),
                     alpha=0.3, color=COLOR_B)
    ax.set_xlabel("Completion rate")
    ax.set_ylabel("Posterior density")
    ax.set_title("Bayesian posteriors: completion rate by arm")
    ax.legend()

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sequential_boundaries(schedule, out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fractions = [s["information_fraction"] for s in schedule]
    boundaries = [s["z_boundary"] for s in schedule]
    ax.plot(fractions, boundaries, marker="o", color=COLOR_B)
    ax.axhline(1.96, color=COLOR_A, linestyle="--", label="Fixed-sample z=1.96 (naive)")
    ax.set_xlabel("Information fraction (share of planned sample collected)")
    ax.set_ylabel("Required |z| to stop early")
    ax.set_title("O'Brien-Fleming stopping boundary vs. naive fixed threshold")
    ax.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
