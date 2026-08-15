"""
report.py
---------
Orchestrates the full analysis pipeline end-to-end and produces both a
machine-readable summary (dict / JSON) and a human-readable console
report. This is the module the CLI (`__main__.py`) calls.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from ab_platform.config import ExperimentConfig, DEFAULT_CONFIG
from ab_platform import frequentist, bayesian, cuped, diagnostics, sequential, viz

logger = logging.getLogger("ab_platform")


def load_data(cfg: ExperimentConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    path = Path(cfg.csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No data at {path}. Run `python -m ab_platform simulate` first."
        )
    return pd.read_csv(path, parse_dates=["signup_date"])


def build_recommendation(primary: dict, segments_ok: bool, power: dict) -> dict:
    if not primary["significant"]:
        decision = "DO NOT SHIP — inconclusive"
        rationale = (
            f"p = {primary['p_value']:.4f} >= {primary['alpha']}. Cannot reject the "
            f"null of no difference. Achieved power was "
            f"{power['achieved_power']:.2f}; if that's below ~0.8, extend the test "
            f"rather than concluding there's truly no effect."
        )
    elif primary["abs_lift_pp"] > 0:
        decision = "SHIP TREATMENT (B)"
        rationale = (
            f"Treatment (B) improves completion rate by {primary['abs_lift_pp']:.2f}pp "
            f"({primary['relative_lift_pct']:.1f}% relative), 95% CI "
            f"[{primary['ci_diff_pp'][0]:.2f}, {primary['ci_diff_pp'][1]:.2f}]pp, "
            f"p = {primary['p_value']:.5f}. Segment direction consistency: "
            f"{'all favor B' if segments_ok else 'MIXED — investigate'}."
        )
    else:
        decision = "KEEP CONTROL (A)"
        rationale = (
            f"Treatment (B) is significantly worse by {abs(primary['abs_lift_pp']):.2f}pp, "
            f"p = {primary['p_value']:.5f}. Do not ship."
        )
    return {"decision": decision, "rationale": rationale}


def run_full_analysis(cfg: ExperimentConfig = DEFAULT_CONFIG,
                       make_plots: bool = True,
                       save_json: bool = True) -> dict:
    df = load_data(cfg)
    results = {"experiment_name": cfg.experiment_name, "n_rows": len(df)}

    # 1. Data quality gates — run first; a failure here should make you
    #    distrust everything downstream.
    logger.info("Running diagnostics...")
    diag = diagnostics.run_all_checks(df)
    results["diagnostics"] = diag
    if diag["srm"]["srm_detected"]:
        logger.warning("SRM DETECTED — results below may not be trustworthy.")

    # 2. Frequentist primary + secondary metrics
    logger.info("Running frequentist tests...")
    primary = frequentist.two_proportion_ztest(df, alpha=cfg.alpha)
    chi_sq = frequentist.chi_square_independence(df)
    secondary_t = frequentist.welch_ttest(df, alpha=cfg.alpha)
    secondary_mw = frequentist.mann_whitney_u(df, alpha=cfg.alpha)
    results["frequentist"] = {
        "primary_completion_rate": primary,
        "chi_square_robustness": chi_sq,
        "secondary_days_to_complete_ttest": secondary_t,
        "secondary_days_to_complete_mannwhitney": secondary_mw,
    }

    # 3. Segment checks + multiple testing correction
    logger.info("Running segment checks...")
    seg_channel = frequentist.segment_check(df, "signup_channel")
    seg_device = frequentist.segment_check(df, "device_type")
    all_p = seg_channel["p_value"].tolist() + seg_device["p_value"].tolist()
    mtc = frequentist.multiple_testing_correction(all_p) if all_p else None
    segments_ok = bool((seg_channel["lift_pp"] > 0).all() and (seg_device["lift_pp"] > 0).all())
    results["segments"] = {
        "by_channel": seg_channel.to_dict(orient="records"),
        "by_device": seg_device.to_dict(orient="records"),
        "multiple_testing_correction": mtc.to_dict(orient="records") if mtc is not None else None,
        "all_segments_favor_treatment": segments_ok,
    }

    # 4. Power / MDE
    power = frequentist.power_and_mde(primary["n_a"], primary["rate_a"], primary["rate_b"],
                                       alpha=cfg.alpha, target_power=cfg.power)
    results["power"] = power

    # 5. CUPED variance reduction
    logger.info("Running CUPED...")
    cuped_result = cuped.cuped_ttest(df, alpha=cfg.alpha)
    results["cuped"] = cuped_result

    # 6. Bayesian analysis
    logger.info("Running Bayesian analysis...")
    bayes_result = bayesian.bayesian_ab_test(df, random_seed=cfg.random_seed)
    bayes_rec = bayesian.bayesian_recommendation(bayes_result)
    results["bayesian"] = {**bayes_result, "recommendation": bayes_rec}

    # 7. Sequential testing illustration
    logger.info("Building sequential testing schedule...")
    schedule = sequential.spending_schedule(cfg.alpha, cfg.interim_looks)
    final_look = sequential.evaluate_interim_look(
        primary["z_stat"], cfg.alpha, information_fraction=1.0
    )
    results["sequential"] = {"spending_schedule": schedule, "final_look_check": final_look}

    # 8. Final recommendation (frequentist-driven, Bayesian shown alongside)
    results["recommendation"] = build_recommendation(primary, segments_ok, power)

    # 9. Plots
    if make_plots:
        logger.info("Generating plots...")
        reports_dir = Path(cfg.reports_dir)
        results["plots"] = {
            "completion_and_speed": viz.plot_completion_and_speed(
                df, str(reports_dir / "completion_and_speed.png")),
            "bayesian_posteriors": viz.plot_bayesian_posteriors(
                bayes_result["posterior_a"], bayes_result["posterior_b"],
                str(reports_dir / "bayesian_posteriors.png")),
            "sequential_boundaries": viz.plot_sequential_boundaries(
                schedule, str(reports_dir / "sequential_boundaries.png")),
        }

    if save_json:
        out_path = Path(cfg.reports_dir) / "results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(_json_safe(results), f, indent=2, default=str)
        results["json_path"] = str(out_path)

    return results


def _json_safe(obj):
    """Recursively convert tuples/numpy types to plain JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def print_console_report(results: dict, cfg: ExperimentConfig = DEFAULT_CONFIG) -> None:
    p = results["frequentist"]["primary_completion_rate"]
    chi = results["frequentist"]["chi_square_robustness"]
    sec = results["frequentist"]["secondary_days_to_complete_ttest"]
    srm = results["diagnostics"]["srm"]
    cov = results["diagnostics"]["covariate_balance"]
    cu = results["cuped"]
    bay = results["bayesian"]
    rec = results["recommendation"]

    W = 78
    print("=" * W)
    print(f"A/B TEST REPORT: {cfg.experiment_name}")
    print("=" * W)
    print(f"Control (A):   ${cfg.deposit_a} deposit, {cfg.cadence_a} | n={p['n_a']}")
    print(f"Treatment (B): ${cfg.deposit_b} deposit, {cfg.cadence_b} | n={p['n_b']}")

    print("\n" + "-" * W)
    print("0. DATA QUALITY GATES")
    print("-" * W)
    print(f"  SRM check:            {srm['verdict']}")
    print(f"  Covariate balance:    {cov['verdict']}")

    print("\n" + "-" * W)
    print("1. PRIMARY METRIC — completion rate (two-proportion z-test)")
    print("-" * W)
    print(f"  A: {p['rate_a']:.4f}   B: {p['rate_b']:.4f}   "
          f"Lift: {p['abs_lift_pp']:+.2f}pp ({p['relative_lift_pct']:+.1f}%)")
    print(f"  95% CI on lift: [{p['ci_diff_pp'][0]:+.2f}, {p['ci_diff_pp'][1]:+.2f}]pp")
    print(f"  z = {p['z_stat']:.4f}, p = {p['p_value']:.6f}  "
          f"({'SIGNIFICANT' if p['significant'] else 'not significant'})")
    print(f"  Chi-square robustness check: p = {chi['p_value']:.6f} "
          f"({'agrees' if (chi['p_value']<p['alpha'])==p['significant'] else 'DISAGREES'})")

    print("\n" + "-" * W)
    print("2. SECONDARY METRIC — days to complete (Welch's t-test)")
    print("-" * W)
    print(f"  A mean: {sec['mean_a']:.2f}d   B mean: {sec['mean_b']:.2f}d   "
          f"p = {sec['p_value']:.6f}")

    print("\n" + "-" * W)
    print("3. CUPED VARIANCE REDUCTION")
    print("-" * W)
    print(f"  Variance reduction: A={cu['variance_reduction_pct_a']:.1f}%, "
          f"B={cu['variance_reduction_pct_b']:.1f}%")
    print(f"  -> Effective sample size multiplier: "
          f"{cu['effective_sample_size_multiplier']:.2f}x "
          f"(same power achievable with ~{100/cu['effective_sample_size_multiplier']:.0f}% "
          f"of the raw sample size)")
    print(f"  CUPED-adjusted p-value: {cu['p_value']:.6f}")

    print("\n" + "-" * W)
    print("4. BAYESIAN ANALYSIS")
    print("-" * W)
    print(f"  P(B > A | data) = {bay['prob_b_beats_a']:.1%}")
    print(f"  Mean estimated lift: {bay['mean_lift']*100:+.2f}pp, 95% credible interval "
          f"[{bay['credible_interval_diff_95'][0]*100:+.2f}, "
          f"{bay['credible_interval_diff_95'][1]*100:+.2f}]pp")
    print(f"  Expected loss if ship B: {bay['expected_loss_if_ship_b']*100:.3f}pp")
    print(f"  Bayesian recommendation: {bay['recommendation']['decision']}")

    print("\n" + "-" * W)
    print("5. POWER / MDE")
    print("-" * W)
    pw = results["power"]
    print(f"  Achieved power: {pw['achieved_power']:.3f}   "
          f"MDE @ 80% power: {pw['min_detectable_lift_pp']:+.2f}pp")

    print("\n" + "=" * W)
    print(f"FINAL RECOMMENDATION (frequentist-driven): {rec['decision']}")
    print("=" * W)
    print(rec["rationale"])
    if "json_path" in results:
        print(f"\nFull machine-readable results: {results['json_path']}")
    if "plots" in results:
        print("Plots:")
        for name, path in results["plots"].items():
            print(f"  - {name}: {path}")
