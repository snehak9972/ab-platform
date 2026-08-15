import json
from pathlib import Path

import pytest

from ab_platform.config import ExperimentConfig
from ab_platform import simulate, report


@pytest.fixture
def report_cfg(tmp_path):
    return ExperimentConfig(
        n_per_group=1500, random_seed=11,
        csv_path=str(tmp_path / "data" / "events.csv"),
        db_path=str(tmp_path / "data" / "events.db"),
        reports_dir=str(tmp_path / "reports"),
    )


def test_load_data_raises_if_missing(report_cfg):
    with pytest.raises(FileNotFoundError):
        report.load_data(report_cfg)


def test_full_pipeline_runs_end_to_end(report_cfg):
    df = simulate.run(report_cfg)
    assert len(df) == report_cfg.n_per_group * 2

    results = report.run_full_analysis(report_cfg, make_plots=True, save_json=True)

    # Top-level structure
    for key in ["diagnostics", "frequentist", "segments", "power", "cuped",
                "bayesian", "sequential", "recommendation", "plots"]:
        assert key in results

    # JSON output is valid and on disk
    json_path = Path(results["json_path"])
    assert json_path.exists()
    with open(json_path) as f:
        loaded = json.load(f)
    assert loaded["experiment_name"] == report_cfg.experiment_name

    # Plots were actually written
    for plot_path in results["plots"].values():
        assert Path(plot_path).exists()


def test_pipeline_without_plots_skips_plot_keys(report_cfg):
    simulate.run(report_cfg)
    results = report.run_full_analysis(report_cfg, make_plots=False, save_json=False)
    assert "plots" not in results
    assert "json_path" not in results


def test_recommendation_decision_is_one_of_expected_values(report_cfg):
    simulate.run(report_cfg)
    results = report.run_full_analysis(report_cfg, make_plots=False, save_json=False)
    assert results["recommendation"]["decision"] in {
        "SHIP TREATMENT (B)", "KEEP CONTROL (A)", "DO NOT SHIP — inconclusive"
    }


def test_print_console_report_does_not_raise(report_cfg, capsys):
    simulate.run(report_cfg)
    results = report.run_full_analysis(report_cfg, make_plots=False, save_json=False)
    report.print_console_report(results, report_cfg)
    captured = capsys.readouterr()
    assert "FINAL RECOMMENDATION" in captured.out
