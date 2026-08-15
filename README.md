# ab-platform

[![CI](https://github.com/YOUR_USERNAME/ab-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/ab-platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A production-shaped A/B testing toolkit, built around a simulated customer-onboarding
experiment (deposit amount + reminder cadence). It goes beyond "run a t-test" to cover
the things a real experimentation platform has to handle: data-quality gates,
frequentist **and** Bayesian inference, variance reduction, and protection against
the most common way experiments get misread — peeking at results early.

Not a toy script — a proper Python package with a CLI, tests, CI, and typed config.

## Why this exists

Most "A/B testing" portfolio projects are a single `scipy.stats.ttest_ind` call.
This one is built the way a data scientist would actually ship an internal
experimentation tool: reusable modules, a data-quality layer that runs *before*
you trust any p-value, more than one statistical lens on the same question, and
an explicit answer to "what happens if someone checks the dashboard every day
instead of waiting for the test to finish."

## What's inside

| Capability | Module | Why it's here |
|---|---|---|
| Realistic data simulation | `simulate.py` | Known ground-truth effect + confounders (channel, device) + a pre-experiment covariate, so every downstream method can be validated against a known answer. |
| Data-quality gates | `diagnostics.py` | **Sample Ratio Mismatch (SRM)** check, missing-data check, duplicate-ID check, pre-experiment covariate balance check. Run first — a "significant" result from a broken experiment is worse than no result. |
| Frequentist testing | `frequentist.py` | Two-proportion z-test with Wilson CIs, chi-square robustness cross-check, Welch's t-test + Mann-Whitney U for the secondary metric, per-segment checks, multiple-testing correction (Holm), power/MDE calculators. |
| Bayesian testing | `bayesian.py` | Beta-Binomial conjugate model, P(B > A \| data), **expected loss** decision rule (the metric that actually drives ship/no-ship in Bayesian experimentation platforms), credible intervals. |
| Variance reduction | `cuped.py` | CUPED (Deng et al., 2013) using a pre-experiment covariate — shows the effective sample-size gain, i.e. how much shorter the test could have run. |
| Sequential testing | `sequential.py` | O'Brien-Fleming alpha-spending boundaries, plus a Monte Carlo demonstration of how much naive daily peeking inflates the true false-positive rate above the nominal 5%. |
| Orchestration + CLI | `report.py`, `__main__.py` | Runs the full pipeline, saves machine-readable JSON + PNG charts, prints a stakeholder-readable console report. |
| Config | `config.py` | Single typed `ExperimentConfig` dataclass — no magic numbers scattered through the codebase. |

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/ab-platform.git
cd ab-platform
pip install -e ".[dev]"

# 1. Simulate the experiment
python -m ab_platform simulate

# 2. Run the full analysis (frequentist + Bayesian + CUPED + sequential + diagnostics)
python -m ab_platform analyze

# 3. (optional) Pre-experiment sample size planning
python -m ab_platform sample-size --baseline 0.34
```

Or use it as a library:

```python
from ab_platform import simulate, report
from ab_platform.config import DEFAULT_CONFIG

df = simulate.run(DEFAULT_CONFIG)
results = report.run_full_analysis(DEFAULT_CONFIG)
print(results["recommendation"])
```

An interactive walkthrough is in [`notebooks/analysis_walkthrough.ipynb`](notebooks/analysis_walkthrough.ipynb).

## Example output

```
------------------------------------------------------------------------------
1. PRIMARY METRIC — completion rate (two-proportion z-test)
------------------------------------------------------------------------------
  A: 0.3444   B: 0.3946   Lift: +5.02pp (+14.6%)
  95% CI on lift: [+3.13, +6.91]pp
  z = 5.2002, p = 0.000000  (SIGNIFICANT)
  Chi-square robustness check: p = 0.000000 (agrees)

------------------------------------------------------------------------------
3. CUPED VARIANCE REDUCTION
------------------------------------------------------------------------------
  Variance reduction: A=0.7%, B=0.5%
  -> Effective sample size multiplier: 1.01x

------------------------------------------------------------------------------
4. BAYESIAN ANALYSIS
------------------------------------------------------------------------------
  P(B > A | data) = 100.0%
  Expected loss if ship B: 0.000pp
  Bayesian recommendation: SHIP TREATMENT (B)

==============================================================================
FINAL RECOMMENDATION: SHIP TREATMENT (B)
==============================================================================
```

![results chart](reports/completion_and_speed.png)

## Project layout

```
ab-platform/
├── src/ab_platform/
│   ├── config.py          # typed experiment configuration
│   ├── simulate.py        # data generation
│   ├── diagnostics.py     # SRM, missing data, balance checks
│   ├── frequentist.py     # z-test, t-test, power, segments
│   ├── bayesian.py        # Beta-Binomial model, expected loss
│   ├── cuped.py           # variance reduction
│   ├── sequential.py      # alpha-spending, peeking simulation
│   ├── viz.py             # matplotlib charts
│   ├── report.py          # pipeline orchestration
│   └── __main__.py        # CLI
├── tests/                 # 48 pytest tests, one file per module
├── notebooks/              # interactive walkthrough
├── .github/workflows/ci.yml
├── pyproject.toml
└── requirements.txt
```

## Design decisions worth knowing for an interview

- **Primary metric decided before results are examined** (completion rate); time-to-complete is explicitly secondary/diagnostic, to avoid metric-shopping after the fact.
- **Wilson score intervals**, not Wald, for proportions — Wald intervals misbehave near 0/1 and at moderate sample sizes.
- **Welch's t-test** (unequal variance assumed) over Student's, plus **Mann-Whitney U** as a non-parametric cross-check since time-to-complete is right-skewed.
- **SRM check uses α = 0.001**, far stricter than the α = 0.05 used for the outcome test — an SRM check should almost never false-alarm on a healthy experiment, but must reliably catch real randomization bugs.
- **Segment checks guard against Simpson's paradox**: a topline lift that's actually reversed within a major segment, usually caused by imbalanced traffic mix rather than a genuine effect.
- **Multiple-testing correction (Holm)** applied when scanning several segments at once — checking 6 segments at α=0.05 gives a real chance of a false positive somewhere by chance alone.
- **CUPED** applied using a genuinely pre-experiment covariate (simulated engagement score), which is the actual requirement for CUPED to be valid — a covariate treatment could have influenced would bias the adjustment.
- **Bayesian decision rule is expected loss, not P(B>A) alone** — a 51% chance of a tiny win is a different decision than a 51% chance of a huge win, and expected loss is what several production Bayesian testing tools (e.g. VWO's, Optimizely's Stats Engine lineage) actually use to gate shipping.
- **Sequential/alpha-spending section exists because naive peeking is the #1 real-world way experiments get misread** — the Monte Carlo simulation in `sequential.py` quantifies the inflation directly instead of just asserting it.

## Running tests

```bash
pytest tests/ -v --cov=ab_platform --cov-report=term-missing
```

48 tests across simulation, each statistical module, diagnostics, sequential testing, and full-pipeline integration.

## Extending this

- Swap `simulate.py` for a real event-log query (the `report.py` / CLI boundary doesn't care where `onboarding_events.csv` came from).
- Add a third arm (`frequentist.py` functions currently assume A/B; extending to A/B/n means adding an ANOVA + post-hoc correction path).
- Point `config.py`'s paths at a real warehouse table instead of SQLite for production use.

## License

MIT — see [LICENSE](LICENSE).
