"""
Command-line interface.

Usage:
    python -m ab_platform simulate [--inject-srm]
    python -m ab_platform analyze [--no-plots]
    python -m ab_platform sample-size --baseline 0.34
"""

import argparse
import logging
import sys

from ab_platform.config import DEFAULT_CONFIG
from ab_platform import simulate as simulate_mod
from ab_platform import report as report_mod
from ab_platform import frequentist


def main():
    parser = argparse.ArgumentParser(prog="ab_platform",
                                      description="A/B testing toolkit for onboarding experiments.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="Generate the simulated experiment dataset.")
    p_sim.add_argument("--inject-srm", action="store_true",
                        help="Deliberately break the group split, to test SRM detection.")

    p_ana = sub.add_parser("analyze", help="Run the full analysis pipeline and print a report.")
    p_ana.add_argument("--no-plots", action="store_true", help="Skip generating chart PNGs.")

    p_ss = sub.add_parser("sample-size", help="Print a pre-experiment sample size table.")
    p_ss.add_argument("--baseline", type=float, default=0.34, help="Baseline completion rate.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                         format="%(levelname)s: %(message)s")

    if args.command == "simulate":
        df = simulate_mod.run(DEFAULT_CONFIG, inject_srm=args.inject_srm)
        print(f"Generated {len(df):,} rows -> {DEFAULT_CONFIG.csv_path}, {DEFAULT_CONFIG.db_path}")
        print(df.groupby("group")["completed_onboarding"].agg(["count", "mean"]))

    elif args.command == "analyze":
        results = report_mod.run_full_analysis(DEFAULT_CONFIG, make_plots=not args.no_plots)
        report_mod.print_console_report(results, DEFAULT_CONFIG)

    elif args.command == "sample-size":
        table = frequentist.sample_size_table(args.baseline)
        print(f"Baseline completion rate: {args.baseline:.0%}\n")
        print(table.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
