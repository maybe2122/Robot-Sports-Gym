"""Command line interface for quantitative fidelity reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .evaluation import (
    aggregate_reports,
    load_report,
    report_markdown,
    run_mujoco_fidelity,
    write_report,
)
from .specs import Sport


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="multisport-eval",
        description="Measure MuJoCo simulation fidelity against regulation and engineering targets.",
    )
    result.add_argument(
        "--sports",
        nargs="+",
        choices=[sport.value for sport in Sport],
        default=[sport.value for sport in Sport],
    )
    result.add_argument("--output", help="write a JSON or Markdown report based on the suffix")
    result.add_argument("--markdown", help="also write a human-readable Markdown report")
    result.add_argument(
        "--aggregate",
        nargs="+",
        metavar="REPORT.json",
        help="aggregate single-sport reports instead of running MuJoCo",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.aggregate:
        report = aggregate_reports(load_report(path) for path in args.aggregate)
    else:
        report = run_mujoco_fidelity(Sport(value) for value in args.sports)
    if args.output:
        print(f"report={write_report(report, args.output)}")
    if args.markdown:
        print(f"markdown={write_report(report, args.markdown)}")
    if not args.output and not args.markdown:
        print(report_markdown(report))
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
