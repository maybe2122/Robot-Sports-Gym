"""Serialization and human-readable output for Shot Skill benchmark reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _rate(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _number(value: object, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}{suffix}"


def report_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary while keeping JSON as the source of truth."""
    task = report.get("task", "table-tennis-return-v0")
    level = report.get("level", "unknown")
    metrics = report.get("metrics", {})
    assessment = report.get("assessment") or {}
    status = (
        "PASS"
        if assessment.get("passed")
        else "NOT PASSED"
        if assessment
        else "NOT ASSESSED"
    )

    lines = [
        f"# {task} — {level}",
        "",
        "> Experimental v0 fixture report; it is not eligible for a public leaderboard.",
        "",
        f"- Result: **{status}**",
        f"- Backend: `{report.get('backend', {}).get('name', 'unknown')}`",
        f"- Robot/adapter: `{report.get('robot', {}).get('id', 'unknown')}`",
        f"- Policy/controller: `{report.get('policy', {}).get('id', 'unknown')}`",
        f"- Split: `{report.get('shot_bank', {}).get('split', 'unknown')}`",
        f"- Episodes: {report.get('episodes', 0)}",
        f"- Seed: {report.get('seed', 0)}",
        "",
        "## Core metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Hit rate | {_rate(metrics.get('hit_rate'))} |",
        f"| Valid return rate | {_rate(metrics.get('valid_return_rate'))} |",
        f"| Target rate | {_rate(metrics.get('target_rate'))} |",
        f"| Mean target error | {_number(metrics.get('mean_target_error_m'), ' m')} |",
        f"| Incoming-valid rate | {_rate(metrics.get('incoming_valid_rate'))} |",
        f"| Safety violation rate | {_rate(metrics.get('safety_violation_rate'))} |",
    ]

    confidence = report.get(
        "confidence_intervals_95", metrics.get("confidence_intervals_95", {})
    )
    if confidence:
        lines += [
            "",
            "## 95% Wilson confidence intervals",
            "",
            "| Metric | Interval |",
            "|---|---:|",
        ]
        for name, interval in confidence.items():
            formatted = (
                "n/a"
                if interval is None
                else f"{_rate(interval[0])}–{_rate(interval[1])}"
            )
            lines.append(f"| {name.replace('_', ' ')} | {formatted} |")

    buckets = report.get("buckets", {})
    if buckets:
        lines += ["", "## Buckets", "", "| Tag | Episodes | Valid return rate |", "|---|---:|---:|"]
        for tag, values in sorted(buckets.items()):
            bucket_metrics = values.get("metrics", values)
            lines.append(
                f"| `{tag}` | {values.get('episodes', 0)} | "
                f"{_rate(bucket_metrics.get('valid_return_rate'))} |"
            )

    failures = report.get("failures", {})
    nonzero_failures = {reason: count for reason, count in failures.items() if count}
    if nonzero_failures:
        lines += ["", "## Failure reasons", "", "| Reason | Count |", "|---|---:|"]
        for reason, count in sorted(nonzero_failures.items()):
            lines.append(f"| `{reason}` | {count} |")

    notes = assessment.get("reasons", [])
    if not notes:
        notes = [
            f"{criterion['metric']}: {_number(criterion.get('value'))} "
            f"(required >= {_number(criterion.get('threshold'))})"
            for criterion in assessment.get("criteria", [])
        ]
    if notes:
        lines += ["", "## Assessment", ""]
        lines.extend(f"- {note}" for note in notes)

    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write JSON or Markdown selected by the output suffix."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".md", ".markdown"}:
        content = report_markdown(report)
    else:
        content = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        content += "\n"
    output.write_text(content, encoding="utf-8")
    return output
