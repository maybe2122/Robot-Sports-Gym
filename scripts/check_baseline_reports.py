#!/usr/bin/env python3
"""Validate committed baseline tables against their packaged shot banks.

This is intentionally fast: CI should catch a stale digest, old sample count,
missing confidence interval, or incomplete robustness breakdown without
rerunning thousands of physics episodes. It validates provenance and shape;
nightly physics regeneration remains a separate concern.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multisport_sim.benchmark.shot_bank import VALID_LEVELS, ShotBank


class BaselineReportError(ValueError):
    """A committed table is inconsistent with its bank or declared schema."""


@dataclass(frozen=True)
class ReportSpec:
    path: Path
    task: str
    bank: str
    splits: tuple[str, ...]
    controllers: tuple[str, ...]


REPORTS = (
    ReportSpec(
        path=Path("reports/table-tennis-panda-baselines.json"),
        task="table-tennis-return-panda-v1",
        bank="table_tennis/return-v1",
        splits=("dev", "test"),
        controllers=("hold", "random", "intercept", "vision"),
    ),
    ReportSpec(
        path=Path("reports/tennis-baselines.json"),
        task="tennis-return-v0",
        bank="tennis/return-v0",
        splits=("test",),
        controllers=("noop", "scripted"),
    ),
)


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineReportError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineReportError(f"{path} must contain a JSON object")
    return payload


def validate_report(spec: ReportSpec) -> dict[str, Any]:
    payload = _load(spec.path)
    if payload.get("schema") != "multisport-baseline-table-v0":
        raise BaselineReportError(f"{spec.path}: unsupported schema")
    if payload.get("task") != spec.task:
        raise BaselineReportError(f"{spec.path}: expected task {spec.task!r}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise BaselineReportError(f"{spec.path}: rows must be a list")

    expected_keys = {
        (split, level, controller)
        for split in spec.splits
        for level in VALID_LEVELS
        for controller in spec.controllers
    }
    actual_keys = {
        (row.get("split", payload.get("split")), row.get("level"), row.get("controller"))
        for row in rows
        if isinstance(row, dict)
    }
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise BaselineReportError(
            f"{spec.path}: incomplete row matrix; missing={missing}, extra={extra}"
        )

    expected_counts: dict[str, int] = {}
    expected_digests: dict[str, str] = {}
    for split in spec.splits:
        bank = ShotBank.from_resource(split=split, task=spec.bank)
        level_counts = Counter(shot.level for shot in bank)
        if len(set(level_counts.values())) != 1:
            raise BaselineReportError(f"{spec.bank}/{split}: levels have unequal sizes")
        expected_counts[split] = next(iter(level_counts.values()))
        expected_digests[split] = bank.digest

    reported_digests = payload.get("shot_bank_digests")
    if reported_digests is None:
        reported_digests = {payload.get("split"): payload.get("shot_bank_digest")}
    if reported_digests != expected_digests:
        raise BaselineReportError(
            f"{spec.path}: bank digest mismatch; expected {expected_digests}"
        )

    for row in rows:
        split = row.get("split", payload.get("split"))
        if row.get("episodes") != expected_counts[split]:
            raise BaselineReportError(
                f"{spec.path}: {split}/{row.get('level')}/{row.get('controller')} "
                f"has n={row.get('episodes')}, expected {expected_counts[split]}"
            )
        interval = row.get("primary_confidence_interval_95")
        if not (
            isinstance(interval, list)
            and len(interval) == 2
            and 0.0 <= interval[0] <= interval[1] <= 1.0
        ):
            raise BaselineReportError(f"{spec.path}: row has no valid primary interval")

    gaps = payload.get("robustness_gap")
    if not isinstance(gaps, list) or len(gaps) != len(spec.splits) * len(spec.controllers):
        raise BaselineReportError(f"{spec.path}: incomplete robustness gaps")
    for gap in gaps:
        breakdown = gap.get("level_breakdown", {})
        if set(breakdown) != set(VALID_LEVELS[1:]):
            raise BaselineReportError(f"{spec.path}: robustness breakdown must cover L1-L5")

    return {
        "path": str(spec.path),
        "rows": len(rows),
        "splits": list(spec.splits),
        "episodes_per_level": expected_counts,
        "shot_bank_digests": expected_digests,
    }


def main() -> int:
    try:
        summaries = [validate_report(spec) for spec in REPORTS]
    except BaselineReportError as exc:
        print(f"check-baseline-reports: error: {exc}", file=sys.stderr)
        return 1
    for summary in summaries:
        counts = ", ".join(
            f"{split}={count}" for split, count in summary["episodes_per_level"].items()
        )
        print(f"{summary['path']}: {summary['rows']} rows, {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
