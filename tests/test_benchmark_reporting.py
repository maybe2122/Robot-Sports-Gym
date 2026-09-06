from __future__ import annotations

import json

import pytest

from multisport_sim.benchmark.reporting import report_markdown, write_report


def _report() -> dict[str, object]:
    return {
        "task": "table-tennis-return-v0",
        "level": "L2",
        "backend": {"name": "mujoco"},
        "robot": {"id": "fixture"},
        "policy": {"id": "scripted-v0"},
        "seed": 7,
        "episodes": 4,
        "shot_bank": {"split": "dev"},
        "metrics": {
            "hit_rate": 0.75,
            "valid_return_rate": 0.5,
            "target_rate": 0.25,
            "mean_target_error_m": 0.1,
            "incoming_valid_rate": 1.0,
            "safety_violation_rate": 0.0,
            "confidence_intervals_95": {"hit_rate": [0.30, 0.95]},
        },
        "buckets": {
            "fast": {
                "episodes": 2,
                "valid_return_rate": 0.5,
                "confidence_intervals_95": {"valid_return_rate": [0.095, 0.905]},
            }
        },
        "bucket_groups": {
            "spin": {
                "episodes": 2,
                "metrics": {"valid_return_rate": 0.5},
                "confidence_intervals_95": {"valid_return_rate": [0.095, 0.905]},
            }
        },
        "failures": {"miss": 1, "timeout": 1},
        "assessment": {"passed": False, "reasons": ["return rate is below 80%"]},
    }


def test_markdown_exposes_result_metrics_buckets_and_failures() -> None:
    markdown = report_markdown(_report())

    assert "table-tennis-return-v0 — L2" in markdown
    assert "NOT PASSED" in markdown
    assert "75.0%" in markdown
    assert "`fast`" in markdown
    assert "`spin`" in markdown
    assert "9.5%–90.5%" in markdown
    assert "`miss`" in markdown
    assert "below 80%" in markdown


def test_write_report_selects_json_or_markdown_by_suffix(tmp_path) -> None:
    report = _report()
    json_path = write_report(report, tmp_path / "nested" / "result.json")
    markdown_path = write_report(report, tmp_path / "result.md")

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert "# table-tennis-return-v0" in markdown_path.read_text(encoding="utf-8")


def test_json_report_rejects_non_finite_values(tmp_path) -> None:
    report = _report()
    report["metrics"] = {"hit_rate": float("nan")}

    with pytest.raises(ValueError):
        write_report(report, tmp_path / "bad.json")
