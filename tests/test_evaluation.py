from __future__ import annotations

import json

import pytest

from multisport_sim.evaluation import (
    BOUNCE_TARGETS,
    aggregate_reports,
    build_report,
    report_markdown,
    run_mujoco_fidelity,
    score_bounce,
    write_report,
)
from multisport_sim.specs import Sport


def test_fidelity_score_has_interpretable_landmarks() -> None:
    target = BOUNCE_TARGETS[Sport.TENNIS]
    ideal = score_bounce(Sport.TENNIS, target.ideal_m)
    boundary = score_bounce(Sport.TENNIS, target.upper_m)
    outside = score_bounce(Sport.TENNIS, target.upper_m + 0.01)

    assert ideal.score == pytest.approx(100.0)
    assert boundary.score == pytest.approx(50.0)
    assert boundary.passed
    assert not outside.passed
    assert outside.score < 50.0


def test_report_separates_official_and_engineering_metrics(tmp_path) -> None:
    report = build_report(
        "test",
        [
            score_bounce(Sport.TENNIS, 1.41),
            score_bounce(Sport.FOOTBALL, 1.0),
        ],
    )
    assert report["summary"]["metric_count"] == 2
    assert report["summary"]["official_metrics_score"] == pytest.approx(100.0)

    output = write_report(report, tmp_path / "fidelity.json")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["metrics"][0]["derived"]["effective_restitution"] > 0.0
    markdown = report_markdown(report)
    assert markdown.startswith("# Robot Sports Gym fidelity report — test")
    assert "acceptance boundary scores 50" in markdown


def test_single_sport_backend_reports_can_be_aggregated() -> None:
    reports = [
        build_report("isaacsim", [score_bounce(Sport.TENNIS, 1.41)]),
        build_report("isaacsim", [score_bounce(Sport.BASKETBALL, 1.06)]),
    ]
    combined = aggregate_reports(reports)
    assert combined["backend"] == "isaacsim"
    assert combined["summary"]["metric_count"] == 2
    assert combined["summary"]["passed"] == 2


def test_real_mujoco_fidelity_suite_passes() -> None:
    report = run_mujoco_fidelity()
    assert report["summary"]["metric_count"] == len(Sport)
    assert report["summary"]["failed"] == 0
    assert report["summary"]["overall_score"] > 50.0
