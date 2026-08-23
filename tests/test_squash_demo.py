from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from multisport_sim.squash_demo import (
    SquashDemoValidationError,
    inspect_gif,
    validate_score_report,
)


def _valid_report() -> dict[str, object]:
    kinds = (
        "serve",
        "front_wall",
        "floor_bounce",
        "racket_hit",
        "front_wall",
        "floor_bounce",
        "floor_bounce",
        "point",
    )
    players = ("A", "A", "A", "B", "B", "B", "B", "B")
    times = (0.0, 0.633, 0.983, 1.583, 2.167, 2.450, 2.833, 2.833)
    details = {3: "1", 6: "1", 7: "2", 8: "second_bounce"}
    events = []
    for number, (kind, player, time_s) in enumerate(
        zip(kinds, players, times, strict=True), start=1
    ):
        event: dict[str, object] = {
            "number": number,
            "time_s": time_s,
            "kind": kind,
            "player": player,
        }
        if number in details:
            event["detail"] = details[number]
        events.append(event)
    return {
        "schema": "squash-score-demo-v1",
        "complete": True,
        "duration_s": 3.833,
        "score": {"A": 0, "B": 1},
        "server": "B",
        "rally": {
            "number": 1,
            "server": "A",
            "winner": "B",
            "shots": 2,
            "reason": "second_bounce",
        },
        "events": events,
    }


def test_score_demo_report_contract() -> None:
    summary = validate_score_report(_valid_report())
    assert summary == {
        "schema": "squash-score-demo-v1",
        "score": {"A": 0, "B": 1},
        "winner": "B",
        "reason": "second_bounce",
        "events": 8,
        "point_time_s": 2.833,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda report: report.update(score={"A": 1, "B": 0}), "final score"),
        (lambda report: report["events"].pop(3), "expected 8 events"),
        (
            lambda report: report["events"][4].update(time_s=0.2),
            "monotonic",
        ),
        (
            lambda report: report["events"][6].update(detail="1"),
            "counters",
        ),
    ),
)
def test_invalid_score_demo_reports_are_rejected(mutation, message: str) -> None:
    report = deepcopy(_valid_report())
    mutation(report)
    with pytest.raises(SquashDemoValidationError, match=message):
        validate_score_report(report)


def test_truncated_gif_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.gif"
    path.write_bytes(b"GIF89a\x80")
    with pytest.raises(SquashDemoValidationError, match="truncated GIF"):
        inspect_gif(path)
