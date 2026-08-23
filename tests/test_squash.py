from __future__ import annotations

import pytest

from multisport_sim.squash import Player, SquashMatch, SquashRallyJudge


def test_rally_scoring_changes_server_and_records_statistics() -> None:
    match = SquashMatch()
    assert match.begin_serve() is Player.A
    match.record_shot()
    point = match.award_point(Player.B, reason="tin")
    assert (point.server, point.winner, point.shots) == (Player.A, Player.B, 2)
    assert match.server is Player.B
    assert match.statistics()["score"] == {"A": 0, "B": 1}


def test_game_requires_two_point_margin_after_ten_all() -> None:
    match = SquashMatch(score_a=10, score_b=10)
    for winner in (Player.A, Player.B, Player.A, Player.A):
        match.begin_serve()
        match.award_point(winner, reason="out")
    assert match.game_winner is Player.A
    assert match.statistics()["longest_rally_shots"] == 1


def test_score_operations_require_an_active_rally() -> None:
    match = SquashMatch()
    with pytest.raises(RuntimeError, match="begin_serve"):
        match.award_point(Player.A, reason="out")


def test_complete_demo_rally_is_awarded_on_the_second_bounce() -> None:
    match = SquashMatch()
    judge = SquashRallyJudge(match)

    assert judge.begin() is Player.A
    judge.front_wall(at=0.61)
    assert judge.floor_bounce(at=0.92) is None
    assert judge.racket_hit(Player.B, at=1.31) is None
    judge.front_wall(at=1.86)
    assert judge.floor_bounce(at=2.18) is None
    point = judge.floor_bounce(at=3.01)

    assert point is not None
    assert point.as_dict() == {
        "number": 1,
        "server": "A",
        "winner": "B",
        "shots": 2,
        "reason": "second_bounce",
    }
    assert match.statistics()["score"] == {"A": 0, "B": 1}
    assert [event.kind.value for event in judge.events] == [
        "serve",
        "front_wall",
        "floor_bounce",
        "racket_hit",
        "front_wall",
        "floor_bounce",
        "floor_bounce",
        "point",
    ]


def test_floor_before_front_wall_loses_the_point() -> None:
    match = SquashMatch()
    judge = SquashRallyJudge(match)
    judge.begin()
    point = judge.floor_bounce(at=0.2)
    assert point is not None
    assert (point.winner, point.reason) == (Player.B, "floor_before_front_wall")


def test_rally_events_must_be_ordered_and_stop_after_the_point() -> None:
    judge = SquashRallyJudge(SquashMatch())
    judge.begin(at=1.0)
    with pytest.raises(ValueError, match="monotonic"):
        judge.front_wall(at=0.9)
    judge.front_wall(at=1.1)
    judge.floor_bounce(at=1.2)
    judge.floor_bounce(at=1.3)
    with pytest.raises(RuntimeError, match="already complete"):
        judge.front_wall(at=1.4)
