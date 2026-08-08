from __future__ import annotations

import pytest

from multisport_sim.benchmark.rules.table_tennis import TABLE_TENNIS, TableTennisReturnJudge
from multisport_sim.benchmark.types import BallState, SemanticContact, ShotSpec, TargetSpec
from multisport_sim.specs import TABLE_TENNIS as PHYSICS_TABLE_TENNIS


def state(
    x: float,
    y: float = 0.0,
    z: float = 1.0,
    *,
    vx: float = -2.0,
) -> BallState:
    return BallState((x, y, z), (vx, 0.0, 0.0), (0.0, -20.0, 0.0))


def contact(first: str, second: str, *, x: float, y: float = 0.0) -> SemanticContact:
    return SemanticContact.between(
        first,
        second,
        position=(x, y, 0.76),
        normal=(0.0, 0.0, 1.0),
    )


def shot(level: str = "L2", *, target: TargetSpec | None = None) -> ShotSpec:
    return ShotSpec(
        f"tt-{level.lower()}-test",
        "table_tennis",
        level,
        (1.0, 0.0, 1.2),
        (-3.0, 0.0, 0.0),
        (0.0, -20.0, 0.0),
        target=target,
    )


def prepare_legal_incoming(judge: TableTennisReturnJudge) -> None:
    judge.update(time_s=0.0, ball=state(1.0), contacts=[])
    judge.update(time_s=0.2, ball=state(-0.1), contacts=[])
    judge.update(
        time_s=0.3,
        ball=state(-0.5),
        contacts=[contact("ball", "table", x=-0.5)],
    )
    judge.update(time_s=0.31, ball=state(-0.55), contacts=[])
    assert judge.result.incoming_valid


def hit_ball(judge: TableTennisReturnJudge, *, time_s: float = 0.4) -> None:
    judge.update(
        time_s=time_s,
        ball=state(-0.8, vx=3.0),
        contacts=[SemanticContact.between("ball", "robot_racket")],
    )
    judge.update(time_s=time_s + 0.01, ball=state(-0.7, vx=3.0), contacts=[])


def cross_net(judge: TableTennisReturnJudge, *, time_s: float = 0.6) -> None:
    judge.update(time_s=time_s, ball=state(0.1, vx=3.0), contacts=[])
    assert judge.result.crossed_net


def test_rule_geometry_is_derived_from_authoritative_physics_spec() -> None:
    assert TABLE_TENNIS.length_m == PHYSICS_TABLE_TENNIS.table_length
    assert TABLE_TENNIS.width_m == PHYSICS_TABLE_TENNIS.table_width
    assert TABLE_TENNIS.top_height_m == PHYSICS_TABLE_TENNIS.table_height
    assert TABLE_TENNIS.net_plane_x_m == PHYSICS_TABLE_TENNIS.net_plane_x


def test_l0_finishes_as_soon_as_first_incoming_bounce_is_legal() -> None:
    judge = TableTennisReturnJudge()
    judge.reset(shot("L0"))

    judge.update(time_s=0.0, ball=state(1.0), contacts=[])
    judge.update(time_s=0.2, ball=state(-0.1), contacts=[])
    result = judge.update(
        time_s=0.3,
        ball=state(-0.5),
        contacts=[contact("ball", "table", x=-0.5)],
    )

    assert judge.done
    assert result.incoming_valid
    assert result.failure_reason is None


def test_l0_rejects_first_bounce_on_wrong_side() -> None:
    judge = TableTennisReturnJudge()
    judge.reset(shot("L0"))
    judge.update(time_s=0.0, ball=state(1.0), contacts=[])

    result = judge.update(
        time_s=0.1,
        ball=state(0.5),
        contacts=[contact("ball", "table", x=0.5)],
    )

    assert judge.done
    assert not result.incoming_valid
    assert result.failure_reason == "out"


def test_racket_contact_is_counted_once_and_l1_finishes_on_hit() -> None:
    judge = TableTennisReturnJudge()
    judge.reset(shot("L1"))
    prepare_legal_incoming(judge)

    first = judge.update(
        time_s=0.4,
        ball=state(-0.8, vx=3.0),
        contacts=[SemanticContact.between("robot_racket", "ball")],
    )
    same = judge.update(
        time_s=0.5,
        ball=state(-0.7, vx=3.0),
        contacts=[SemanticContact.between("ball", "robot_racket")],
    )

    assert first is same
    assert judge.done
    assert same.hit
    assert same.contact_time_s == pytest.approx(0.4)


def test_hit_uses_previous_tick_for_incoming_and_racket_release_for_outgoing_state() -> None:
    judge = TableTennisReturnJudge()
    judge.reset(shot())
    prepare_legal_incoming(judge)
    judge.update(
        time_s=0.39,
        ball=BallState((-0.75, 0.0, 1.0), (-4.0, 0.0, 0.0), (0.0, -30.0, 0.0)),
        contacts=[],
    )

    result = judge.update(
        time_s=0.4,
        ball=BallState((-0.8, 0.0, 1.0), (1.1, 0.0, 0.0), (0.0, -10.0, 0.0)),
        contacts=[SemanticContact.between("ball", "robot_racket")],
    )

    assert result.incoming_speed_mps == pytest.approx(4.0)
    assert result.incoming_spin_radps == pytest.approx((0.0, -30.0, 0.0))
    assert result.outgoing_speed_mps is None

    judge.update(
        time_s=0.401,
        ball=BallState((-0.799, 0.0, 1.0), (2.0, 0.0, 0.0), (0.0, -9.0, 0.0)),
        contacts=[SemanticContact.between("ball", "robot_racket")],
    )
    assert result.outgoing_speed_mps is None

    judge.update(
        time_s=0.402,
        ball=BallState((-0.795, 0.0, 1.0), (4.06, 0.0, 0.0), (0.0, -8.0, 0.0)),
        contacts=[],
    )
    assert result.outgoing_speed_mps == pytest.approx(4.06)


def test_hit_cross_net_and_first_opponent_landing_is_valid_return() -> None:
    judge = TableTennisReturnJudge()
    judge.reset(shot())
    prepare_legal_incoming(judge)
    hit_ball(judge)
    cross_net(judge)

    result = judge.update(
        time_s=0.7,
        ball=state(0.6, 0.2, 0.78, vx=2.0),
        contacts=[contact("table", "ball", x=0.6, y=0.2)],
    )

    assert judge.done
    assert result.hit
    assert result.valid_return
    assert result.landing_xy == pytest.approx((0.6, 0.2))
    assert result.failure_reason is None


def test_hit_then_own_side_bounce_fails_and_cannot_later_flip() -> None:
    judge = TableTennisReturnJudge()
    judge.reset(shot())
    prepare_legal_incoming(judge)
    hit_ball(judge)

    failed = judge.update(
        time_s=0.5,
        ball=state(-0.3, vx=2.0),
        contacts=[contact("ball", "table", x=-0.3)],
    )
    snapshot = failed.to_dict()
    later = judge.update(
        time_s=0.8,
        ball=state(0.5, vx=2.0),
        contacts=[contact("ball", "table", x=0.5)],
    )

    assert failed.failure_reason == "own_side"
    assert not failed.valid_return
    assert failed.landing_xy == pytest.approx((-0.3, 0.0))
    assert later.to_dict() == snapshot


def test_floor_before_and_after_hit_are_miss_and_out() -> None:
    before = TableTennisReturnJudge()
    before.reset(shot())
    missed = before.update(
        time_s=0.2,
        ball=state(-1.8, z=0.02),
        contacts=[SemanticContact.between("ball", "floor")],
    )
    assert missed.failure_reason == "miss"

    after = TableTennisReturnJudge()
    after.reset(shot())
    prepare_legal_incoming(after)
    hit_ball(after)
    out = after.update(
        time_s=0.6,
        ball=state(1.8, z=0.02, vx=2.0),
        contacts=[SemanticContact.between("floor", "ball")],
    )
    assert out.failure_reason == "out"


def test_second_robot_side_bounce_before_hit_is_a_miss() -> None:
    judge = TableTennisReturnJudge()
    judge.reset(shot())
    prepare_legal_incoming(judge)

    result = judge.update(
        time_s=0.7,
        ball=state(-1.1, z=0.78),
        contacts=[contact("ball", "table", x=-1.1)],
    )

    assert judge.done
    assert result.incoming_valid
    assert not result.hit
    assert result.failure_reason == "miss"


def test_net_touch_is_not_immediate_failure_and_can_still_succeed() -> None:
    judge = TableTennisReturnJudge()
    judge.reset(shot())
    prepare_legal_incoming(judge)
    hit_ball(judge)

    touched = judge.update(
        time_s=0.5,
        ball=state(-0.01, z=0.9, vx=2.0),
        contacts=[SemanticContact.between("ball", "net")],
    )
    assert not judge.done
    assert touched.net_touch

    cross_net(judge, time_s=0.55)
    result = judge.update(
        time_s=0.7,
        ball=state(0.5, vx=2.0),
        contacts=[contact("ball", "table", x=0.5)],
    )
    assert result.valid_return
    assert result.net_touch
    assert result.failure_reason is None


def test_net_then_floor_reports_net_failure() -> None:
    judge = TableTennisReturnJudge()
    judge.reset(shot())
    prepare_legal_incoming(judge)
    hit_ball(judge)
    judge.update(
        time_s=0.5,
        ball=state(-0.01, z=0.85, vx=2.0),
        contacts=[SemanticContact.between("ball", "net")],
    )
    judge.update(time_s=0.51, ball=state(-0.05, z=0.5), contacts=[])

    result = judge.update(
        time_s=0.7,
        ball=state(-0.2, z=0.02),
        contacts=[SemanticContact.between("ball", "floor")],
    )
    assert result.failure_reason == "net"


def test_net_touch_that_crosses_before_landing_out_reports_out() -> None:
    judge = TableTennisReturnJudge()
    judge.reset(shot())
    prepare_legal_incoming(judge)
    hit_ball(judge)
    judge.update(
        time_s=0.5,
        ball=state(-0.01, z=0.9, vx=2.0),
        contacts=[SemanticContact.between("ball", "net")],
    )
    judge.update(time_s=0.55, ball=state(0.1, z=0.7, vx=2.0), contacts=[])

    result = judge.update(
        time_s=0.7,
        ball=state(1.8, z=0.02, vx=2.0),
        contacts=[SemanticContact.between("ball", "floor")],
    )
    assert result.crossed_net
    assert result.failure_reason == "out"


def test_timeout_is_terminal_but_success_at_boundary_has_priority() -> None:
    timeout = TableTennisReturnJudge(timeout_s=2.0)
    timeout.reset(shot())
    timed_out = timeout.update(time_s=2.0, ball=state(1.0), contacts=[])
    assert timed_out.failure_reason == "timeout"

    success = TableTennisReturnJudge(timeout_s=2.0)
    success.reset(shot())
    prepare_legal_incoming(success)
    hit_ball(success)
    cross_net(success)
    returned = success.update(
        time_s=2.0,
        ball=state(0.5, vx=2.0),
        contacts=[contact("ball", "table", x=0.5)],
    )
    assert returned.valid_return
    assert returned.failure_reason is None


def test_target_uses_first_landing_error_and_includes_radius_boundary() -> None:
    target = TargetSpec((0.6, 0.1), 0.2)
    judge = TableTennisReturnJudge()
    judge.reset(shot("L3", target=target))
    prepare_legal_incoming(judge)
    hit_ball(judge)
    cross_net(judge)

    result = judge.update(
        time_s=0.7,
        ball=state(0.8, 0.1, vx=2.0),
        contacts=[contact("ball", "table", x=0.8, y=0.1)],
    )

    assert result.valid_return
    assert result.target_error_m == pytest.approx(0.2)
    assert result.target_hit


@pytest.mark.parametrize(
    ("x", "y", "valid"),
    [
        (1.37, 0.7625, True),
        (1.370001, 0.0, False),
        (0.5, 0.762501, False),
        (0.0, 0.0, False),
    ],
)
def test_opponent_table_boundaries(x: float, y: float, valid: bool) -> None:
    judge = TableTennisReturnJudge()
    judge.reset(shot())
    prepare_legal_incoming(judge)
    hit_ball(judge)
    cross_net(judge)

    result = judge.update(
        time_s=0.7,
        ball=state(x, y, vx=2.0),
        contacts=[contact("ball", "table", x=x, y=y)],
    )
    assert result.valid_return is valid


def test_vertical_table_side_contact_is_not_a_valid_landing() -> None:
    judge = TableTennisReturnJudge()
    judge.reset(shot())
    prepare_legal_incoming(judge)
    hit_ball(judge)
    cross_net(judge)

    result = judge.update(
        time_s=0.7,
        ball=state(0.5, vx=2.0),
        contacts=[
            SemanticContact.between(
                "ball",
                "table",
                position=(0.5, 0.0, 0.72),
                normal=(1.0, 0.0, 0.0),
            )
        ],
    )
    assert not judge.done
    assert not result.valid_return


def test_done_judge_is_idempotent_even_if_later_input_is_stale() -> None:
    judge = TableTennisReturnJudge()
    judge.reset(shot())
    result = judge.update(
        time_s=0.1,
        ball=state(-1.8, z=0.02),
        contacts=[SemanticContact.between("ball", "floor")],
    )
    snapshot = result.to_dict()

    same = judge.update(
        time_s=-100.0,
        ball=state(0.5),
        contacts=[contact("ball", "table", x=0.5)],
    )

    assert same is result
    assert same.to_dict() == snapshot


def test_adapter_can_abort_a_numerical_failure_into_the_episode_denominator() -> None:
    judge = TableTennisReturnJudge()
    judge.reset(shot())

    result = judge.abort(failure_reason="numerical", time_s=0.2)

    assert judge.done
    assert result.failure_reason == "numerical"
    assert result.episode_time_s == pytest.approx(0.2)
    assert judge.abort(failure_reason="numerical", time_s=0.1) is result
