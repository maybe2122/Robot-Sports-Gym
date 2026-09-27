"""Football and basketball launch judges on synthetic episodes.

Task frames: football's origin is the centre of the goal line on the ground,
basketball's the floor below the rim's centre; in both the robot is at x < 0
and launches toward +x.
"""

from __future__ import annotations

import pytest

from multisport_sim.benchmark.rules.basketball import (
    BASKET,
    RIM_HEIGHT_M,
    BasketballShootJudge,
)
from multisport_sim.benchmark.rules.football import (
    CROSSBAR_HEIGHT_M,
    FOOTBALL_GOAL,
    GOAL_HALF_WIDTH_M,
    FootballKickJudge,
)
from multisport_sim.benchmark.types import BallState, SemanticContact, ShotSpec, TargetSpec

R_FOOT = FOOTBALL_GOAL.ball_radius_m


def ball(x, y, z, vx=0.0, vy=0.0, vz=0.0) -> BallState:
    return BallState((x, y, z), (vx, vy, vz), (0.0, 0.0, 0.0))


def boot() -> SemanticContact:
    return SemanticContact.between("ball", "robot_racket")


def football_shot(level="L2", *, target=None, vx=0.0) -> ShotSpec:
    return ShotSpec(
        f"fb-{level.lower()}", "football", level, (-16.0, 0.0, R_FOOT), (vx, 0.0, 0.0),
        (0.0, 0.0, 0.0), target=target,
    )


def kick(judge, *, y=0.0, z=1.0):
    judge.update(time_s=0.0, ball=ball(-16.0, 0.0, R_FOOT), contacts=[])
    judge.update(time_s=0.2, ball=ball(-16.0, 0.0, R_FOOT), contacts=[boot()])
    judge.update(time_s=0.21, ball=ball(-15.8, 0.0, R_FOOT, vx=20.0), contacts=[])
    judge.update(time_s=0.95, ball=ball(-0.1, y, z, vx=18.0), contacts=[])
    return judge.update(time_s=0.96, ball=ball(0.2, y, z, vx=18.0), contacts=[])


def test_football_geometry_is_the_ifab_goal() -> None:
    assert GOAL_HALF_WIDTH_M * 2 == pytest.approx(7.32)
    assert CROSSBAR_HEIGHT_M == 2.44


def test_a_ball_wholly_over_the_line_inside_the_frame_is_a_goal() -> None:
    judge = FootballKickJudge()
    judge.reset(football_shot())
    result = kick(judge, y=1.5, z=1.2)
    assert judge.done and result.hit and result.valid_return
    assert result.landing_xy == (1.5, 1.2)
    assert result.outgoing_speed_mps == pytest.approx(20.0)


def test_a_ball_touching_the_line_is_not_yet_a_goal() -> None:
    judge = FootballKickJudge()
    judge.reset(football_shot())
    judge.update(time_s=0.0, ball=ball(-16.0, 0.0, R_FOOT), contacts=[])
    judge.update(time_s=0.2, ball=ball(-16.0, 0.0, R_FOOT), contacts=[boot()])
    judge.update(time_s=0.9, ball=ball(-0.2, 0.0, 1.0, vx=18.0), contacts=[])
    judge.update(time_s=0.91, ball=ball(0.05, 0.0, 1.0, vx=18.0), contacts=[])
    assert not judge.done


@pytest.mark.parametrize(("y", "z"), [(4.0, 1.0), (0.0, 2.6), (-3.8, 0.5)])
def test_wide_or_over_is_out(y: float, z: float) -> None:
    judge = FootballKickJudge()
    judge.reset(football_shot())
    result = kick(judge, y=y, z=z)
    assert not result.valid_return
    assert result.failure_reason == "out"


def test_l3_measures_placement_in_the_goal_mouth() -> None:
    judge = FootballKickJudge()
    judge.reset(football_shot("L3", target=TargetSpec((2.8, 1.8), 0.5)))
    result = kick(judge, y=2.8, z=1.5)
    assert result.target_hit
    assert result.target_error_m == pytest.approx(0.3)


def test_woodwork_is_recorded_and_the_ball_can_still_go_in() -> None:
    judge = FootballKickJudge()
    judge.reset(football_shot())
    judge.update(time_s=0.0, ball=ball(-16.0, 0.0, R_FOOT), contacts=[])
    judge.update(time_s=0.2, ball=ball(-16.0, 0.0, R_FOOT), contacts=[boot()])
    judge.update(
        time_s=0.9, ball=ball(-0.1, 3.5, 2.3, vx=10.0),
        contacts=[SemanticContact.between("ball", "post")],
    )
    judge.update(time_s=0.91, ball=ball(-0.05, 3.4, 2.2, vx=10.0), contacts=[])
    result = judge.update(time_s=0.93, ball=ball(0.15, 3.3, 2.1, vx=10.0), contacts=[])
    assert result.net_touch and result.valid_return


def test_football_l0_settles_an_untouched_ball() -> None:
    judge = FootballKickJudge()
    judge.reset(football_shot("L0"))
    judge.update(time_s=0.0, ball=ball(-16.0, 0.0, R_FOOT), contacts=[])
    judge.update(time_s=0.5, ball=ball(-16.0, 0.0, R_FOOT), contacts=[])
    assert not judge.done
    result = judge.update(time_s=1.0, ball=ball(-16.0, 0.0, R_FOOT), contacts=[])
    assert judge.done and result.incoming_valid and result.failure_reason is None


def test_a_ball_rolling_away_from_the_kicker_is_not_a_legal_start() -> None:
    judge = FootballKickJudge()
    judge.reset(football_shot("L2", vx=2.0))
    judge.update(time_s=0.0, ball=ball(-16.0, 0.0, R_FOOT, vx=2.0), contacts=[])
    assert not judge.result.incoming_valid


def basket_shot(level="L2", *, target=None) -> ShotSpec:
    return ShotSpec(
        f"bb-{level.lower()}", "basketball", level, (-4.2, 0.0, 2.1), (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0), target=target,
    )


def shoot(judge, *, x=0.02, y=0.0, contacts=()):
    judge.update(time_s=0.0, ball=ball(-4.2, 0.0, 2.1), contacts=[])
    judge.update(time_s=0.15, ball=ball(-4.2, 0.0, 2.0), contacts=[boot()])
    judge.update(time_s=0.16, ball=ball(-4.1, 0.0, 2.05, vx=4.0, vz=6.0), contacts=[])
    judge.update(time_s=0.6, ball=ball(-1.5, 0.0, 4.0, vx=4.0, vz=0.0), contacts=[])
    judge.update(time_s=0.9, ball=ball(x - 0.04, y, RIM_HEIGHT_M + 0.03, vx=4.0, vz=-4.0),
                 contacts=list(contacts))
    return judge.update(time_s=0.91, ball=ball(x, y, RIM_HEIGHT_M - 0.03, vx=4.0, vz=-4.0),
                        contacts=[])


def test_down_through_the_ring_is_a_basket() -> None:
    judge = BasketballShootJudge()
    judge.reset(basket_shot())
    result = shoot(judge)
    assert judge.done and result.valid_return and not result.net_touch
    assert result.landing_xy == (0.02, 0.0)


def test_a_rim_touch_that_drops_is_still_a_basket() -> None:
    judge = BasketballShootJudge()
    judge.reset(basket_shot())
    result = shoot(judge, x=0.15, contacts=[SemanticContact.between("ball", "rim")])
    assert result.valid_return and result.net_touch


def test_passing_beside_the_ring_is_not_a_basket() -> None:
    judge = BasketballShootJudge()
    judge.reset(basket_shot())
    shoot(judge, x=0.30)
    assert not judge.done
    result = judge.update(
        time_s=1.4, ball=ball(0.5, 0.0, 0.12),
        contacts=[SemanticContact.between("ball", "floor")],
    )
    assert result.failure_reason == "out"


def test_up_through_the_ring_is_a_violation() -> None:
    judge = BasketballShootJudge()
    judge.reset(basket_shot())
    judge.update(time_s=0.0, ball=ball(-4.2, 0.0, 2.1), contacts=[])
    judge.update(time_s=0.15, ball=ball(-4.2, 0.0, 2.0), contacts=[boot()])
    judge.update(time_s=0.5, ball=ball(0.0, 0.0, RIM_HEIGHT_M - 0.05, vz=3.0), contacts=[])
    result = judge.update(time_s=0.51, ball=ball(0.0, 0.0, RIM_HEIGHT_M + 0.01, vz=3.0),
                          contacts=[])
    assert result.failure_reason == "fault"


def test_basketball_l0_accepts_a_release_that_falls_in_front_of_the_basket() -> None:
    judge = BasketballShootJudge()
    judge.reset(basket_shot("L0"))
    judge.update(time_s=0.0, ball=ball(-4.2, 0.0, 2.1), contacts=[])
    result = judge.update(
        time_s=0.6, ball=ball(-4.2, 0.0, 0.12),
        contacts=[SemanticContact.between("ball", "floor")],
    )
    assert result.incoming_valid and result.failure_reason is None


def test_a_centred_entry_hits_the_l3_target() -> None:
    judge = BasketballShootJudge()
    judge.reset(basket_shot("L3", target=TargetSpec((0.0, 0.0), 0.08)))
    result = shoot(judge, x=0.05)
    assert result.target_hit
    assert BASKET.inside_ring(0.2, 0.0) and not BASKET.inside_ring(0.23, 0.0)
