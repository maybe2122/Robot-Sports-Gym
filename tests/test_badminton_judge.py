"""BWF serve rules as the badminton serve judge applies them.

Coordinates follow every task in this project: the net is the plane x = 0, the
robot serves from x < 0 facing +x, and z is up.  Facing +x, the server's right
hand is -y; the receiver faces -x, so the receiver's right hand is +y.  A serve
from the server's right court (y < 0) therefore belongs in the receiver's right
court (y > 0): diagonally opposite, across the centre line.
"""

from __future__ import annotations

import pytest

from multisport_sim.benchmark.rules.badminton import (
    BADMINTON,
    SERVICE_HEIGHT_LIMIT_M,
    BadmintonServeJudge,
)
from multisport_sim.benchmark.types import BallState, SemanticContact, ShotSpec, TargetSpec
from multisport_sim.specs import COURTS, Sport


def state(x: float, y: float = -0.8, z: float = 1.0, *, vx: float = 0.0) -> BallState:
    return BallState((x, y, z), (vx, 0.0, 0.0), (0.0, 0.0, 0.0))


def court(x: float, y: float) -> SemanticContact:
    return SemanticContact.between(
        "ball", "table", position=(x, y, 0.0), normal=(0.0, 0.0, 1.0)
    )


def racket(z: float = 0.90) -> SemanticContact:
    return SemanticContact.between("ball", "robot_racket", position=(-2.6, -0.8, z))


def shot(level: str = "L2", *, y: float = -0.8, target: TargetSpec | None = None) -> ShotSpec:
    return ShotSpec(
        f"bd-{level.lower()}-test",
        "badminton",
        level,
        (-2.6, y, 1.05),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        target=target,
    )


def serve(judge: BadmintonServeJudge, *, y: float = -0.8, contact_z: float = 0.90) -> None:
    """Release, then strike: the shuttle falls, and the face meets it."""
    judge.update(time_s=0.0, ball=state(-2.6, y, 1.05), contacts=[])
    judge.update(time_s=0.20, ball=state(-2.6, y, contact_z), contacts=[racket(contact_z)])
    judge.update(time_s=0.21, ball=state(-2.4, y, contact_z + 0.2, vx=20.0), contacts=[])


def fly_over(judge: BadmintonServeJudge, *, y: float = 0.0, time_s: float = 0.8) -> None:
    judge.update(time_s=time_s, ball=state(-0.1, y, 4.0, vx=8.0), contacts=[])
    judge.update(time_s=time_s + 0.02, ball=state(0.1, y, 4.0, vx=8.0), contacts=[])
    assert judge.result.crossed_net


def land(judge: BadmintonServeJudge, x: float, y: float, *, time_s: float = 2.0):
    return judge.update(time_s=time_s, ball=state(x, y, 0.01), contacts=[court(x, y)])


def test_court_geometry_is_the_bwf_singles_court() -> None:
    assert BADMINTON.length_m == COURTS[Sport.BADMINTON].length == 13.40
    assert BADMINTON.width_m == 5.18
    assert BADMINTON.short_service_line_m == 1.98
    assert BADMINTON.net_height_m == 1.524
    assert SERVICE_HEIGHT_LIMIT_M == 1.15


@pytest.mark.parametrize(
    ("server_y", "landing", "inside"),
    [
        (-0.8, (4.0, 1.0), True),  # right court to the receiver's right court
        (0.8, (4.0, -1.0), True),  # left court to the receiver's left court
        (-0.8, (4.0, -1.0), False),  # straight ahead is the wrong court
        (-0.8, (1.9, 1.0), False),  # short of the short service line
        (-0.8, (6.71, 1.0), False),  # beyond the back boundary line
        (-0.8, (4.0, 2.60), False),  # outside the singles sideline
        (-0.8, (1.98, 0.0), True),  # the lines are part of the court
        (-0.8, (6.70, 2.59), True),
    ],
)
def test_the_receiving_service_court_is_diagonally_opposite(
    server_y: float, landing: tuple[float, float], inside: bool
) -> None:
    assert BADMINTON.in_receiving_service_court(*landing, server_y=server_y) is inside


def test_l0_accepts_a_release_that_falls_on_the_servers_own_half() -> None:
    judge = BadmintonServeJudge()
    judge.reset(shot("L0"))
    judge.update(time_s=0.0, ball=state(-2.6, z=1.05), contacts=[])
    result = land(judge, -2.6, -0.8, time_s=0.45)
    assert judge.done
    assert result.incoming_valid
    assert result.failure_reason is None


def test_l0_rejects_a_release_from_the_receivers_half() -> None:
    judge = BadmintonServeJudge()
    judge.reset(shot("L0"))
    judge.update(time_s=0.0, ball=state(2.6, z=1.05), contacts=[])
    result = land(judge, 2.6, -0.8, time_s=0.45)
    assert not result.incoming_valid
    assert result.failure_reason == "out"


def test_a_legal_serve_into_the_diagonal_court_is_valid() -> None:
    judge = BadmintonServeJudge()
    judge.reset(shot("L2"))
    serve(judge)
    fly_over(judge)
    result = land(judge, 4.5, 1.2)
    assert judge.done
    assert result.incoming_valid and result.hit and result.crossed_net
    assert result.valid_return
    assert result.failure_reason is None
    assert result.landing_xy == (4.5, 1.2)
    assert result.outgoing_speed_mps == pytest.approx(20.0)


def test_a_serve_into_the_straight_court_is_out() -> None:
    judge = BadmintonServeJudge()
    judge.reset(shot("L2"))
    serve(judge)
    fly_over(judge, y=-1.0)
    result = land(judge, 4.5, -1.2)
    assert not result.valid_return
    assert result.failure_reason == "out"


def test_a_strike_above_the_service_height_is_a_fault_at_once() -> None:
    judge = BadmintonServeJudge()
    judge.reset(shot("L2"))
    serve(judge, contact_z=1.20)
    assert judge.done
    assert judge.result.hit
    assert judge.result.failure_reason == "fault"


def test_a_net_cord_serve_that_carries_into_the_court_is_good() -> None:
    judge = BadmintonServeJudge()
    judge.reset(shot("L2"))
    serve(judge)
    judge.update(
        time_s=0.8,
        ball=state(-0.02, 0.0, 1.5, vx=3.0),
        contacts=[SemanticContact.between("ball", "net")],
    )
    fly_over(judge, time_s=0.9)
    result = land(judge, 2.2, 0.5)
    assert result.net_touch
    assert result.valid_return


def test_a_serve_caught_by_the_net_is_scored_as_net() -> None:
    judge = BadmintonServeJudge()
    judge.reset(shot("L2"))
    serve(judge)
    judge.update(
        time_s=0.8,
        ball=state(-0.02, 0.0, 1.2, vx=3.0),
        contacts=[SemanticContact.between("ball", "net")],
    )
    result = land(judge, -0.1, 0.0, time_s=1.2)
    assert not result.crossed_net
    assert result.failure_reason == "net"


def test_a_serve_that_never_reaches_the_net_is_own_side() -> None:
    judge = BadmintonServeJudge()
    judge.reset(shot("L2"))
    serve(judge)
    result = land(judge, -1.0, 0.0, time_s=1.0)
    assert result.failure_reason == "own_side"


def test_l1_finishes_at_the_strike() -> None:
    judge = BadmintonServeJudge()
    judge.reset(shot("L1"))
    serve(judge)
    assert judge.done
    assert judge.result.hit
    assert judge.result.failure_reason is None


def test_a_shuttle_dropped_without_a_strike_is_a_miss() -> None:
    judge = BadmintonServeJudge()
    judge.reset(shot("L2"))
    judge.update(time_s=0.0, ball=state(-2.6, z=1.05), contacts=[])
    result = land(judge, -2.6, -0.8, time_s=0.45)
    assert result.incoming_valid
    assert not result.hit
    assert result.failure_reason == "miss"


def test_l3_scores_the_landing_against_its_target() -> None:
    target = TargetSpec((5.5, 1.5), 0.6)
    judge = BadmintonServeJudge()
    judge.reset(shot("L3", target=target))
    serve(judge)
    fly_over(judge)
    result = land(judge, 5.8, 1.5)
    assert result.valid_return and result.target_hit
    assert result.target_error_m == pytest.approx(0.3)


def test_an_unfinished_flight_times_out() -> None:
    judge = BadmintonServeJudge(timeout_s=1.0)
    judge.reset(shot("L2"))
    serve(judge)
    result = judge.update(time_s=1.0, ball=state(-1.0, z=5.0, vx=5.0), contacts=[])
    assert result.failure_reason == "timeout"


def test_a_shot_for_another_sport_is_refused() -> None:
    judge = BadmintonServeJudge()
    with pytest.raises(ValueError, match="badminton"):
        judge.reset(ShotSpec("x", "tennis", "L2", (-2.6, 0.0, 1.0), (0, 0, 0), (0, 0, 0)))
