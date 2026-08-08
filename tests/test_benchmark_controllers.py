from __future__ import annotations

from math import sqrt

import pytest

from multisport_sim.benchmark.controllers import (
    ControllerObservation,
    NoOpController,
    PaddleCommand,
    ScriptedPaddleController,
)
from multisport_sim.benchmark.types import BallState, ShotSpec


SHOT = ShotSpec(
    shot_id="controller-test",
    sport="table_tennis",
    level="L2",
    position=(1.8, 0.1, 1.1),
    linear_velocity=(-5.6, -0.35, 1.62),
    angular_velocity=(4.0, 55.0, 2.0),
)


def observation(
    time_s: float,
    *,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float] = (-4.0, 0.0, 0.0),
) -> ControllerObservation:
    return ControllerObservation(
        time_s=time_s,
        ball=BallState(position, velocity, (0.0, 0.0, 0.0)),
        paddle_position=(-1.7, 1.25, 1.0),
        paddle_quaternion=(1.0, 0.0, 0.0, 0.0),
    )


def test_noop_controller_never_generates_a_fixture_action() -> None:
    controller = NoOpController()
    controller.reset(SHOT, seed=123)
    assert controller.act(observation(0.0, position=(1.8, 0.0, 1.1))) is None


def test_scripted_controller_tracks_then_executes_a_bounded_swing() -> None:
    controller = ScriptedPaddleController()
    controller.reset(SHOT, seed=123)

    waiting = controller.act(observation(0.4, position=(-0.9, 0.9, 1.5)))
    assert isinstance(waiting, PaddleCommand)
    assert waiting.position == pytest.approx((-1.70, 0.70, 1.35))

    triggered = controller.act(observation(0.5, position=(-1.05, -0.2, 1.0)))
    assert triggered.position == pytest.approx((-1.70, -0.2, 1.0))
    swinging = controller.act(observation(0.55, position=(-1.3, -0.1, 0.9)))
    assert swinging.position == pytest.approx((-1.55, -0.1, 0.9))
    followed_through = controller.act(observation(0.8, position=(-1.5, 0.0, 0.7)))
    assert followed_through.position == pytest.approx((-1.38, 0.0, 0.82))
    assert sqrt(sum(value * value for value in swinging.quaternion)) == pytest.approx(1.0)


def test_scripted_controller_reset_clears_swing_state() -> None:
    controller = ScriptedPaddleController()
    controller.reset(SHOT)
    controller.act(observation(0.5, position=(-1.1, 0.0, 1.0)))
    assert controller.act(observation(0.6, position=(-1.3, 0.0, 1.0))).position[0] > -1.7

    controller.reset(SHOT)
    waiting = controller.act(observation(1.0, position=(-0.5, 0.0, 1.0)))
    assert waiting.position[0] == pytest.approx(-1.7)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backswing_x": -1.3, "follow_through_x": -1.4}, "backswing_x"),
        ({"swing_speed_mps": 0.0}, "swing_speed_mps"),
        ({"minimum_height": 1.2, "maximum_height": 1.0}, "minimum_height"),
        ({"maximum_lateral": -0.1}, "maximum_lateral"),
        ({"swing_trigger_x": float("nan")}, "finite numbers"),
        ({"upward_tilt_degrees": float("inf")}, "finite numbers"),
    ],
)
def test_scripted_controller_rejects_invalid_fixture_limits(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ScriptedPaddleController(**kwargs)
