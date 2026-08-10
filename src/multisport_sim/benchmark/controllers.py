"""Backend-neutral controller contracts and MuJoCo reference-fixture controllers."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin, sqrt
from typing import Protocol, TypeVar, runtime_checkable

from .types import BallState, ShotSpec

Vec3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
ObservationT = TypeVar("ObservationT", contravariant=True)
ActionT = TypeVar("ActionT", covariant=True)


@dataclass(frozen=True)
class PaddleCommand:
    """World-frame target pose for the optional benchmark paddle.

    This is an action understood by the reference mocap fixture, not a required
    action representation for robot adapters.  A real robot backend is free to
    expose joint position, velocity, torque, or end-effector commands instead.
    """

    position: Vec3
    quaternion: Quaternion


@dataclass(frozen=True)
class ControllerObservation:
    """Privileged state observation used only by the scripted reference fixture."""

    time_s: float
    ball: BallState
    paddle_position: Vec3
    paddle_quaternion: Quaternion


@runtime_checkable
class Controller(Protocol[ObservationT, ActionT]):
    """Minimal policy contract; backend adapters retain ownership of actions."""

    controller_id: str

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        ...

    def act(self, observation: ObservationT) -> ActionT:
        ...


class NoOpController:
    """A deterministic controller that leaves the benchmark blade parked."""

    controller_id = "noop-v0"

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        del shot, seed

    def act(self, observation: object) -> None:
        del observation
        return None


class ScriptedPaddleController:
    """Track the ball laterally and execute a deterministic forward swing.

    The controller intentionally uses privileged ball state.  It exists to
    exercise real MuJoCo contact and the benchmark judge before a robot asset is
    attached; its scores must not be reported as robot-policy results.
    """

    controller_id = "scripted-mocap-paddle-v0"

    def __init__(
        self,
        *,
        backswing_x: float = -1.70,
        follow_through_x: float = -1.38,
        swing_trigger_x: float = -1.0,
        swing_speed_mps: float = 3.0,
        upward_tilt_degrees: float = 28.0,
        minimum_height: float = 0.82,
        maximum_height: float = 1.35,
        maximum_lateral: float = 0.70,
    ) -> None:
        try:
            values = tuple(
                float(value)
                for value in (
                    backswing_x,
                    follow_through_x,
                    swing_trigger_x,
                    swing_speed_mps,
                    upward_tilt_degrees,
                    minimum_height,
                    maximum_height,
                    maximum_lateral,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("scripted controller limits must be finite numbers") from exc
        if not all(isfinite(value) for value in values):
            raise ValueError("scripted controller limits must be finite numbers")
        (
            backswing_x,
            follow_through_x,
            swing_trigger_x,
            swing_speed_mps,
            upward_tilt_degrees,
            minimum_height,
            maximum_height,
            maximum_lateral,
        ) = values
        if backswing_x >= follow_through_x:
            raise ValueError("backswing_x must be less than follow_through_x")
        if swing_speed_mps <= 0.0:
            raise ValueError("swing_speed_mps must be positive")
        if minimum_height > maximum_height:
            raise ValueError("minimum_height cannot exceed maximum_height")
        if maximum_lateral < 0.0:
            raise ValueError("maximum_lateral must be non-negative")
        self.backswing_x = backswing_x
        self.follow_through_x = follow_through_x
        self.swing_trigger_x = swing_trigger_x
        self.swing_speed_mps = swing_speed_mps
        self.minimum_height = minimum_height
        self.maximum_height = maximum_height
        self.maximum_lateral = maximum_lateral
        # Start with a +90 degree world-Z rotation so the blade's thin local Y
        # axis is normal to world X, then tilt the return face upward around
        # world Y.  The negative composition angle produces a +Z component on
        # the face seen by an incoming ball.
        half_tilt = radians(-upward_tilt_degrees) / 2.0
        half_turn = 1.0 / sqrt(2.0)
        self.paddle_quaternion: Quaternion = (
            half_turn * cos(half_tilt),
            half_turn * sin(half_tilt),
            half_turn * sin(half_tilt),
            half_turn * cos(half_tilt),
        )
        self._swing_started_at: float | None = None

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        del shot, seed
        self._swing_started_at = None

    def act(self, observation: ControllerObservation) -> PaddleCommand:
        ball_position = observation.ball.position
        ball_velocity = observation.ball.linear_velocity
        if (
            self._swing_started_at is None
            and float(ball_position[0]) <= self.swing_trigger_x
            and float(ball_velocity[0]) < 0.0
        ):
            self._swing_started_at = observation.time_s

        paddle_x = self.backswing_x
        if self._swing_started_at is not None:
            elapsed = max(0.0, observation.time_s - self._swing_started_at)
            paddle_x = min(
                self.follow_through_x,
                self.backswing_x + self.swing_speed_mps * elapsed,
            )
        lateral = min(self.maximum_lateral, max(-self.maximum_lateral, float(ball_position[1])))
        height = min(
            self.maximum_height,
            max(self.minimum_height, float(ball_position[2])),
        )
        return PaddleCommand(
            position=(paddle_x, lateral, height),
            quaternion=self.paddle_quaternion,
        )
