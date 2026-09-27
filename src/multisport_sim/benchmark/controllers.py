"""Backend-neutral controller contracts and MuJoCo reference-fixture controllers."""

from __future__ import annotations

import itertools
from bisect import bisect_left
from dataclasses import dataclass
from math import atan2, cos, isfinite, radians, sin, sqrt
from typing import Protocol, TypeVar, runtime_checkable

from .types import BallState, ShotSpec

Vec3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
ObservationT_contra = TypeVar("ObservationT_contra", contravariant=True)
ActionT_co = TypeVar("ActionT_co", covariant=True)


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
class Controller(Protocol[ObservationT_contra, ActionT_co]):
    """Minimal policy contract; backend adapters retain ownership of actions."""

    controller_id: str

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        ...

    def act(self, observation: ObservationT_contra) -> ActionT_co:
        ...


class NoOpController:
    """A deterministic controller that leaves the benchmark blade parked."""

    controller_id = "noop-v0"

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        del shot, seed

    def act(self, observation: object) -> None:
        """No action at all: the blade stays where the backend parked it."""
        del observation


SCRIPTED_DEFAULTS: dict[str, dict[str, float]] = {
    # Table tennis: the frozen values the v0 fixture was calibrated with.
    "table_tennis": {
        "backswing_x": -1.70,
        "follow_through_x": -1.38,
        "swing_trigger_x": -1.0,
        "swing_speed_mps": 3.0,
        "upward_tilt_degrees": 28.0,
        "minimum_height": 0.82,
        "maximum_height": 1.35,
        "maximum_lateral": 0.70,
    },
    # Tennis: the same swing an order of magnitude larger.  The strike happens
    # behind the baseline rather than at the net, the racket travels metres
    # instead of centimetres, and it must be moving four times faster because
    # the incoming ball is.
    "tennis": {
        "backswing_x": -11.30,
        "follow_through_x": -9.40,
        "swing_trigger_x": -5.0,
        "swing_speed_mps": 14.0,
        # Far flatter than table tennis.  A 20 m/s ball leaving a 20-degree face
        # is lofted, lands beyond the baseline or does not land inside the
        # three-second episode at all; the swing that returns a tennis ball into
        # the court is nearly horizontal.
        "upward_tilt_degrees": 14.0,
        "minimum_height": 0.35,
        "maximum_height": 2.20,
        "maximum_lateral": 4.00,
    },
}


def scripted_controller_for(sport: str) -> ScriptedPaddleController:
    """The mocap fixture tuned for one sport's scale.

    The control law is identical; only distances and speeds differ, because
    that is the only thing that differs between striking a 2.7 g ball across a
    2.74 m table and a 57 g ball across a 23.77 m court.
    """
    try:
        defaults = SCRIPTED_DEFAULTS[sport]
    except KeyError:
        known = ", ".join(sorted(SCRIPTED_DEFAULTS))
        raise ValueError(
            f"no scripted fixture is tuned for sport {sport!r}; known: {known}"
        ) from None
    return ScriptedPaddleController(**defaults)


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


SERVE_CARRY_TABLES: dict[float, tuple[tuple[float, float], ...]] = {
    # elevation degrees -> (racket speed m/s, horizontal carry m from the strike
    # point to the first landing), measured on the MuJoCo badminton fixture by
    # scripts/calibrate_serve.py (2026-09-28).  Above 24 m/s the carry stops
    # growing (8.10 m at 26 m/s, 45 degrees): the face is near the speed at
    # which it outruns the soft contact, so the tables stop there.
    45.0: (
        (8.0, 4.31),
        (10.0, 5.04),
        (12.0, 5.68),
        (14.0, 6.22),
        (16.0, 6.69),
        (18.0, 7.10),
        (20.0, 7.46),
        (22.0, 7.79),
        (24.0, 8.05),
    ),
    # A flatter serve carries further; used only when 45 degrees cannot reach.
    35.0: (
        (16.0, 6.97),
        (18.0, 7.44),
        (20.0, 7.86),
        (22.0, 8.24),
        (24.0, 8.54),
    ),
}


def _quaternion_taking_y_to(direction: tuple[float, float, float]) -> Quaternion:
    """``wxyz`` rotation that turns the face's local +y (its normal) onto ``direction``."""
    dx, dy, dz = direction
    # cross((0, 1, 0), d) = (dz, 0, -dx); dot = dy.
    w = 1.0 + dy
    if w <= 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    norm = sqrt(w * w + dz * dz + dx * dx)
    return (w / norm, dz / norm, 0.0, -dx / norm)


class ScriptedServeController:
    """Strike a released shuttle toward a service-court target with the mocap face.

    Privileged, like every reference fixture: it reads the shuttle state and the
    shot's target.  The swing is a straight line through the predicted contact
    point along the launch direction -- aimed at the target in yaw, fixed in
    elevation, with the speed looked up from a measured carry table -- so the
    only thing that varies between shots is where and how hard.

    The backend spreads each command over one control period, so every command
    is the pose the face should reach one period from now.
    """

    controller_id = "scripted-mocap-serve-v0"

    def __init__(
        self,
        *,
        strike_height_m: float = 0.85,
        elevation_degrees: float | None = None,
        backswing_m: float = 0.5,
        follow_through_m: float = 0.10,
        control_dt: float = 0.005,
        face_half_thickness_m: float = 0.025,
        cork_radius_m: float = 0.0135,
        cork_offset_m: float = 0.012,
        default_depth_m: float = 5.0,
        default_lateral_m: float = 1.3,
        carry_tables: dict[float, tuple[tuple[float, float], ...]] = SERVE_CARRY_TABLES,
        speed_mps: float | None = None,
    ) -> None:
        if speed_mps is not None and not (isfinite(speed_mps) and speed_mps > 0.0):
            raise ValueError("speed_mps must be a positive finite number")
        self.speed_mps = None if speed_mps is None else float(speed_mps)
        values = (
            strike_height_m,
            backswing_m,
            follow_through_m,
            control_dt,
            face_half_thickness_m,
            cork_radius_m,
            cork_offset_m,
            default_depth_m,
            default_lateral_m,
        )
        if not all(isinstance(value, (int, float)) and isfinite(value) for value in values):
            raise ValueError("serve controller parameters must be finite numbers")
        if strike_height_m <= 0.0 or backswing_m <= 0.0 or control_dt <= 0.0:
            raise ValueError("strike height, backswing and control_dt must be positive")
        tables: dict[float, tuple[tuple[float, float], ...]] = {}
        for elevation, rows in carry_tables.items():
            if not 0.0 < float(elevation) < 90.0:
                raise ValueError("carry table elevations must be between 0 and 90 degrees")
            table = tuple(sorted((float(speed), float(carry)) for speed, carry in rows))
            if len(table) < 2 or any(b[1] <= a[1] for a, b in itertools.pairwise(table)):
                raise ValueError(
                    "a carry table needs two or more rows with carry rising with speed"
                )
            tables[float(elevation)] = table
        if not tables:
            raise ValueError("at least one carry table is required")
        if elevation_degrees is not None:
            if not 0.0 < elevation_degrees < 90.0:
                raise ValueError("elevation_degrees must be between 0 and 90")
            if speed_mps is None and float(elevation_degrees) not in tables:
                raise ValueError("a fixed elevation needs its own carry table or a fixed speed")
        self.fixed_elevation = None if elevation_degrees is None else float(elevation_degrees)
        self.carry_tables = tables
        self.strike_height_m = float(strike_height_m)
        self.backswing_m = float(backswing_m)
        self.follow_through_m = float(follow_through_m)
        self.control_dt = float(control_dt)
        self.standoff_m = float(face_half_thickness_m) + float(cork_radius_m)
        self.cork_offset_m = float(cork_offset_m)
        self.default_depth_m = float(default_depth_m)
        self.default_lateral_m = float(default_lateral_m)
        self._target: tuple[float, float] | None = None
        self._idle = False
        self._plan: tuple[Vec3, Vec3, float, float, Quaternion] | None = None

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        del seed
        self._plan = None
        self._target = shot.target.center_xy if shot.target is not None else None
        # L0 is scored without robot action: it checks the release, and a
        # served shuttle never shows where a released one would have landed.
        self._idle = shot.level == "L0"

    def elevation_for_carry(self, carry_m: float) -> float:
        """The steepest calibrated elevation that reaches ``carry_m``.

        A high serve is a steep one; a flatter swing is used only when the
        steep one cannot reach, and the flattest table is the fallback.
        """
        if self.fixed_elevation is not None:
            return self.fixed_elevation
        for elevation in sorted(self.carry_tables, reverse=True):
            if carry_m <= self.carry_tables[elevation][-1][1]:
                return elevation
        return min(self.carry_tables)

    def speed_for_carry(self, carry_m: float, elevation: float | None = None) -> float:
        """Racket speed whose measured carry is ``carry_m``, clamped to the table."""
        table = self.carry_tables[
            self.elevation_for_carry(carry_m) if elevation is None else elevation
        ]
        speeds = [row[0] for row in table]
        carries = [row[1] for row in table]
        if carry_m <= carries[0]:
            return speeds[0]
        if carry_m >= carries[-1]:
            return speeds[-1]
        index = bisect_left(carries, carry_m)
        c0, c1 = carries[index - 1], carries[index]
        v0, v1 = speeds[index - 1], speeds[index]
        return v0 + (v1 - v0) * (carry_m - c0) / (c1 - c0)

    def _plan_swing(self, observation: ControllerObservation) -> None:
        x, y, z = observation.ball.position
        vx, vy, vz = observation.ball.linear_velocity
        cork_z = z - self.cork_offset_m
        # Fall to the strike height; drag is negligible below 2 m/s.
        drop = cork_z - self.strike_height_m
        fall = (vz + sqrt(max(vz * vz + 2.0 * 9.81 * max(drop, 0.0), 0.0))) / 9.81
        contact_time = observation.time_s + fall
        contact = (x + vx * fall, y + vy * fall, self.strike_height_m)
        if self._target is not None:
            target = self._target
        else:
            side = -1.0 if y > 0.0 else 1.0
            target = (self.default_depth_m, side * self.default_lateral_m)
        yaw = atan2(target[1] - contact[1], target[0] - contact[0])
        carry = sqrt((target[0] - contact[0]) ** 2 + (target[1] - contact[1]) ** 2)
        elevation_degrees = self.elevation_for_carry(carry)
        elevation = radians(elevation_degrees)
        direction = (
            cos(elevation) * cos(yaw),
            cos(elevation) * sin(yaw),
            sin(elevation),
        )
        speed = (
            self.speed_mps
            if self.speed_mps is not None
            else self.speed_for_carry(carry, elevation_degrees)
        )
        # The face centre sits one stand-off behind the cork when they meet.
        face_at_contact = tuple(
            contact[axis] - direction[axis] * self.standoff_m for axis in range(3)
        )
        self._plan = (
            face_at_contact,
            direction,
            contact_time,
            speed,
            _quaternion_taking_y_to(direction),
        )

    def act(self, observation: ControllerObservation) -> PaddleCommand | None:
        if self._idle:
            return None
        if self._plan is None:
            self._plan_swing(observation)
        assert self._plan is not None
        face, direction, contact_time, speed, quaternion = self._plan
        when = observation.time_s + self.control_dt
        travel = min(
            max(speed * (when - contact_time), -self.backswing_m), self.follow_through_m
        )
        position = tuple(face[axis] + direction[axis] * travel for axis in range(3))
        return PaddleCommand(position=position, quaternion=quaternion)
