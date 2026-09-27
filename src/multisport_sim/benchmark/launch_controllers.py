"""Scripted mocap fixtures for the kick and shoot launch tasks.

Both work the same way.  At the first control step they pick a contact time a
fixed lead ahead, predict where the ball will be and how it will be moving
then, decide the launch velocity they want (by integrating the backend's own
flight model -- drag included -- toward the target), and invert the measured
contact law to get the face's normal and speed.  The face then travels a
straight line through the contact point at that speed.

The contact law was measured on both fixtures (``tests/test_launch_tasks.py``
pins it): the impulse a face delivers is along its normal, and

    v_out = v_in + IMPULSE_GAIN * (u - v_in . n) * n

for face speed ``u`` along normal ``n``.  Solving it for a wanted ``v_out``
gives ``n`` parallel to ``v_out - v_in`` and ``u = |v_out - v_in| / gain +
v_in . n``.

The contact is not instantaneous: the face pushes the ball for 7-10 ms, and
the ball leaves at ``v_out`` from a point about ``v_out * tau`` beyond the
contact point (measured: 7.5 cm at 7.6 m/s and 9.4 cm at 9.6 m/s for the
basketball plate).  Planning as if it left from the contact point sent every
shot 10-15 cm long, onto the back of the rim.

Like every reference fixture these read privileged ball state and the shot's
target; they bound what the fixture can do, not what a robot could.
"""

from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt

from ..specs import BALLS, Sport
from .ball_flight import GRAVITY, BallFlightModel
from .controllers import ControllerObservation, PaddleCommand, _quaternion_taking_y_to
from .rules.basketball import BASKET, BasketSpec
from .rules.football import FOOTBALL_GOAL, FootballGoalSpec
from .types import ShotSpec

IMPULSE_GAIN = 1.21
"""Measured ``|delta v| / relative normal approach speed`` of both fixtures."""

Vec3 = tuple[float, float, float]


def _norm(v: Vec3) -> float:
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


class _ScriptedLaunch:
    """Plan once, then drive the face along a straight line through contact."""

    controller_id = "scripted-launch-v0"

    def __init__(
        self,
        *,
        sport: Sport,
        lead_s: float,
        face_half_thickness_m: float,
        backswing_m: float = 0.4,
        follow_through_m: float = 0.10,
        control_dt: float = 0.005,
        impulse_gain: float = IMPULSE_GAIN,
        max_face_speed_mps: float = 24.0,
        release_delay_s: float = 0.0,
    ) -> None:
        if min(lead_s, face_half_thickness_m, backswing_m, control_dt, impulse_gain) <= 0.0:
            raise ValueError("launch fixture parameters must be positive")
        self.sport = sport
        self.ball_radius_m = float(BALLS[sport].radius)
        self.flight = BallFlightModel(sport)
        self.lead_s = float(lead_s)
        self.standoff_m = float(face_half_thickness_m) + self.ball_radius_m
        self.backswing_m = float(backswing_m)
        self.follow_through_m = float(follow_through_m)
        self.control_dt = float(control_dt)
        self.impulse_gain = float(impulse_gain)
        self.max_face_speed_mps = float(max_face_speed_mps)
        self.release_delay_s = float(release_delay_s)
        self._target: tuple[float, float] | None = None
        self._idle = False
        self._plan: tuple[Vec3, Vec3, float, float, tuple[float, ...]] | None = None
        self.planned_launch: Vec3 | None = None

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        del seed
        self._plan = None
        self.planned_launch = None
        self._target = shot.target.center_xy if shot.target is not None else None
        # L0 is scored without robot action.
        self._idle = shot.level == "L0"

    # -- sport hooks --------------------------------------------------------

    def _ball_at(self, observation: ControllerObservation, dt: float) -> tuple[Vec3, Vec3]:
        raise NotImplementedError

    def _launch_velocity(self, contact: Vec3) -> Vec3:
        raise NotImplementedError

    # -- the shared plan ----------------------------------------------------

    def _plan_strike(self, observation: ControllerObservation) -> None:
        contact, incoming = self._ball_at(observation, self.lead_s)
        # The ball leaves from beyond the contact point; where depends on the
        # launch velocity, which depends on where it leaves from.  Three
        # fixed-point passes settle it to well under a millimetre.
        release = contact
        for _ in range(3):
            wanted = self._launch_velocity(release)
            release = tuple(
                contact[axis] + wanted[axis] * self.release_delay_s for axis in range(3)
            )
        self.planned_launch = wanted
        delta = tuple(wanted[axis] - incoming[axis] for axis in range(3))
        size = _norm(delta)
        if size < 1e-9:
            normal = (1.0, 0.0, 0.0)
            speed = 0.0
        else:
            normal = tuple(value / size for value in delta)
            approach = sum(incoming[axis] * normal[axis] for axis in range(3))
            speed = size / self.impulse_gain + approach
        speed = min(max(speed, 0.5), self.max_face_speed_mps)
        face = tuple(contact[axis] - normal[axis] * self.standoff_m for axis in range(3))
        self._plan = (
            face,
            normal,
            observation.time_s + self.lead_s,
            speed,
            _quaternion_taking_y_to(normal),
        )

    def act(self, observation: ControllerObservation) -> PaddleCommand | None:
        if self._idle:
            return None
        if self._plan is None:
            self._plan_strike(observation)
        assert self._plan is not None
        face, normal, contact_time, speed, quaternion = self._plan
        when = observation.time_s + self.control_dt
        travel = min(
            max(speed * (when - contact_time), -self.backswing_m), self.follow_through_m
        )
        return PaddleCommand(
            position=tuple(face[axis] + normal[axis] * travel for axis in range(3)),
            quaternion=quaternion,
        )


class ScriptedKickController(_ScriptedLaunch):
    """Kick a grounded ball through a point of the goal mouth.

    The aim point is the shot's target ``(y, z)`` in the goal mouth, or the
    middle of the goal at 1 m.  Ball speed grows with distance, 14-26 m/s; the
    elevation is solved against the drag model so the ball crosses the line at
    the aim height.
    """

    controller_id = "scripted-mocap-kick-v0"

    def __init__(
        self,
        *,
        goal: FootballGoalSpec = FOOTBALL_GOAL,
        default_aim: tuple[float, float] = (0.0, 1.0),
        **kwargs: float,
    ) -> None:
        kwargs.setdefault("lead_s", 0.30)
        kwargs.setdefault("face_half_thickness_m", 0.030)
        # Measured, not assumed: planning the kick from the contact point puts
        # it within 1 cm of the aim point, and the plate's 6 ms push model
        # made it worse.  The ball rolls on the ground during the push.
        kwargs.setdefault("release_delay_s", 0.0)
        super().__init__(sport=Sport.FOOTBALL, **kwargs)
        self.goal = goal
        self.default_aim = default_aim

    def _ball_at(self, observation: ControllerObservation, dt: float) -> tuple[Vec3, Vec3]:
        # Rolling resistance on grass is small over a third of a second.
        x, y, z = observation.ball.position
        vx, vy, _ = observation.ball.linear_velocity
        return (x + vx * dt, y + vy * dt, z), (vx, vy, 0.0)

    def launch_speed(self, distance_m: float) -> float:
        return min(max(12.0 + 0.5 * distance_m, 14.0), 26.0)

    def _launch_velocity(self, contact: Vec3) -> Vec3:
        aim_y, aim_z = self._target if self._target is not None else self.default_aim
        line = self.goal.ball_radius_m
        dx, dy = line - contact[0], aim_y - contact[1]
        yaw = atan2(dy, dx)
        speed = self.launch_speed(sqrt(dx * dx + dy * dy))

        def height_at_line(elevation: float) -> float | None:
            velocity = (
                speed * cos(elevation) * cos(yaw),
                speed * cos(elevation) * sin(yaw),
                speed * sin(elevation),
            )
            crossing = self.flight.fly_until(
                contact, velocity, lambda p, v: p[0] - line, horizon_s=4.0
            )
            return None if crossing is None else crossing[1][2]

        # Height at the line rises with elevation over this range; bisect.
        low, high = 0.0, radians(35.0)
        for _ in range(30):
            middle = 0.5 * (low + high)
            height = height_at_line(middle)
            if height is None or height > aim_z:
                high = middle
            else:
                low = middle
        elevation = 0.5 * (low + high)
        return (
            speed * cos(elevation) * cos(yaw),
            speed * cos(elevation) * sin(yaw),
            speed * sin(elevation),
        )


class ScriptedShootController(_ScriptedLaunch):
    """Strike a released ball on an arc that comes down through the rim.

    The launch angle steepens as the shot shortens -- 65 degrees at 2 m, 52 at
    6 m and beyond -- because a close shot on a flat arc meets the front of the
    ring (measured: at 2-2.5 m, 55 degrees made 0 of 2 and 65 made 2 of 2).
    The speed is then solved against the drag model so the ball's centre comes
    down through the rim's plane at the aim point: the shot's target, or the
    rim's centre.
    """

    controller_id = "scripted-mocap-shoot-v0"

    def __init__(
        self,
        *,
        basket: BasketSpec = BASKET,
        launch_elevation_degrees: float | None = None,
        **kwargs: float,
    ) -> None:
        kwargs.setdefault("lead_s", 0.15)
        kwargs.setdefault("face_half_thickness_m", 0.030)
        kwargs.setdefault("max_face_speed_mps", 12.0)
        kwargs.setdefault("release_delay_s", 0.010)
        super().__init__(sport=Sport.BASKETBALL, **kwargs)
        self.basket = basket
        self.fixed_elevation = (
            None if launch_elevation_degrees is None else radians(launch_elevation_degrees)
        )

    def elevation_for(self, distance_m: float) -> float:
        if self.fixed_elevation is not None:
            return self.fixed_elevation
        return radians(min(66.0, max(52.0, 72.0 - 3.5 * distance_m)))

    def _ball_at(self, observation: ControllerObservation, dt: float) -> tuple[Vec3, Vec3]:
        x, y, z = observation.ball.position
        vx, vy, vz = observation.ball.linear_velocity
        return (
            (x + vx * dt, y + vy * dt, z + vz * dt - 0.5 * GRAVITY * dt * dt),
            (vx, vy, vz - GRAVITY * dt),
        )

    def _launch_velocity(self, contact: Vec3) -> Vec3:
        aim_x, aim_y = self._target if self._target is not None else (0.0, 0.0)
        rim = self.basket.rim_height_m
        dx, dy = aim_x - contact[0], aim_y - contact[1]
        distance = sqrt(dx * dx + dy * dy)
        yaw = atan2(dy, dx)
        elevation = self.elevation_for(distance)
        c, s = cos(elevation), sin(elevation)

        def reach(speed: float) -> float | None:
            velocity = (speed * c * cos(yaw), speed * c * sin(yaw), speed * s)
            crossing = self.flight.fly_until(
                contact,
                velocity,
                lambda p, v: (rim - p[2]) if v[2] < 0.0 else -1.0,
                horizon_s=4.0,
            )
            if crossing is None:
                return None
            px, py, _ = crossing[1]
            return sqrt((px - contact[0]) ** 2 + (py - contact[1]) ** 2)

        low, high = 2.0, 16.0
        for _ in range(30):
            middle = 0.5 * (low + high)
            travelled = reach(middle)
            if travelled is None or travelled < distance:
                low = middle
            else:
                high = middle
        speed = 0.5 * (low + high)
        return (speed * c * cos(yaw), speed * c * sin(yaw), speed * s)


__all__ = ["IMPULSE_GAIN", "ScriptedKickController", "ScriptedShootController"]
