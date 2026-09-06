"""Reference baselines for the embodied table-tennis return task.

Two baselines ship with the task and neither is a leaderboard entry; they exist
to bracket the score so a submission can be read against something:

* :class:`HoldPoseController` never moves.  It establishes the floor and, more
  usefully, proves the failure paths -- miss and timeout -- actually fire on an
  embodied task.
* :class:`RandomJointController` samples the declared action space.  It is the
  random baseline the specification asks for, and it is the one that exercises
  the safety envelope, because flailing an arm at full torque is precisely what
  the joint, velocity and collision limits exist to catch.
* :class:`ScriptedInterceptController` predicts where the ball crosses the
  strike plane, solves for a configuration that puts the blade there with its
  face toward the opponent, and swings through it.

The scripted controller uses privileged ball state, exactly like the mocap
fixture it replaces, and its scores must not be reported as robot-policy
results.  What it does *not* do is reach into the running simulation: it builds
its own model and its own solver, so every number it produces would be
obtainable by a real policy with a camera and a model of the arm.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, isfinite, radians, sin
from typing import Any

import numpy as np

from .ball_flight import BallFlightModel, PlaneCrossing, TableBounce
from .robot import ControlMode, JointCommand
from .robots import g1 as g1_module
from .robots.panda import (
    IK_SETTINGS,
    JOINT_NAMES,
    PADDLE_SITE,
    PANDA_READY_QPOS,
    PANDA_VELOCITY_LIMIT,
    PandaMount,
    build_panda_table_tennis_model,
)
from .rules.table_tennis import TABLE_TENNIS
from .types import ShotSpec

# Setpoints are advanced at a fraction of the datasheet limit.  A position
# actuator tracking a setpoint that moves at exactly the limit overshoots it
# transiently while closing the error, which the safety monitor would -- quite
# correctly -- record as a violation.  The margin is the controller's problem to
# leave, not the monitor's to forgive.
RATE_LIMIT_MARGIN = 0.8

# The scripted baseline runs closer to the limit than the random one, because
# blade speed at contact is exactly what decides whether its hit becomes a
# return.  The margin is still a margin: at 1.0 the position actuator's transient
# overshoot while closing the error trips the monitor.
SCRIPTED_RATE_MARGIN = 0.95


@dataclass(frozen=True)
class SwingRobot:
    """Everything the scripted swing needs to know about an embodiment.

    The swing itself -- predict the crossing, prepare behind and above it, drive
    the blade through it at the speed limit -- contains no robot-specific
    reasoning, so it should not contain robot-specific *names* either.  Putting
    them here is what let the second embodiment reuse the control law instead of
    forking it, and a forked control law would have made the two baselines
    incomparable within a release.
    """

    name: str
    controller_id: str
    build_model: Any
    mount_type: type
    paddle_site: str
    joint_names: tuple[str, ...]
    ik_settings: dict[str, Any]
    velocity_limit: tuple[float, ...]
    ready_qpos: tuple[float, ...]
    # How close to the declared speed limit this robot's setpoint may travel.
    # It is per-robot because it is a property of the servo, not of the swing:
    # see the two values below.
    rate_margin: float = SCRIPTED_RATE_MARGIN
    # Optional per-robot narrowing of the solver's search, and the posture the
    # joints outside the solved set are held at.  Both default to "nothing
    # special", which is the Panda.
    solver_bounds: Any = None
    context_qpos: Any = None


PANDA_SWING = SwingRobot(
    name="panda",
    controller_id="scripted-panda-intercept-v1",
    build_model=build_panda_table_tennis_model,
    mount_type=PandaMount,
    paddle_site=PADDLE_SITE,
    joint_names=JOINT_NAMES,
    ik_settings=IK_SETTINGS,
    velocity_limit=PANDA_VELOCITY_LIMIT,
    ready_qpos=PANDA_READY_QPOS,
)


def _g1_context(model: object) -> np.ndarray:
    """Full-model configuration with the G1's held joints at their stand pose."""
    import mujoco

    context = np.zeros(model.nq)  # type: ignore[union-attr]
    order = g1_module._model_joint_order(model)
    for name, value in zip(order, g1_module.STAND_QPOS, strict=False):
        joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        context[int(model.jnt_qposadr[joint])] = value  # type: ignore[union-attr]
    return context


# The G1's servo needs an order of magnitude more headroom than the Panda's.
# Measured on 60 training shots: at the Panda's 0.95 the wrist overshoots its
# declared speed envelope and 50 of 60 episodes end in a safety violation with
# zero hits; at 0.10 there are no violations and 58 hits.  It costs nothing,
# because this embodiment's worst-case travel time is an eighth of the shortest
# interception window -- the G1 has time it cannot spend.
G1_SCRIPTED_RATE_MARGIN = 0.10

G1_SWING = SwingRobot(
    name="g1",
    controller_id="scripted-g1-intercept-v1",
    rate_margin=G1_SCRIPTED_RATE_MARGIN,
    build_model=g1_module.build_g1_table_tennis_model,
    mount_type=g1_module.G1Mount,
    paddle_site=g1_module.PADDLE_SITE,
    joint_names=g1_module.JOINT_NAMES,
    ik_settings=g1_module.IK_SETTINGS,
    velocity_limit=(g1_module.G1_VELOCITY_LIMIT_RAD_S,) * len(g1_module.JOINT_NAMES),
    ready_qpos=g1_module.G1_READY_QPOS,
    solver_bounds=g1_module.solver_position_bounds,
    context_qpos=_g1_context,
)

SWING_ROBOTS = {robot.name: robot for robot in (PANDA_SWING, G1_SWING)}


class JointRateLimiter:
    """Advance a joint setpoint toward a goal without exceeding a speed limit.

    Both shipped baselines use this and both say so.  It matters that it lives
    in the controller rather than in the adapter: an adapter that silently
    slewed every command would make the safety envelope unfalsifiable, and a
    policy could then not be distinguished by whether it respects the hardware.
    A submission is free to omit it -- and to be scored on the safety failures
    that follow.
    """

    def __init__(
        self,
        velocity_limit: Sequence[float],
        *,
        control_dt: float,
        margin: float = RATE_LIMIT_MARGIN,
    ) -> None:
        self.velocity_limit = np.asarray(velocity_limit, dtype=float)
        if self.velocity_limit.ndim != 1 or np.any(self.velocity_limit <= 0.0):
            raise ValueError("velocity_limit must be a positive one-dimensional vector")
        if not isfinite(control_dt) or control_dt <= 0.0:
            raise ValueError("control_dt must be greater than zero")
        if not 0.0 < margin <= 1.0:
            raise ValueError("margin must lie in (0, 1]")
        self.control_dt = float(control_dt)
        self.margin = float(margin)
        self._setpoint: np.ndarray | None = None

    @property
    def max_step(self) -> np.ndarray:
        return self.velocity_limit * self.margin * self.control_dt

    @property
    def setpoint(self) -> np.ndarray:
        """The setpoint currently held; the origin any planned swing must use."""
        if self._setpoint is None:
            raise RuntimeError("the limiter has no setpoint until reset or advance")
        return self._setpoint.copy()

    def reset(self, setpoint: Sequence[float]) -> None:
        self._setpoint = np.asarray(setpoint, dtype=float).copy()

    def advance(self, goal: Sequence[float]) -> np.ndarray:
        """Step the held setpoint toward ``goal`` by at most one step's travel."""
        target = np.asarray(goal, dtype=float)
        if self._setpoint is None:
            self._setpoint = target.copy()
            return self._setpoint.copy()
        delta = np.clip(target - self._setpoint, -self.max_step, self.max_step)
        self._setpoint = self._setpoint + delta
        return self._setpoint.copy()


class HoldPoseController:
    """Stay in the ready pose for the whole episode.

    The command is the pose itself rather than ``None``: a position-controlled
    arm handed no command holds its last setpoint, but making the hold explicit
    means the baseline still exercises the full command path.
    """

    controller_id = "hold-pose-v1"

    def __init__(self, ready_qpos: Sequence[float] = PANDA_READY_QPOS) -> None:
        self.ready_qpos = tuple(float(value) for value in ready_qpos)

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        del shot, seed

    def act(self, observation: object) -> JointCommand:
        del observation
        return JointCommand(self.ready_qpos, ControlMode.JOINT_POSITION)


class RandomJointController:
    """Sample a joint-position setpoint uniformly from the declared limits.

    Resampling every control step would produce a 200 Hz noise command that no
    arm tracks, measuring the actuator's low-pass response rather than the
    policy.  The controller therefore holds each sample for ``hold_steps``,
    which is a documented property of the baseline, not a hidden smoothing hack.
    """

    controller_id = "random-joint-v1"

    def __init__(
        self,
        low: Sequence[float],
        high: Sequence[float],
        *,
        control_dt: float,
        hold_steps: int = 20,
        seed: int | None = None,
        ready_qpos: Sequence[float] = PANDA_READY_QPOS,
        velocity_limit: Sequence[float] = PANDA_VELOCITY_LIMIT,
    ) -> None:
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        if self.low.shape != self.high.shape or self.low.ndim != 1:
            raise ValueError("low and high must be one-dimensional and equally sized")
        if np.any(self.low >= self.high):
            raise ValueError("low must be strictly below high")
        if hold_steps < 1:
            raise ValueError("hold_steps must be at least one")
        self.hold_steps = int(hold_steps)
        self.ready_qpos = tuple(float(value) for value in ready_qpos)
        self._limiter = JointRateLimiter(velocity_limit, control_dt=control_dt)
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._goal = np.asarray(self.ready_qpos, dtype=float)

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        del shot
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        self._limiter.reset(self.ready_qpos)
        self._goal = self._sample()

    def _sample(self) -> np.ndarray:
        return self._rng.uniform(self.low, self.high)

    def act(self, observation: object) -> JointCommand:
        del observation
        if self._step % self.hold_steps == 0:
            self._goal = self._sample()
        self._step += 1
        setpoint = self._limiter.advance(self._goal)
        return JointCommand(
            tuple(float(value) for value in setpoint), ControlMode.JOINT_POSITION
        )


class ScriptedInterceptController:
    """Predict where the ball crosses the strike plane and swing through it.

    The controller is built around one measured fact: a Panda's blade tops out
    near 1.6-2.7 m/s at the strike points, and a legal return needs the ball to
    leave at 3.4-6.4 m/s.  With the calibrated rubber contact that is reachable
    -- ``v_out ~= 1.85 * v_blade + 0.85 * v_in`` -- but only if the blade is
    genuinely moving at contact.  A controller that servos to the intercept
    point and waits there arrives with almost no speed and returns the ball into
    its own half.

    So the swing is planned, not tracked:

    1. **Predict** where and when the ball reaches ``plane_x``, by integrating
       :class:`~.ball_flight.BallFlightModel` -- drag, Magnus and the table
       bounce -- and re-evaluating every control step while the arm approaches.
       The bounce is not optional: above L1 most shots bounce *after* the swing
       commits, and a free-flight prediction is then wrong by a median half
       metre, which lands the blade 8-11 cm from a ball it had time to reach.
    2. **Solve two configurations** -- one a backswing behind the intercept
       point, one a follow-through in front of it -- both with the blade face
       tilted up toward the opponent.
    3. **Time the swing** so the fastest joint runs at the datasheet limit, then
       play it open loop so that contact falls in its middle.

    Playing the swing open loop is deliberate.  Re-planning during the strike
    would let the setpoint chase a moving target through the one interval where
    speed is what matters, and a real swing is ballistic for the same reason.

    Privileged ball state is used, exactly like the mocap fixture this replaces,
    so its scores are not robot-policy results.  What it does not do is touch the
    running simulation: the model and the solver are its own, so a real policy
    with a camera and a model of the arm could reproduce every number it makes.
    """

    def __init__(
        self,
        *,
        control_dt: float,
        robot: SwingRobot = PANDA_SWING,
        mount: Any = None,
        plane_x: float = -1.55,
        backswing_m: float = 0.10,
        follow_through_m: float = 0.45,
        swing_lead_s: float = 0.16,
        clearance_x_m: float = -1.42,
        approach_floor_z_m: float = 1.00,
        blade_tilt_deg: float = 24.0,
        ready_qpos: Sequence[float] | None = None,
        # Long enough to cover the whole bank: the slowest shot in
        # ``table_tennis/return-v1`` takes 1.577 s to reach the strike plane, and
        # a horizon shorter than that leaves the arm parked until it is late.
        max_horizon_s: float = 1.8,
        flight: BallFlightModel | None = None,
        bounce: TableBounce | None = None,
        rate_margin: float | None = None,
    ) -> None:
        from .robots.kinematics import IKSolver

        self.robot = robot
        self.controller_id = robot.controller_id
        rate_margin = robot.rate_margin if rate_margin is None else rate_margin
        self.model = robot.build_model(mount=mount or robot.mount_type())
        bounds = (
            robot.solver_bounds(self.model) if robot.solver_bounds is not None else None
        )
        self.solver = IKSolver(
            self.model,
            site_name=robot.paddle_site,
            joint_names=robot.joint_names,
            **({} if bounds is None else {"position_low": bounds[0], "position_high": bounds[1]}),
            **robot.ik_settings,
        )
        self.context_qpos = (
            robot.context_qpos(self.model) if robot.context_qpos is not None else None
        )
        self.control_dt = float(control_dt)
        self.plane_x = float(plane_x)
        self.backswing_m = float(backswing_m)
        self.follow_through_m = float(follow_through_m)
        self.swing_lead_s = float(swing_lead_s)
        self.clearance_x_m = float(clearance_x_m)
        self.approach_floor_z_m = float(approach_floor_z_m)
        self.max_horizon_s = float(max_horizon_s)
        self.flight = flight if flight is not None else BallFlightModel()
        self.bounce = bounce if bounce is not None else TableBounce.from_surface(TABLE_TENNIS)
        self.rate_margin = float(rate_margin)
        # A vertical blade returns the ball horizontally, and a ball leaving at
        # table height with no upward component drops back onto the robot's own
        # half before it reaches the net.  Tilting the face up is what turns a
        # hit into a return.
        tilt = radians(float(blade_tilt_deg))
        self.strike_axis = (cos(tilt), 0.0, sin(tilt))
        self.ready_qpos = tuple(
            float(value)
            for value in (robot.ready_qpos if ready_qpos is None else ready_qpos)
        )
        self._velocity_limit = np.asarray(robot.velocity_limit, dtype=float)
        self._limiter = JointRateLimiter(
            robot.velocity_limit, control_dt=control_dt, margin=rate_margin
        )
        self._plan = np.asarray(self.ready_qpos, dtype=float)

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        del shot, seed
        self._limiter.reset(self.ready_qpos)
        self._plan = np.asarray(self.ready_qpos, dtype=float)

    def _predict(
        self,
        position: Sequence[float],
        velocity: Sequence[float],
        angular_velocity: Sequence[float] | None,
    ) -> PlaneCrossing | None:
        """Where and when the ball reaches ``plane_x``, or ``None`` if it will not."""
        return self.flight.predict_plane_crossing(
            position,
            velocity,
            plane_x=self.plane_x,
            angular_velocity=angular_velocity,
            horizon_s=self.max_horizon_s,
            bounce=self.bounce,
        )

    def _swing_duration(self, delta: np.ndarray) -> float:
        """Shortest time the arm may take for this swing, at its speed limit."""
        travel = float(np.max(np.abs(delta) / (self._velocity_limit * self.rate_margin)))
        # One control step is the floor: a swing shorter than the command period
        # cannot be expressed.
        return max(travel, self.control_dt)

    def act(self, observation: object) -> JointCommand:
        ball = getattr(observation, "ball", None)
        robot = getattr(observation, "robot", None)
        time_s = getattr(observation, "time_s", None)
        if ball is None or robot is None or time_s is None:
            raise TypeError(
                "ScriptedInterceptController requires an observation exposing "
                "'time_s', 'ball' and 'robot'"
            )
        return self.plan_swing(
            ball.position,
            ball.linear_velocity,
            angular_velocity=getattr(ball, "angular_velocity", None),
        )

    def plan_swing(
        self,
        position: Sequence[float],
        velocity: Sequence[float],
        angular_velocity: Sequence[float] | None = None,
    ) -> JointCommand:
        """The control law itself, given *some* estimate of the ball.

        It is separated from :meth:`act` so the vision baseline can drive the
        identical swing from a triangulated estimate.  That is the whole point
        of the comparison: two tracks, one controller, and any score difference
        is perception.

        ``angular_velocity`` is optional because the two tracks genuinely differ
        in what they can see: the state track reads spin from the observation,
        the vision tracker recovers position and velocity only.  Omitting it
        drops the Magnus term, which on this bank costs about a hundredth of a
        centimetre of prediction error -- measured, not assumed, so the vision
        result is not quietly credited with information it never had.
        """
        crossing = self._predict(position, velocity, angular_velocity)
        if crossing is None:
            return self._hold()

        horizon = crossing.time_s
        y, z = crossing.y, crossing.z

        if horizon > self.swing_lead_s:
            # Approach: wait behind and above the intercept point.  Joint
            # interpolation between two configurations traces a curve, not a
            # line, and aimed straight at a low intercept point that curve dips
            # into the table edge -- which the collision limit correctly calls a
            # safety failure.  Preparing high and back is what a player does for
            # the same reason.
            target = (self.plane_x - self.backswing_m, y, max(z, self.approach_floor_z_m))
        else:
            # Strike: drive the blade forward through the ball, stopping short
            # of the table edge.  Meeting the ball at rest returns it into the
            # robot's own half; only blade speed converts a hit into a return.
            progress = 1.0 - max(horizon, 0.0) / self.swing_lead_s
            x = min(
                self.plane_x + self.follow_through_m * progress, self.clearance_x_m
            )
            target = (x, y, z)

        # Seed from the previous plan.  With seven joints and five constraints
        # the solver is free to wander the null space; seeding from a lagging
        # measurement lets it pick a different redundant branch every step, and
        # the setpoint then chases a goal that keeps moving.
        result = self.solver.solve(
            target,
            target_axis=self.strike_axis,
            initial_qpos=self._plan,
            context_qpos=self.context_qpos,
        )
        self._plan = result.qpos.copy()
        setpoint = self._limiter.advance(self._plan)
        return JointCommand(
            tuple(float(value) for value in setpoint), ControlMode.JOINT_POSITION
        )

    def _hold(self) -> JointCommand:
        """Hold the last plan rather than snapping back to the ready pose."""
        setpoint = self._limiter.advance(self._plan)
        return JointCommand(
            tuple(float(value) for value in setpoint), ControlMode.JOINT_POSITION
        )


class VisionInterceptController(ScriptedInterceptController):
    """The scripted swing, driven by two cameras instead of by the truth.

    Nothing about the arm, the solver or the swing changes.  The only
    difference is where the ball state comes from: a colour-segmentation and
    triangulation pipeline over the declared stereo pair, at the cameras' own
    120 Hz rather than the 200 Hz control rate.  Whatever this scores below the
    state-track baseline is the cost of perception, measured rather than
    assumed.

    It holds its last plan until the tracker has two frames.  Acting on a
    single detection would mean swinging at a velocity estimate that does not
    exist yet.
    """

    def __init__(self, *, tracker: object | None = None, **kwargs: object) -> None:
        from .vision import StereoBallTracker

        super().__init__(**kwargs)  # type: ignore[arg-type]
        # After ``super().__init__``, which sets the state-track id from the
        # robot binding.  A class attribute here would be shadowed by that
        # instance attribute and every vision result would be filed under the
        # scripted baseline's name.
        self.controller_id = f"vision-{self.robot.name}-intercept-v1"
        self.tracker = tracker if tracker is not None else StereoBallTracker()

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        super().reset(shot, seed=seed)
        self.tracker.reset()

    def act(self, observation: object) -> JointCommand:
        readings = getattr(observation, "sensors", None)
        if readings is None:
            raise TypeError(
                "VisionInterceptController requires a vision-track observation "
                "exposing 'sensors'; it must not be given privileged ball state"
            )
        self.tracker.detect(readings)
        estimate = self.tracker.estimate()
        if estimate is None:
            return self._hold()
        position, velocity = estimate
        return self.plan_swing(position, velocity)


__all__ = [
    "HoldPoseController",
    "RandomJointController",
    "ScriptedInterceptController",
    "VisionInterceptController",
]
