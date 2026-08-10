"""Backend-neutral task configuration shared by every shot-skill implementation.

A task configuration is the single source of truth for everything a backend must
agree on before its episodes can be compared: the task frame and axis
convention, the rule geometry handed to the judge, the control rate and episode
timeout, the declared action/observation ranges, and the benchmark reward terms.
MuJoCo and Isaac implementations differ in their simulator world frame and in
how an action reaches the hardware; they must not differ in any of the values
below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite
from numbers import Real
from typing import Any, Literal, TypeAlias

from ..specs import CAMPUS_OFFSETS, Sport
from .rules.table_tennis import TABLE_TENNIS as TABLE_TENNIS_RULES
from .rules.table_tennis import TableTennisTableSpec
from .types import BallState, EpisodeResult, SemanticContact, Vec3

Axis: TypeAlias = Literal["x", "y", "z"]


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _positive(value: object, *, field: str) -> float:
    result = _finite(value, field=field)
    if result <= 0.0:
        raise ValueError(f"{field} must be greater than zero")
    return result


def _vec3(value: object, *, field: str) -> Vec3:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must contain exactly three finite numbers")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{field} must contain exactly three finite numbers") from exc
    if len(items) != 3:
        raise ValueError(f"{field} must contain exactly three finite numbers")
    x, y, z = (
        _finite(item, field=f"{field}[{index}]") for index, item in enumerate(items)
    )
    return (x, y, z)


@dataclass(frozen=True)
class CoordinateConvention:
    """Axis semantics every backend must reproduce in its task frame.

    The convention is deliberately narrow: it fixes which world axis carries the
    playing-surface length, which carries its width, which points up, and which
    side of the net belongs to the robot.  ``quaternion_order`` documents the
    component order used by every pose in this package; MuJoCo mocap poses and
    Isaac Lab root states both use ``wxyz``, so an adapter that speaks ``xyzw``
    must convert at its own boundary.
    """

    length_axis: Axis = "x"
    width_axis: Axis = "y"
    up_axis: Axis = "z"
    robot_side_sign: int = -1
    quaternion_order: Literal["wxyz", "xyzw"] = "wxyz"
    length_unit: str = "m"
    angle_unit: str = "rad"

    def __post_init__(self) -> None:
        axes = (self.length_axis, self.width_axis, self.up_axis)
        if sorted(axes) != ["x", "y", "z"]:
            raise ValueError("length, width and up axes must be a permutation of x/y/z")
        if isinstance(self.robot_side_sign, bool) or self.robot_side_sign not in (-1, 1):
            raise ValueError("robot_side_sign must be -1 or 1")
        if self.quaternion_order not in ("wxyz", "xyzw"):
            raise ValueError("quaternion_order must be 'wxyz' or 'xyzw'")
        for name in ("length_unit", "angle_unit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    @property
    def opponent_side_sign(self) -> int:
        return -self.robot_side_sign

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_axis": self.length_axis,
            "width_axis": self.width_axis,
            "up_axis": self.up_axis,
            "robot_side_sign": self.robot_side_sign,
            "quaternion_order": self.quaternion_order,
            "length_unit": self.length_unit,
            "angle_unit": self.angle_unit,
        }


CANONICAL_CONVENTION = CoordinateConvention()


@dataclass(frozen=True)
class TaskFrame:
    """Placement of the task frame inside one backend's world frame.

    Shot banks, judges, targets and reports are expressed in the task frame,
    whose origin sits at the centre of the table's playing surface projected to
    the floor.  A backend that renders the same table somewhere else in its
    world -- the Isaac campus lays five sports out side by side -- declares that
    placement here and converts at its own boundary, so no rule code ever needs
    a backend-specific offset.

    The transform is a pure translation because both current backends share
    :data:`CANONICAL_CONVENTION`; a rotated placement is rejected rather than
    silently mishandled.
    """

    origin_xyz: Vec3 = (0.0, 0.0, 0.0)
    convention: CoordinateConvention = CANONICAL_CONVENTION

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_xyz", _vec3(self.origin_xyz, field="origin_xyz"))
        if not isinstance(self.convention, CoordinateConvention):
            raise TypeError("convention must be a CoordinateConvention")
        if self.convention != CANONICAL_CONVENTION:
            raise ValueError(
                "task frames currently support translation only; the axis "
                "convention must match CANONICAL_CONVENTION"
            )

    @classmethod
    def for_campus(cls, sport: Sport | str) -> TaskFrame:
        """Return the frame of one sport inside the shared multi-sport campus."""
        selected = Sport(sport)
        x, y = CAMPUS_OFFSETS[selected]
        return cls(origin_xyz=(x, y, 0.0))

    @property
    def is_identity(self) -> bool:
        return self.origin_xyz == (0.0, 0.0, 0.0)

    def to_task_position(self, position: Vec3) -> Vec3:
        world = _vec3(position, field="position")
        return (
            world[0] - self.origin_xyz[0],
            world[1] - self.origin_xyz[1],
            world[2] - self.origin_xyz[2],
        )

    def to_world_position(self, position: Vec3) -> Vec3:
        task = _vec3(position, field="position")
        return (
            task[0] + self.origin_xyz[0],
            task[1] + self.origin_xyz[1],
            task[2] + self.origin_xyz[2],
        )

    def to_task_vector(self, vector: Vec3) -> Vec3:
        """Return a free vector unchanged; a translation cannot rotate it."""
        return _vec3(vector, field="vector")

    to_world_vector = to_task_vector

    def to_task_ball(self, ball: BallState) -> BallState:
        if not isinstance(ball, BallState):
            raise TypeError("ball must be a BallState")
        if self.is_identity:
            return ball
        return BallState(
            position=self.to_task_position(ball.position),
            linear_velocity=ball.linear_velocity,
            angular_velocity=ball.angular_velocity,
        )

    def to_world_ball(self, ball: BallState) -> BallState:
        if not isinstance(ball, BallState):
            raise TypeError("ball must be a BallState")
        if self.is_identity:
            return ball
        return BallState(
            position=self.to_world_position(ball.position),
            linear_velocity=ball.linear_velocity,
            angular_velocity=ball.angular_velocity,
        )

    def to_task_contact(self, contact: SemanticContact) -> SemanticContact:
        """Move a contact sample into the task frame, keeping its normal."""
        if not isinstance(contact, SemanticContact):
            raise TypeError("contact must be a SemanticContact")
        if self.is_identity or contact.position is None:
            return contact
        return SemanticContact(
            contact.pair,
            position=self.to_task_position(contact.position),
            normal=contact.normal,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_xyz": list(self.origin_xyz),
            "convention": self.convention.to_dict(),
        }


@dataclass(frozen=True)
class PaddleWorkspace:
    """Declared position range of the end-effector pose command, in metres.

    These bounds define the published action space.  They are a task-level
    envelope, not a robot safety limit: a robot adapter is expected to clamp
    further to its own reachable and safe workspace.
    """

    position_low: Vec3 = (-2.0, -1.5, 0.5)
    position_high: Vec3 = (0.0, 1.5, 1.8)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position_low", _vec3(self.position_low, field="position_low")
        )
        object.__setattr__(
            self, "position_high", _vec3(self.position_high, field="position_high")
        )
        if any(
            low >= high for low, high in zip(self.position_low, self.position_high)
        ):
            raise ValueError("position_low must be strictly below position_high")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_low": list(self.position_low),
            "position_high": list(self.position_high),
        }


@dataclass(frozen=True)
class BallObservationLimits:
    """Declared observation range of the ball state."""

    position_low: Vec3 = (-5.0, -5.0, -1.0)
    position_high: Vec3 = (5.0, 5.0, 5.0)
    linear_velocity_limit: float = 50.0
    angular_velocity_limit: float = 2000.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position_low", _vec3(self.position_low, field="position_low")
        )
        object.__setattr__(
            self, "position_high", _vec3(self.position_high, field="position_high")
        )
        if any(
            low >= high for low, high in zip(self.position_low, self.position_high)
        ):
            raise ValueError("position_low must be strictly below position_high")
        object.__setattr__(
            self,
            "linear_velocity_limit",
            _positive(self.linear_velocity_limit, field="linear_velocity_limit"),
        )
        object.__setattr__(
            self,
            "angular_velocity_limit",
            _positive(self.angular_velocity_limit, field="angular_velocity_limit"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_low": list(self.position_low),
            "position_high": list(self.position_high),
            "linear_velocity_limit": self.linear_velocity_limit,
            "angular_velocity_limit": self.angular_velocity_limit,
        }


@dataclass(frozen=True)
class RewardWeights:
    """Benchmark reward terms, awarded once per episode event.

    The benchmark score is decided by the judge's event sequence, never by this
    reward.  Training code may add dense shaping on top; a submission is still
    scored from :class:`~multisport_sim.benchmark.types.EpisodeResult`.
    """

    hit: float = 1.0
    valid_return: float = 3.0
    target_hit: float = 1.0

    def __post_init__(self) -> None:
        for name in ("hit", "valid_return", "target_hit"):
            value = _finite(getattr(self, name), field=name)
            if value < 0.0:
                raise ValueError(f"{name} reward weight must be non-negative")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, float]:
        return {
            "hit": self.hit,
            "valid_return": self.valid_return,
            "target_hit": self.target_hit,
        }


@dataclass(frozen=True)
class TableTennisReturnTaskConfig:
    """Everything a table-tennis return backend must agree on.

    ``control_hz`` and ``timeout_s`` are task semantics, so a backend derives
    its decimation and step budget from them via :meth:`decimation` and
    :meth:`max_control_steps` instead of hard-coding a physics timestep.
    """

    task_id: str = "table-tennis-return-v0"
    env_id: str = "MultiSportRobot/TableTennisReturn-v0"
    sport: str = Sport.TABLE_TENNIS.value
    split: str = "dev"
    control_hz: float = 200.0
    timeout_s: float = 2.0
    frame: TaskFrame = field(default_factory=TaskFrame)
    table: TableTennisTableSpec = TABLE_TENNIS_RULES
    paddle_workspace: PaddleWorkspace = field(default_factory=PaddleWorkspace)
    ball_limits: BallObservationLimits = field(default_factory=BallObservationLimits)
    reward: RewardWeights = field(default_factory=RewardWeights)

    ACTION_DIM = 7
    OBSERVATION_DIM = 16

    def __post_init__(self) -> None:
        for name in ("task_id", "env_id", "sport", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "control_hz", _positive(self.control_hz, field="control_hz"))
        object.__setattr__(self, "timeout_s", _positive(self.timeout_s, field="timeout_s"))
        if not isinstance(self.frame, TaskFrame):
            raise TypeError("frame must be a TaskFrame")
        if not isinstance(self.table, TableTennisTableSpec):
            raise TypeError("table must be a TableTennisTableSpec")
        if not isinstance(self.paddle_workspace, PaddleWorkspace):
            raise TypeError("paddle_workspace must be a PaddleWorkspace")
        if not isinstance(self.ball_limits, BallObservationLimits):
            raise TypeError("ball_limits must be a BallObservationLimits")
        if not isinstance(self.reward, RewardWeights):
            raise TypeError("reward must be a RewardWeights")

    @property
    def convention(self) -> CoordinateConvention:
        return self.frame.convention

    def decimation(self, physics_dt: float) -> int:
        """Physics steps held per control step, at least one."""
        timestep = _positive(physics_dt, field="physics_dt")
        return max(1, round(1.0 / (self.control_hz * timestep)))

    def control_dt(self, physics_dt: float) -> float:
        return self.decimation(physics_dt) * _positive(physics_dt, field="physics_dt")

    def max_physics_steps(self, physics_dt: float) -> int:
        """Physics steps that cover the timeout, with slack for float time."""
        timestep = _positive(physics_dt, field="physics_dt")
        return int(self.timeout_s / timestep) + 2

    def max_control_steps(self, physics_dt: float) -> int:
        """Control steps that cover the timeout; the Isaac episode length."""
        return max(1, ceil(self.timeout_s / self.control_dt(physics_dt)) + 1)

    def action_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Low/high bounds of ``[x, y, z, qw, qx, qy, qz]``."""
        low = (*self.paddle_workspace.position_low, -1.0, -1.0, -1.0, -1.0)
        high = (*self.paddle_workspace.position_high, 1.0, 1.0, 1.0, 1.0)
        return low, high

    def observation_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Low/high bounds of ball pose/velocity/spin followed by paddle pose."""
        limits = self.ball_limits
        linear = limits.linear_velocity_limit
        angular = limits.angular_velocity_limit
        action_low, action_high = self.action_bounds()
        low = (
            *limits.position_low,
            *(-linear,) * 3,
            *(-angular,) * 3,
            *action_low,
        )
        high = (
            *limits.position_high,
            *(linear,) * 3,
            *(angular,) * 3,
            *action_high,
        )
        return low, high

    def reward_for(self, result: EpisodeResult, *, previously_hit: bool = False) -> float:
        """Score one control step from judge state, identically on any backend."""
        if not isinstance(result, EpisodeResult):
            raise TypeError("result must be an EpisodeResult")
        reward = self.reward.hit * float(result.hit and not previously_hit)
        if result.valid_return:
            reward += self.reward.valid_return
        if result.target_hit:
            reward += self.reward.target_hit
        return float(reward)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "env_id": self.env_id,
            "sport": self.sport,
            "split": self.split,
            "control_hz": self.control_hz,
            "timeout_s": self.timeout_s,
            "action_dim": self.ACTION_DIM,
            "observation_dim": self.OBSERVATION_DIM,
            "frame": self.frame.to_dict(),
            "table": {
                "length_m": self.table.length_m,
                "width_m": self.table.width_m,
                "top_height_m": self.table.top_height_m,
                "net_plane_x_m": self.table.net_plane_x_m,
                "center_y_m": self.table.center_y_m,
            },
            "paddle_workspace": self.paddle_workspace.to_dict(),
            "ball_limits": self.ball_limits.to_dict(),
            "reward": self.reward.to_dict(),
        }


TABLE_TENNIS_RETURN_V0 = TableTennisReturnTaskConfig()
"""Shared configuration of the experimental table-tennis return task."""
