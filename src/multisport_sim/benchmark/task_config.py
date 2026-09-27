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
from .observation import ObservationLayout, joint_space_layout
from .rules.badminton import BADMINTON as BADMINTON_RULES
from .rules.badminton import BadmintonCourtSpec
from .rules.basketball import BASKET as BASKET_RULES
from .rules.basketball import BasketSpec
from .rules.football import FOOTBALL_GOAL as FOOTBALL_GOAL_RULES
from .rules.football import FootballGoalSpec
from .rules.table_tennis import TABLE_TENNIS as TABLE_TENNIS_RULES
from .rules.table_tennis import TableTennisTableSpec
from .rules.tennis import TENNIS as TENNIS_RULES
from .rules.tennis import TennisCourtSpec
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
    world -- the Isaac campus lays six sports out side by side -- declares that
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
class EffectorWorkspace:
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


# The first task shipped a paddle-specific name; both spell the same envelope.
PaddleWorkspace = EffectorWorkspace


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
class ShotTaskConfig:
    """Everything any single-shot sports task must agree on across backends.

    ``control_hz`` and ``timeout_s`` are task semantics, so a backend derives
    its decimation and step budget from them via :meth:`decimation` and
    :meth:`max_control_steps` instead of hard-coding a physics timestep.
    Sport-specific rule geometry lives in a subclass, which also supplies the
    defaults for the identity fields below.
    """

    task_id: str
    env_id: str
    sport: str
    bank_resource: str
    split: str = "dev"
    control_hz: float = 200.0
    timeout_s: float = 2.0
    frame: TaskFrame = field(default_factory=TaskFrame)
    workspace: EffectorWorkspace = field(default_factory=EffectorWorkspace)
    ball_limits: BallObservationLimits = field(default_factory=BallObservationLimits)
    reward: RewardWeights = field(default_factory=RewardWeights)

    ACTION_DIM = 7
    OBSERVATION_DIM = 16

    def __post_init__(self) -> None:
        for name in ("task_id", "env_id", "sport", "bank_resource", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "control_hz", _positive(self.control_hz, field="control_hz"))
        object.__setattr__(self, "timeout_s", _positive(self.timeout_s, field="timeout_s"))
        if not isinstance(self.frame, TaskFrame):
            raise TypeError("frame must be a TaskFrame")
        if not isinstance(self.workspace, EffectorWorkspace):
            raise TypeError("workspace must be an EffectorWorkspace")
        if not isinstance(self.ball_limits, BallObservationLimits):
            raise TypeError("ball_limits must be a BallObservationLimits")
        if not isinstance(self.reward, RewardWeights):
            raise TypeError("reward must be a RewardWeights")

    @property
    def convention(self) -> CoordinateConvention:
        return self.frame.convention

    def rule_geometry(self) -> dict[str, Any]:
        """Serialize the sport's rule geometry; subclasses fill this in."""
        return {}

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
        low = (*self.workspace.position_low, -1.0, -1.0, -1.0, -1.0)
        high = (*self.workspace.position_high, 1.0, 1.0, 1.0, 1.0)
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
        payload = {
            "task_id": self.task_id,
            "env_id": self.env_id,
            "sport": self.sport,
            "bank_resource": self.bank_resource,
            "split": self.split,
            "control_hz": self.control_hz,
            "timeout_s": self.timeout_s,
            "action_dim": self.ACTION_DIM,
            "observation_dim": self.OBSERVATION_DIM,
            "frame": self.frame.to_dict(),
            "workspace": self.workspace.to_dict(),
            "ball_limits": self.ball_limits.to_dict(),
            "reward": self.reward.to_dict(),
        }
        payload.update(self.rule_geometry())
        return payload


@dataclass(frozen=True)
class TableTennisReturnTaskConfig(ShotTaskConfig):
    """Table-tennis return task: the shared core plus regulation table geometry."""

    task_id: str = "table-tennis-return-v0"
    env_id: str = "MultiSportRobot/TableTennisReturn-v0"
    sport: str = Sport.TABLE_TENNIS.value
    bank_resource: str = "table_tennis/return-v0"
    table: TableTennisTableSpec = TABLE_TENNIS_RULES
    workspace: EffectorWorkspace = field(
        default_factory=lambda: EffectorWorkspace(
            position_low=(-2.0, -1.5, 0.5), position_high=(0.0, 1.5, 1.8)
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.table, TableTennisTableSpec):
            raise TypeError("table must be a TableTennisTableSpec")

    @property
    def paddle_workspace(self) -> EffectorWorkspace:
        """Name kept from the first released task; the envelope is shared."""
        return self.workspace

    def rule_geometry(self) -> dict[str, Any]:
        return {
            "table": {
                "length_m": self.table.length_m,
                "width_m": self.table.width_m,
                "top_height_m": self.table.top_height_m,
                "net_plane_x_m": self.table.net_plane_x_m,
                "center_y_m": self.table.center_y_m,
            }
        }


TABLE_TENNIS_RETURN_V0 = TableTennisReturnTaskConfig()
"""Shared configuration of the experimental table-tennis return task."""


@dataclass(frozen=True)
class TennisReturnTaskConfig(ShotTaskConfig):
    """Tennis return task: the shared core plus regulation singles-court geometry.

    Only the numbers differ from table tennis, and every one of them differs for
    the same reason: the court is roughly nine times longer, the ball arrives
    three to five times faster, and the strike happens near the baseline instead
    of a arm's length from the net.  The rules, the judge, the report schema and
    the level thresholds are the ones table tennis already uses.
    """

    task_id: str = "tennis-return-v0"
    env_id: str = "MultiSportRobot/TennisReturn-v0"
    sport: str = Sport.TENNIS.value
    bank_resource: str = "tennis/return-v0"
    # A rally ball crosses 20 m of court; two seconds is not enough time for it
    # to arrive, be struck and land again.
    timeout_s: float = 3.0
    court: TennisCourtSpec = TENNIS_RULES
    workspace: EffectorWorkspace = field(
        default_factory=lambda: EffectorWorkspace(
            # Behind the robot's baseline, across the singles width, from just
            # above the court to a high overhead.
            position_low=(-13.0, -5.0, 0.1),
            position_high=(-6.0, 5.0, 3.2),
        )
    )
    ball_limits: BallObservationLimits = field(
        default_factory=lambda: BallObservationLimits(
            position_low=(-20.0, -10.0, -1.0),
            position_high=(20.0, 10.0, 15.0),
            linear_velocity_limit=80.0,
            angular_velocity_limit=800.0,
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.court, TennisCourtSpec):
            raise TypeError("court must be a TennisCourtSpec")

    @property
    def table(self) -> TennisCourtSpec:
        """The playing surface, under the name the shared engine reads."""
        return self.court

    def rule_geometry(self) -> dict[str, Any]:
        return {
            "court": {
                "length_m": self.court.length_m,
                "width_m": self.court.width_m,
                "top_height_m": self.court.top_height_m,
                "net_plane_x_m": self.court.net_plane_x_m,
                "net_height_m": self.court.net_height_m,
                "center_y_m": self.court.center_y_m,
            }
        }


TENNIS_RETURN_V0 = TennisReturnTaskConfig()
"""Shared configuration of the experimental tennis return task."""


@dataclass(frozen=True)
class BadmintonServeTaskConfig(ShotTaskConfig):
    """Badminton serve task: the shared core plus BWF singles service geometry.

    The first launch task.  The action and observation are the return tasks':
    a 7-value effector pose and the 16-value ball-plus-effector state, so a
    policy interface written for table tennis runs here unchanged.  What
    differs is who moves first -- the shuttle is released on the robot's side
    and waits to be struck -- and what counts as success.
    """

    task_id: str = "badminton-serve-v0"
    env_id: str = "MultiSportRobot/BadmintonServe-v0"
    sport: str = Sport.BADMINTON.value
    bank_resource: str = "badminton/serve-v0"
    # A high serve is in the air for about two seconds; four leaves room for a
    # slow release, the swing and the flight.
    timeout_s: float = 4.0
    court: BadmintonCourtSpec = BADMINTON_RULES
    workspace: EffectorWorkspace = field(
        default_factory=lambda: EffectorWorkspace(
            # The server's half, from the back boundary line to just short of
            # the net, across the doubles width, from the floor to overhead.
            position_low=(-6.8, -3.1, 0.05),
            position_high=(-0.3, 3.1, 2.6),
        )
    )
    ball_limits: BallObservationLimits = field(
        default_factory=lambda: BallObservationLimits(
            position_low=(-10.0, -6.0, -1.0),
            position_high=(10.0, 6.0, 12.0),
            linear_velocity_limit=80.0,
            angular_velocity_limit=400.0,
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.court, BadmintonCourtSpec):
            raise TypeError("court must be a BadmintonCourtSpec")

    @property
    def table(self) -> BadmintonCourtSpec:
        """The playing surface, under the name the shared engine reads."""
        return self.court

    def rule_geometry(self) -> dict[str, Any]:
        return {
            "court": {
                "length_m": self.court.length_m,
                "width_m": self.court.width_m,
                "top_height_m": self.court.top_height_m,
                "net_plane_x_m": self.court.net_plane_x_m,
                "net_height_m": self.court.net_height_m,
                "center_y_m": self.court.center_y_m,
                "short_service_line_m": self.court.short_service_line_m,
                "service_height_limit_m": self.court.service_height_limit_m,
            }
        }


BADMINTON_SERVE_V0 = BadmintonServeTaskConfig()
"""Shared configuration of the experimental badminton serve task."""


@dataclass(frozen=True)
class FootballKickTaskConfig(ShotTaskConfig):
    """Football kick-to-target: IFAB goal geometry, task frame at the goal line.

    The frame's origin is the centre of the east goal line on the ground; the
    kicker is at x < 0, so every shared convention -- robot side negative,
    launch toward +x -- holds without a special case.
    """

    task_id: str = "football-kick-v0"
    env_id: str = "MultiSportRobot/FootballKick-v0"
    sport: str = Sport.FOOTBALL.value
    bank_resource: str = "football/kick-v0"
    timeout_s: float = 3.0
    goal: FootballGoalSpec = FOOTBALL_GOAL_RULES
    frame: TaskFrame = field(default_factory=lambda: TaskFrame(origin_xyz=(52.5, 0.0, 0.0)))
    workspace: EffectorWorkspace = field(
        default_factory=lambda: EffectorWorkspace(
            position_low=(-30.0, -34.0, 0.02),
            position_high=(-0.5, 34.0, 1.2),
        )
    )
    ball_limits: BallObservationLimits = field(
        default_factory=lambda: BallObservationLimits(
            position_low=(-110.0, -40.0, -1.0),
            position_high=(10.0, 40.0, 20.0),
            linear_velocity_limit=60.0,
            angular_velocity_limit=200.0,
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.goal, FootballGoalSpec):
            raise TypeError("goal must be a FootballGoalSpec")

    def rule_geometry(self) -> dict[str, Any]:
        return {
            "goal": {
                "goal_half_width_m": self.goal.goal_half_width_m,
                "crossbar_height_m": self.goal.crossbar_height_m,
                "pitch_length_m": self.goal.pitch_length_m,
                "pitch_width_m": self.goal.pitch_width_m,
                "ball_radius_m": self.goal.ball_radius_m,
            }
        }


FOOTBALL_KICK_V0 = FootballKickTaskConfig()
"""Shared configuration of the experimental football kick task."""


@dataclass(frozen=True)
class BasketballShootTaskConfig(ShotTaskConfig):
    """Basketball shooting: FIBA basket geometry, task frame below the rim."""

    task_id: str = "basketball-shoot-v0"
    env_id: str = "MultiSportRobot/BasketballShoot-v0"
    sport: str = Sport.BASKETBALL.value
    bank_resource: str = "basketball/shoot-v0"
    timeout_s: float = 4.0
    basket: BasketSpec = BASKET_RULES
    frame: TaskFrame = field(default_factory=lambda: TaskFrame(origin_xyz=(12.425, 0.0, 0.0)))
    workspace: EffectorWorkspace = field(
        default_factory=lambda: EffectorWorkspace(
            position_low=(-9.0, -7.5, 0.3),
            position_high=(-0.3, 7.5, 3.2),
        )
    )
    ball_limits: BallObservationLimits = field(
        default_factory=lambda: BallObservationLimits(
            position_low=(-16.0, -9.0, -1.0),
            position_high=(3.0, 9.0, 10.0),
            linear_velocity_limit=30.0,
            angular_velocity_limit=100.0,
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.basket, BasketSpec):
            raise TypeError("basket must be a BasketSpec")

    def rule_geometry(self) -> dict[str, Any]:
        return {
            "basket": {
                "rim_height_m": self.basket.rim_height_m,
                "rim_inner_radius_m": self.basket.rim_inner_radius_m,
                "ball_radius_m": self.basket.ball_radius_m,
            }
        }


BASKETBALL_SHOOT_V0 = BasketballShootTaskConfig()
"""Shared configuration of the experimental basketball shooting task."""


@dataclass(frozen=True)
class StrikeZone:
    """The region of the task frame the frozen shot bank actually crosses.

    This is a *measured* property of the shot bank, not a design choice: it is
    produced by ``scripts/calibrate_reachability.py``, which launches every shot
    and records where the ball passes the strike plane.  A robot mount is judged
    against it, and a task that no embodiment can cover is a task that needs a
    new bank -- not a robot that needs excusing.

    The defaults are the ``table_tennis/return-v1`` train-split crossings
    (``reports/table-tennis-*-reachability.json``), rounded outward to 1 cm.
    The rectangle bounds the crossings; its corners are not visited by any shot.
    """

    plane_x_m: float = -1.55
    y_low: float = -0.72
    y_high: float = 0.71
    z_low: float = 0.70
    z_high: float = 1.42

    def __post_init__(self) -> None:
        for name in ("plane_x_m", "y_low", "y_high", "z_low", "z_high"):
            object.__setattr__(self, name, _finite(getattr(self, name), field=name))
        if self.y_low >= self.y_high or self.z_low >= self.z_high:
            raise ValueError("strike zone bounds must be strictly increasing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plane_x_m": self.plane_x_m,
            "y_m": [self.y_low, self.y_high],
            "z_m": [self.z_low, self.z_high],
        }


@dataclass(frozen=True)
class JointActionLimits:
    """Declared per-joint action range of an embodied task, in SI units.

    A joint-space task publishes the robot's own limits as its action space, so
    a policy cannot request motion the hardware could not perform and then have
    the adapter quietly clip it into something else.
    """

    joint_names: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    control_mode: str = "joint_position"

    def __post_init__(self) -> None:
        names = tuple(self.joint_names)
        if not names or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("joint_names must be non-empty strings")
        object.__setattr__(self, "joint_names", names)
        for field_name in ("low", "high"):
            values = tuple(
                _finite(value, field=f"{field_name}[{index}]")
                for index, value in enumerate(getattr(self, field_name))
            )
            if len(values) != len(names):
                raise ValueError(f"{field_name} must hold one value per joint")
            object.__setattr__(self, field_name, values)
        if any(low >= high for low, high in zip(self.low, self.high)):
            raise ValueError("low must be strictly below high")
        if not isinstance(self.control_mode, str) or not self.control_mode.strip():
            raise ValueError("control_mode must be a non-empty string")

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_names": list(self.joint_names),
            "low": list(self.low),
            "high": list(self.high),
            "control_mode": self.control_mode,
        }


PANDA_JOINT_ACTION = JointActionLimits(
    joint_names=tuple(f"rb_joint{index}" for index in range(1, 8)),
    low=(-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973),
    high=(2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973),
)
"""Franka Panda position limits, mirrored from the Menagerie asset.

They are restated here so that listing the task's action space never needs the
asset on disk; the adapter asserts that the compiled model agrees.
"""


G1_JOINT_ACTION = JointActionLimits(
    joint_names=(
        "g1_waist_yaw_joint",
        "g1_waist_roll_joint",
        "g1_waist_pitch_joint",
        "g1_right_shoulder_pitch_joint",
        "g1_right_shoulder_roll_joint",
        "g1_right_shoulder_yaw_joint",
        "g1_right_elbow_joint",
        "g1_right_wrist_roll_joint",
        "g1_right_wrist_pitch_joint",
        "g1_right_wrist_yaw_joint",
    ),
    low=(-2.6180, -0.5200, -0.5200, -3.0892, -2.2515, -2.6180, -1.0472, -1.9722, -1.6144, -1.6144),
    high=(2.6180, 0.5200, 0.5200, 2.6704, 1.5882, 2.6180, 2.0944, 1.9722, 1.6144, 1.6144),
)
"""Unitree G1 position limits for the ten joints this task commands.

The waist and the right arm.  The other nineteen joints are present in the
model and held at the asset's stand pose; publishing them as actions would
invite a policy to learn a nineteen-dimensional no-op.

These are the robot's *own* ranges, deliberately including the shoulder
adduction that folds the arm into the torso.  Narrowing the published action
space to keep a policy out of that region would be the benchmark hiding a
failure mode it is supposed to score: self-collision is a safety violation and
is reported as one.  The reference solver keeps itself out of that region --
see ``robots/g1.SHOULDER_ROLL_SOLVER_MAX_RAD`` -- because a baseline may be
smart, but the contract may not be narrow.
"""


@dataclass(frozen=True)
class EmbodiedTableTennisReturnTaskConfig(TableTennisReturnTaskConfig):
    """A table-tennis return task performed by an actuated robot.

    This is a separate versioned task from the mocap fixture, not a new revision
    of it.  The two measure different things -- one asks whether the rule engine
    and report chain are correct, the other whether a robot can play -- and
    mixing their scores would be meaningless.  They deliberately share the same
    judge, the same frozen shot bank and the same reward terms, so the only
    difference between a v0 and an embodied result is the embodiment.

    The action is a joint-position setpoint vector held for one control step.
    The observation is the ball state followed by the robot's joint state and
    its blade pose and velocity in the task frame -- no geometry names, no
    simulator handles, nothing a different robot could not supply.

    Nothing in this class names a robot.  Its two concrete subclasses differ
    only in identifiers and in the joint set they publish, which is the whole
    claim the second embodiment exists to test.
    """

    robot_id: str = "unspecified-robot"
    joint_action: JointActionLimits = PANDA_JOINT_ACTION
    strike_zone: StrikeZone = field(default_factory=StrikeZone)
    workspace: EffectorWorkspace = field(
        default_factory=lambda: EffectorWorkspace(
            # A safety envelope, not a reachability estimate.  It describes the
            # robot's operating cell: the blade may go anywhere the arm can
            # physically reach behind the table, but not over the playing
            # surface and not down among the legs.  Making it any tighter would
            # score ordinary arm motion as a safety failure, which would tell a
            # submission nothing about its own behaviour.  Floor and table
            # strikes are caught separately, as collisions.
            position_low=(-2.95, -1.00, 0.30),
            position_high=(-1.05, 1.00, 2.00),
        )
    )

    # The action width is the robot's joint count, and the observation width
    # follows from the published layout rather than being asserted alongside it:
    # a six-axis arm on this same task has a different number and nothing here
    # needs editing for that to be true.  See ``observation_layout``.
    # Joint speed is bounded by the robot's datasheet; the envelope below is
    # deliberately loose so that a safety violation is reported as such rather
    # than silently clipped out of the observation.
    JOINT_VELOCITY_OBSERVATION_LIMIT = 10.0

    @property
    def ACTION_DIM(self) -> int:  # published name, kept stable
        return self.joint_action.dof

    @property
    def OBSERVATION_DIM(self) -> int:  # published name, kept stable
        return self.observation_layout().size

    def observation_layout(self) -> ObservationLayout:
        """The named, per-robot observation contract this task publishes.

        Everything that reads an observation -- the Gymnasium environment, the
        offline scoring path, a submission's own policy -- goes through this,
        so "the observation" is one object with one order, and a second robot
        changes its field *sizes* without changing what any field means.
        """
        limits = self.ball_limits
        joint_low, joint_high = self.action_bounds()
        return joint_space_layout(
            ball_position_low=limits.position_low,
            ball_position_high=limits.position_high,
            linear_velocity_limit=limits.linear_velocity_limit,
            angular_velocity_limit=limits.angular_velocity_limit,
            joint_names=self.joint_action.joint_names,
            joint_position_low=joint_low,
            joint_position_high=joint_high,
            joint_velocity_limit=self.JOINT_VELOCITY_OBSERVATION_LIMIT,
            effector_position_low=self.workspace.position_low,
            effector_position_high=self.workspace.position_high,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.joint_action, JointActionLimits):
            raise TypeError("joint_action must be a JointActionLimits")
        if not isinstance(self.strike_zone, StrikeZone):
            raise TypeError("strike_zone must be a StrikeZone")
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot_id must be a non-empty string")

    def action_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Low/high joint-position setpoints, ordered like ``joint_action``."""
        return self.joint_action.low, self.joint_action.high

    def observation_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Flat bounds, in the order the published layout declares."""
        return self.observation_layout().bounds()

    def rule_geometry(self) -> dict[str, Any]:
        payload = super().rule_geometry()
        payload.update(
            {
                "robot_id": self.robot_id,
                "joint_action": self.joint_action.to_dict(),
                "strike_zone": self.strike_zone.to_dict(),
                # A score is only auditable if the reader can tell what the
                # policy was looking at, which is a per-robot fact.
                "observation_layout": self.observation_layout().to_dict(),
            }
        )
        return payload


@dataclass(frozen=True)
class TableTennisReturnPandaTaskConfig(EmbodiedTableTennisReturnTaskConfig):
    """The table-tennis return task performed by an actuated Franka Panda."""

    task_id: str = "table-tennis-return-panda-v1"
    env_id: str = "MultiSportRobot/TableTennisReturn-Panda-v1"
    # The embodied task is the publishable-scale benchmark, not the tiny v0
    # rule-engine fixture.  Keeping this explicit also prevents a future
    # TableTennisReturnTaskConfig default change from silently moving scores.
    bank_resource: str = "table_tennis/return-v1"
    # The same task on the vision track.  One task id, two environments: a
    # result names the track it ran on, not a different benchmark.
    vision_env_id: str = "MultiSportRobot/TableTennisReturn-Panda-Vision-v1"
    robot_id: str = "franka-panda-tabletennis-v1"
    joint_action: JointActionLimits = PANDA_JOINT_ACTION


@dataclass(frozen=True)
class TableTennisReturnG1TaskConfig(EmbodiedTableTennisReturnTaskConfig):
    """The same task, played by a fixed-base Unitree G1.

    Same judge, same frozen shot bank, same L0-L5 thresholds, same report
    schema, same operating envelope.  What differs is the robot: ten commanded
    joints instead of seven, so a 39-number observation instead of 33 -- and
    neither number is written anywhere, because both are the width the published
    observation layout comes out to.

    The vision environment supports stereo RGB and aligned RGB-D input.
    """

    task_id: str = "table-tennis-return-g1-v1"
    env_id: str = "MultiSportRobot/TableTennisReturn-G1-v1"
    bank_resource: str = "table_tennis/return-v1"
    vision_env_id: str = "MultiSportRobot/TableTennisReturn-G1-Vision-v1"
    robot_id: str = "unitree-g1-tabletennis-v1"
    joint_action: JointActionLimits = G1_JOINT_ACTION


TABLE_TENNIS_RETURN_PANDA_V1 = TableTennisReturnPandaTaskConfig()
"""Shared configuration of the embodied table-tennis return task."""

TABLE_TENNIS_RETURN_G1_V1 = TableTennisReturnG1TaskConfig()
"""The same task on a fixed-base Unitree G1; the benchmark's second embodiment."""


@dataclass(frozen=True)
class TableTennisReturnStandingG1TaskConfig(TableTennisReturnG1TaskConfig):
    """Free-pelvis G1 standing with four locally controlled ankle joints."""

    task_id: str = "table-tennis-return-g1-standing-v2"
    env_id: str = "MultiSportRobot/TableTennisReturn-G1-Standing-v2"
    vision_env_id: str = "MultiSportRobot/TableTennisReturn-G1-Standing-Vision-v2"
    robot_id: str = "unitree-g1-standing-tabletennis-v2"


TABLE_TENNIS_RETURN_STANDING_G1_V2 = TableTennisReturnStandingG1TaskConfig()
