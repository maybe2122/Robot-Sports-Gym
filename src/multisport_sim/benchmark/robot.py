"""Backend-neutral robot contracts shared by every embodied benchmark task.

A benchmark task must be able to score a policy without knowing which robot
executes it, and a robot must be attachable to a task without that task naming
any of its joints.  This module is the boundary that makes both true:

* :class:`RobotAdapter` is the only interface a task uses to reset, observe and
  command a robot.  It exposes an *ordered* joint list, so task code addresses
  joints by index and semantic role, never by name.
* :class:`ControlMode` and the command dataclasses fix what "an action" means.
  An adapter declares the modes it supports and rejects the others explicitly
  rather than silently reinterpreting a command.
* :class:`SafetyLimits` turns joint, velocity, torque, collision and workspace
  envelopes into a list of :class:`SafetyViolation` records, which is what
  drives the benchmark's ``failure_reason="safety"`` termination path.

Nothing here imports a simulator.  MuJoCo and Isaac adapters both implement
these types; the rule engine and the metrics layer only ever see them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, sqrt
from numbers import Real
from typing import Any, Protocol, runtime_checkable

Vec3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


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


def _finite_vector(values: object, *, size: int | None, field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must contain finite numbers")
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{field} must contain finite numbers") from exc
    if size is not None and len(items) != size:
        raise ValueError(f"{field} must contain exactly {size} finite numbers")
    return tuple(
        _finite(item, field=f"{field}[{index}]") for index, item in enumerate(items)
    )


def _unit_quaternion(values: object, *, field: str) -> Quaternion:
    """Normalize a ``wxyz`` quaternion, rejecting a degenerate one."""
    components = _finite_vector(values, size=4, field=field)
    norm = sqrt(sum(value * value for value in components))
    if norm <= 1e-12:
        raise ValueError(f"{field} must have non-zero norm")
    w, x, y, z = (value / norm for value in components)
    return (w, x, y, z)


class ControlMode(str, Enum):
    """How a command reaches the actuators.

    The string values are stable: they appear in reports and in submission
    manifests, so a result can be compared only against results produced in the
    same mode.
    """

    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"
    EFFECTOR_POSE = "effector_pose"

    @property
    def is_joint_space(self) -> bool:
        return self is not ControlMode.EFFECTOR_POSE


class UnsupportedControlMode(ValueError):
    """Raised when an adapter is handed a command it does not implement.

    A distinct type matters: a task that falls back to another mode after
    catching a bare ``ValueError`` would silently change what it measured.
    """

    def __init__(self, robot_id: str, mode: ControlMode, supported: Iterable[ControlMode]) -> None:
        names = ", ".join(sorted(item.value for item in supported)) or "none"
        super().__init__(
            f"robot {robot_id!r} does not support control mode {mode.value!r}; "
            f"supported modes: {names}"
        )
        self.robot_id = robot_id
        self.mode = mode
        self.supported = tuple(supported)


@dataclass(frozen=True)
class JointCommand:
    """One per-joint setpoint vector, interpreted by :attr:`mode`.

    Units follow the mode: radians for position, rad/s for velocity and newton
    metres for torque.  The vector length must equal the adapter's ``dof`` and
    is ordered exactly like :attr:`RobotAdapter.joint_names`.
    """

    targets: tuple[float, ...]
    mode: ControlMode = ControlMode.JOINT_POSITION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "targets", _finite_vector(self.targets, size=None, field="targets")
        )
        if not self.targets:
            raise ValueError("targets must contain at least one value")
        mode = self.mode if isinstance(self.mode, ControlMode) else ControlMode(self.mode)
        if not mode.is_joint_space:
            raise ValueError("JointCommand requires a joint-space control mode")
        object.__setattr__(self, "mode", mode)

    def __len__(self) -> int:
        return len(self.targets)


@dataclass(frozen=True)
class EffectorPoseCommand:
    """A task-frame target pose for the robot's end effector.

    The quaternion is stored normalized in ``wxyz`` order, matching
    :class:`~multisport_sim.benchmark.task_config.CoordinateConvention`.  An
    adapter that accepts this mode owns the inverse kinematics or operational
    space controller that realizes it, and must document that choice: the
    solver becomes part of what a submission measured.
    """

    position: Vec3
    quaternion: Quaternion = (1.0, 0.0, 0.0, 0.0)
    mode: ControlMode = ControlMode.EFFECTOR_POSE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", _finite_vector(self.position, size=3, field="position")
        )
        object.__setattr__(
            self, "quaternion", _unit_quaternion(self.quaternion, field="quaternion")
        )
        mode = self.mode if isinstance(self.mode, ControlMode) else ControlMode(self.mode)
        if mode is not ControlMode.EFFECTOR_POSE:
            raise ValueError("EffectorPoseCommand requires ControlMode.EFFECTOR_POSE")
        object.__setattr__(self, "mode", mode)


RobotCommand = JointCommand | EffectorPoseCommand


@dataclass(frozen=True)
class RobotObservation:
    """Everything a task may read back from a robot in one control step.

    Joint vectors are ordered like :attr:`RobotAdapter.joint_names`.  Effector
    quantities are in the task frame, so a task never has to know where the
    backend placed the robot in its own world.  ``applied_torque`` is the torque
    the actuators actually delivered, which is what the energy and safety
    metrics integrate -- not the commanded value.
    """

    time_s: float
    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    applied_torque: tuple[float, ...]
    effector_position: Vec3
    effector_quaternion: Quaternion
    effector_linear_velocity: Vec3 = (0.0, 0.0, 0.0)
    contacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", _finite(self.time_s, field="time_s"))
        positions = _finite_vector(self.joint_positions, size=None, field="joint_positions")
        if not positions:
            raise ValueError("joint_positions must contain at least one value")
        object.__setattr__(self, "joint_positions", positions)
        for name in ("joint_velocities", "applied_torque"):
            values = _finite_vector(getattr(self, name), size=len(positions), field=name)
            object.__setattr__(self, name, values)
        object.__setattr__(
            self,
            "effector_position",
            _finite_vector(self.effector_position, size=3, field="effector_position"),
        )
        object.__setattr__(
            self,
            "effector_quaternion",
            _unit_quaternion(self.effector_quaternion, field="effector_quaternion"),
        )
        object.__setattr__(
            self,
            "effector_linear_velocity",
            _finite_vector(
                self.effector_linear_velocity, size=3, field="effector_linear_velocity"
            ),
        )
        contacts = tuple(self.contacts)
        if any(not isinstance(name, str) or not name.strip() for name in contacts):
            raise ValueError("contacts must be non-empty strings")
        object.__setattr__(self, "contacts", contacts)

    @property
    def dof(self) -> int:
        return len(self.joint_positions)

    def mechanical_power_w(self) -> float:
        """Instantaneous absolute joint power; the integrand of ``energy_joule``.

        Absolute value is deliberate: a policy that dumps energy by braking
        against its own motion is spending it, not recovering it, so the
        benchmark charges for both directions.
        """
        return sum(
            abs(torque * velocity)
            for torque, velocity in zip(self.applied_torque, self.joint_velocities)
        )


@dataclass(frozen=True)
class JointLimits:
    """Per-joint position, velocity and torque envelope, in SI units.

    These are the *robot's* limits as published by its manufacturer or asset,
    not the task's preferences.  ``position_low``/``position_high`` come from
    the asset; velocity and torque limits usually do not appear in an MJCF and
    must be supplied from the datasheet, which is why they are required here.
    """

    joint_names: tuple[str, ...]
    position_low: tuple[float, ...]
    position_high: tuple[float, ...]
    velocity_limit: tuple[float, ...]
    torque_limit: tuple[float, ...]

    def __post_init__(self) -> None:
        names = tuple(self.joint_names)
        if not names or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("joint_names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique")
        object.__setattr__(self, "joint_names", names)
        for name in ("position_low", "position_high", "velocity_limit", "torque_limit"):
            values = _finite_vector(getattr(self, name), size=len(names), field=name)
            object.__setattr__(self, name, values)
        if any(low >= high for low, high in zip(self.position_low, self.position_high)):
            raise ValueError("position_low must be strictly below position_high")
        for name in ("velocity_limit", "torque_limit"):
            if any(value <= 0.0 for value in getattr(self, name)):
                raise ValueError(f"{name} entries must be greater than zero")

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def clamp_positions(self, targets: Sequence[float]) -> tuple[float, ...]:
        """Clip a position command into the joint range, rejecting a wrong size."""
        values = _finite_vector(targets, size=self.dof, field="targets")
        return tuple(
            min(max(value, low), high)
            for value, low, high in zip(values, self.position_low, self.position_high)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_names": list(self.joint_names),
            "position_low": list(self.position_low),
            "position_high": list(self.position_high),
            "velocity_limit": list(self.velocity_limit),
            "torque_limit": list(self.torque_limit),
        }


@dataclass(frozen=True)
class WorkspaceBox:
    """An axis-aligned task-frame box the effector must stay inside."""

    low: Vec3
    high: Vec3

    def __post_init__(self) -> None:
        object.__setattr__(self, "low", _finite_vector(self.low, size=3, field="low"))
        object.__setattr__(self, "high", _finite_vector(self.high, size=3, field="high"))
        if any(low >= high for low, high in zip(self.low, self.high)):
            raise ValueError("low must be strictly below high")

    def contains(self, position: Sequence[float]) -> bool:
        values = _finite_vector(position, size=3, field="position")
        return all(
            low <= value <= high
            for value, low, high in zip(values, self.low, self.high)
        )

    def excess(self, position: Sequence[float]) -> float:
        """Largest single-axis distance outside the box; zero when inside."""
        values = _finite_vector(position, size=3, field="position")
        return max(
            max(low - value, value - high, 0.0)
            for value, low, high in zip(values, self.low, self.high)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"low": list(self.low), "high": list(self.high)}


VIOLATION_KINDS = (
    "joint_position",
    "joint_velocity",
    "joint_torque",
    "collision",
    "workspace",
)


@dataclass(frozen=True)
class SafetyViolation:
    """One breach of the declared envelope, with the numbers that caused it.

    Reports carry these verbatim.  A submission that violates safety is not
    quietly rescored: the episode terminates with ``failure_reason="safety"``
    and counts in the denominator like any other failure.
    """

    kind: str
    detail: str
    value: float
    limit: float
    joint_index: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in VIOLATION_KINDS:
            raise ValueError(
                f"unknown violation kind {self.kind!r}; expected one of "
                f"{', '.join(VIOLATION_KINDS)}"
            )
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError("detail must be a non-empty string")
        object.__setattr__(self, "value", _finite(self.value, field="value"))
        object.__setattr__(self, "limit", _finite(self.limit, field="limit"))
        if self.joint_index is not None:
            if isinstance(self.joint_index, bool) or not isinstance(self.joint_index, int):
                raise ValueError("joint_index must be an integer or None")
            if self.joint_index < 0:
                raise ValueError("joint_index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "value": self.value,
            "limit": self.limit,
            "joint_index": self.joint_index,
        }


@dataclass(frozen=True)
class SafetyLimits:
    """The four safety classes the benchmark specification requires.

    Joint-position, velocity and torque envelopes come from :attr:`joints`;
    reachable space comes from :attr:`workspace`; anything the robot must not
    touch is named in :attr:`forbidden_contacts`, matched against the semantic
    contact labels an adapter reports.

    The tolerances exist because a simulator resolves a limit over a few steps:
    a joint sitting exactly on its stop, or a torque overshooting by a fraction
    of a percent during contact, is normal actuator behaviour and must not be
    scored as a safety failure.  They are declared, versioned numbers, not
    private fudge factors.
    """

    joints: JointLimits
    workspace: WorkspaceBox | None = None
    forbidden_contacts: frozenset[str] = frozenset({"table", "net", "floor"})
    position_tolerance_rad: float = 0.02
    velocity_tolerance_radps: float = 0.05
    torque_tolerance_nm: float = 1.0
    workspace_tolerance_m: float = 0.02

    def __post_init__(self) -> None:
        if not isinstance(self.joints, JointLimits):
            raise TypeError("joints must be a JointLimits")
        if self.workspace is not None and not isinstance(self.workspace, WorkspaceBox):
            raise TypeError("workspace must be a WorkspaceBox or None")
        contacts = frozenset(self.forbidden_contacts)
        if any(not isinstance(name, str) or not name.strip() for name in contacts):
            raise ValueError("forbidden_contacts must be non-empty strings")
        object.__setattr__(self, "forbidden_contacts", contacts)
        for name in (
            "position_tolerance_rad",
            "velocity_tolerance_radps",
            "torque_tolerance_nm",
            "workspace_tolerance_m",
        ):
            value = _finite(getattr(self, name), field=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)

    def violations(self, observation: RobotObservation) -> tuple[SafetyViolation, ...]:
        """Return every envelope breach in one observation, worst class first.

        All four classes are always evaluated; the caller decides whether one
        violation ends the episode.  Returning the full list keeps the reported
        ``safety_violations`` count honest when several limits break at once.
        """
        if not isinstance(observation, RobotObservation):
            raise TypeError("observation must be a RobotObservation")
        if observation.dof != self.joints.dof:
            raise ValueError(
                f"observation has {observation.dof} joints but the limits "
                f"declare {self.joints.dof}"
            )

        found: list[SafetyViolation] = []
        names = self.joints.joint_names

        for index, position in enumerate(observation.joint_positions):
            low = self.joints.position_low[index] - self.position_tolerance_rad
            high = self.joints.position_high[index] + self.position_tolerance_rad
            if position < low or position > high:
                limit = self.joints.position_low[index] if position < low else self.joints.position_high[index]
                found.append(
                    SafetyViolation(
                        kind="joint_position",
                        detail=f"joint {names[index]} left its range",
                        value=position,
                        limit=limit,
                        joint_index=index,
                    )
                )

        for index, velocity in enumerate(observation.joint_velocities):
            limit = self.joints.velocity_limit[index]
            if abs(velocity) > limit + self.velocity_tolerance_radps:
                found.append(
                    SafetyViolation(
                        kind="joint_velocity",
                        detail=f"joint {names[index]} exceeded its speed limit",
                        value=abs(velocity),
                        limit=limit,
                        joint_index=index,
                    )
                )

        for index, torque in enumerate(observation.applied_torque):
            limit = self.joints.torque_limit[index]
            if abs(torque) > limit + self.torque_tolerance_nm:
                found.append(
                    SafetyViolation(
                        kind="joint_torque",
                        detail=f"joint {names[index]} exceeded its torque limit",
                        value=abs(torque),
                        limit=limit,
                        joint_index=index,
                    )
                )

        for label in sorted(self.forbidden_contacts & set(observation.contacts)):
            found.append(
                SafetyViolation(
                    kind="collision",
                    detail=f"robot touched {label}",
                    value=1.0,
                    limit=0.0,
                )
            )

        if self.workspace is not None:
            excess = self.workspace.excess(observation.effector_position)
            if excess > self.workspace_tolerance_m:
                found.append(
                    SafetyViolation(
                        kind="workspace",
                        detail="effector left the declared workspace",
                        value=excess,
                        limit=self.workspace_tolerance_m,
                    )
                )

        order = {kind: index for index, kind in enumerate(VIOLATION_KINDS)}
        return tuple(sorted(found, key=lambda item: order[item.kind]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "joints": self.joints.to_dict(),
            "workspace": None if self.workspace is None else self.workspace.to_dict(),
            "forbidden_contacts": sorted(self.forbidden_contacts),
            "tolerances": {
                "position_rad": self.position_tolerance_rad,
                "velocity_radps": self.velocity_tolerance_radps,
                "torque_nm": self.torque_tolerance_nm,
                "workspace_m": self.workspace_tolerance_m,
            },
        }


@runtime_checkable
class RobotAdapter(Protocol):
    """The whole interface a benchmark task has to a robot.

    Task code addresses joints by index into :attr:`joint_names` and reads
    semantic roles off this protocol.  It must never look a joint up by name,
    because the same task has to run on a different arm without edits.
    """

    robot_id: str
    """Stable identifier written into every report, e.g. ``"franka-panda-v1"``."""

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Actuated joints in command order; the ordering is part of the contract."""
        ...

    @property
    def dof(self) -> int:
        ...

    @property
    def effector_name(self) -> str:
        """Semantic role of the controlled frame, e.g. ``"racket"`` or ``"foot"``."""
        ...

    @property
    def control_modes(self) -> frozenset[ControlMode]:
        """Modes this adapter implements; anything else must raise."""
        ...

    @property
    def safety_limits(self) -> SafetyLimits:
        ...

    def reset(self) -> None:
        """Restore the documented default pose deterministically."""
        ...

    def observe(self) -> RobotObservation:
        ...

    def apply(self, command: RobotCommand) -> None:
        """Send one command, raising :class:`UnsupportedControlMode` if needed."""
        ...

    def describe(self) -> dict[str, Any]:
        """Report-ready identity, asset provenance and declared limits."""
        ...


@dataclass(frozen=True)
class SafetyMonitor:
    """Accumulates safety state across one episode.

    An episode ends on the first violation, but the count of *all* violations
    seen in that step is what the report carries, so a policy that breaks three
    limits at once is not recorded as breaking one.
    """

    limits: SafetyLimits
    _seen: list[SafetyViolation] = field(default_factory=list)

    def reset(self) -> None:
        self._seen.clear()

    def check(self, observation: RobotObservation) -> tuple[SafetyViolation, ...]:
        violations = self.limits.violations(observation)
        self._seen.extend(violations)
        return violations

    @property
    def violations(self) -> tuple[SafetyViolation, ...]:
        return tuple(self._seen)

    @property
    def count(self) -> int:
        return len(self._seen)

    @property
    def triggered(self) -> bool:
        return bool(self._seen)


__all__ = [
    "VIOLATION_KINDS",
    "ControlMode",
    "EffectorPoseCommand",
    "JointCommand",
    "JointLimits",
    "RobotAdapter",
    "RobotCommand",
    "RobotObservation",
    "SafetyLimits",
    "SafetyMonitor",
    "SafetyViolation",
    "UnsupportedControlMode",
    "WorkspaceBox",
]
