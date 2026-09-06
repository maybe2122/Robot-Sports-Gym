"""What a policy is handed, described rather than assumed.

Every embodied task on this benchmark faces the same problem: the observation a
policy reads is *partly* a property of the task and *partly* a property of the
robot.  The ball state is the task's -- nine numbers, identical for a 7-DoF arm,
a 6-DoF arm and a biped.  The joint block is the robot's, and it is 14 numbers
on a Panda, 12 on a UR5 and something else again on a biped.  Publishing one
flat width -- "the observation is 33 numbers" -- states the second half as if it
were the first, and the moment a second robot appears that number is a lie that
a policy has already been trained against.

So the contract published here is a *layout*: an ordered list of named fields,
each with its own size and bounds.

* The **task** fixes which fields exist and what each one means.  Every robot
  playing table-tennis return exposes ``ball.position`` and it always means the
  same thing in the same frame.
* The **robot** fixes the sizes.  ``robot.joint_position`` has one entry per
  joint, named, in the adapter's own order.
* The **flat vector** is derived, never authored.  It is what a Box-space
  learner consumes, and it exists so that training and offline scoring cannot
  drift apart -- but the layout is what a submission is scored against, and it
  is what goes into the report.

The payoff is :meth:`ObservationLayout.view`.  A policy declares the fields it
actually needs; the layout hands back the indices of those fields *for this
robot*.  A policy written against ``("ball.position", "ball.linear_velocity",
"effector.position")`` is dimension-stable across every arm on the benchmark and
needs no retraining to be scored on a new one, while a policy that asks for
``robot.joint_position`` is honestly robot-specific and the layout says so.  A
request for a field a robot does not publish raises, rather than being padded
with zeros into something that trains but means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Sequence

# Field names are a published vocabulary, not free-form strings: a submission
# names them, a report records them, and two tasks that both expose
# "ball.position" have to mean the same thing by it.
BALL_POSITION = "ball.position"
BALL_LINEAR_VELOCITY = "ball.linear_velocity"
BALL_ANGULAR_VELOCITY = "ball.angular_velocity"
JOINT_POSITION = "robot.joint_position"
JOINT_VELOCITY = "robot.joint_velocity"
EFFECTOR_POSITION = "effector.position"
EFFECTOR_QUATERNION = "effector.quaternion"
EFFECTOR_LINEAR_VELOCITY = "effector.linear_velocity"

# Fields whose width does not depend on the embodiment.  A policy restricted to
# these is portable across robots by construction, which is the property that
# makes a cross-robot comparison possible at all.
ROBOT_INDEPENDENT_FIELDS = (
    BALL_POSITION,
    BALL_LINEAR_VELOCITY,
    BALL_ANGULAR_VELOCITY,
    EFFECTOR_POSITION,
    EFFECTOR_QUATERNION,
    EFFECTOR_LINEAR_VELOCITY,
)


class ObservationLayoutError(ValueError):
    """A layout was asked for something it does not publish."""


@dataclass(frozen=True)
class ObservationField:
    """One named block of the observation, with its own bounds.

    ``element_names`` is present for the blocks whose entries are individually
    meaningful -- joints, mostly -- so a report can say *which* joint an entry
    is rather than leaving a reader to count.
    """

    name: str
    size: int
    low: tuple[float, ...]
    high: tuple[float, ...]
    source: str
    element_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ObservationLayoutError("field name must be a non-empty string")
        if self.size < 1:
            raise ObservationLayoutError(f"{self.name}: size must be at least one")
        for bound in ("low", "high"):
            values = tuple(float(value) for value in getattr(self, bound))
            if len(values) != self.size:
                raise ObservationLayoutError(
                    f"{self.name}: {bound} must hold {self.size} values"
                )
            object.__setattr__(self, bound, values)
        if any(low > high for low, high in zip(self.low, self.high, strict=True)):
            raise ObservationLayoutError(f"{self.name}: low must not exceed high")
        if self.source not in {"task", "robot"}:
            raise ObservationLayoutError(f"{self.name}: source must be 'task' or 'robot'")
        names = tuple(str(name) for name in self.element_names)
        if names and len(names) != self.size:
            raise ObservationLayoutError(
                f"{self.name}: element_names must be empty or hold {self.size} names"
            )
        object.__setattr__(self, "element_names", names)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "size": self.size,
            "source": self.source,
            "low": list(self.low),
            "high": list(self.high),
        }
        if self.element_names:
            payload["element_names"] = list(self.element_names)
        return payload


@dataclass(frozen=True)
class ObservationLayout:
    """The ordered fields of one task-and-robot pairing."""

    fields: tuple[ObservationField, ...]

    def __post_init__(self) -> None:
        if not self.fields:
            raise ObservationLayoutError("a layout must publish at least one field")
        seen: set[str] = set()
        for field in self.fields:
            if field.name in seen:
                raise ObservationLayoutError(f"duplicate field {field.name!r}")
            seen.add(field.name)

    @property
    def size(self) -> int:
        """Width of the flat vector."""
        return sum(field.size for field in self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def field(self, name: str) -> ObservationField:
        for field in self.fields:
            if field.name == name:
                return field
        raise ObservationLayoutError(
            f"this robot does not publish {name!r}; it publishes {list(self.names)}"
        )

    def slice(self, name: str) -> slice:
        """Where ``name`` sits in the flat vector."""
        offset = 0
        for field in self.fields:
            if field.name == name:
                return slice(offset, offset + field.size)
            offset += field.size
        raise ObservationLayoutError(
            f"this robot does not publish {name!r}; it publishes {list(self.names)}"
        )

    def view(self, names: Iterable[str]) -> np.ndarray:
        """Flat indices of the requested fields, in the order requested.

        This is the whole point of the layout.  ``observation[layout.view(...)]``
        gives a policy exactly the fields it asked for, whatever robot produced
        the vector, and asking for something the robot has no notion of fails
        here rather than silently returning the wrong numbers.
        """
        requested = tuple(names)
        if not requested:
            raise ObservationLayoutError("view requires at least one field name")
        indices: list[int] = []
        for name in requested:
            span = self.slice(name)
            indices.extend(range(span.start, span.stop))
        return np.asarray(indices, dtype=int)

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Flat low/high vectors, in field order."""
        low: list[float] = []
        high: list[float] = []
        for field in self.fields:
            low.extend(field.low)
            high.extend(field.high)
        return tuple(low), tuple(high)

    def robot_independent(self) -> tuple[str, ...]:
        """The published fields whose width does not depend on the embodiment."""
        return tuple(name for name in self.names if name in ROBOT_INDEPENDENT_FIELDS)

    def pack(self, values: dict[str, Sequence[float]]) -> np.ndarray:
        """Assemble the flat vector from per-field values.

        Every field must be supplied at its declared size.  A short or long
        block is an error and not a reshape: a silently truncated joint vector
        is exactly the bug this class exists to make impossible.
        """
        blocks: list[np.ndarray] = []
        for field in self.fields:
            if field.name not in values:
                raise ObservationLayoutError(f"missing value for field {field.name!r}")
            block = np.asarray(values[field.name], dtype=np.float32).reshape(-1)
            if block.size != field.size:
                raise ObservationLayoutError(
                    f"{field.name}: expected {field.size} values, got {block.size}"
                )
            blocks.append(block)
        extra = set(values) - set(self.names)
        if extra:
            raise ObservationLayoutError(f"unknown fields: {sorted(extra)}")
        return np.concatenate(blocks).astype(np.float32)

    def build(self, state: Any) -> np.ndarray:
        """Pack one backend observation, reading the fields this layout declares.

        The backend observation is duck-typed on purpose: a MuJoCo backend, an
        Isaac backend and a replayed episode all expose ``ball`` and ``robot``
        with the same attribute names, and none of them should have to import
        this module to be readable by it.
        """
        ball = getattr(state, "ball", None)
        robot = getattr(state, "robot", None)
        if ball is None or robot is None:
            raise ObservationLayoutError(
                "observation state must expose 'ball' and 'robot'"
            )
        sources: dict[str, Sequence[float]] = {
            BALL_POSITION: ball.position,
            BALL_LINEAR_VELOCITY: ball.linear_velocity,
            BALL_ANGULAR_VELOCITY: ball.angular_velocity,
            JOINT_POSITION: robot.joint_positions,
            JOINT_VELOCITY: robot.joint_velocities,
            EFFECTOR_POSITION: robot.effector_position,
            EFFECTOR_QUATERNION: robot.effector_quaternion,
            EFFECTOR_LINEAR_VELOCITY: robot.effector_linear_velocity,
        }
        return self.pack({field.name: sources[field.name] for field in self.fields})

    def to_dict(self) -> dict[str, Any]:
        """The layout as it appears in a report, so a score is auditable."""
        return {
            "size": self.size,
            "fields": [field.to_dict() for field in self.fields],
            "robot_independent_fields": list(self.robot_independent()),
        }


def joint_space_layout(
    *,
    ball_position_low: Sequence[float],
    ball_position_high: Sequence[float],
    linear_velocity_limit: float,
    angular_velocity_limit: float,
    joint_names: Sequence[str],
    joint_position_low: Sequence[float],
    joint_position_high: Sequence[float],
    joint_velocity_limit: float,
    effector_position_low: Sequence[float],
    effector_position_high: Sequence[float],
) -> ObservationLayout:
    """The layout shared by every joint-space embodiment of a ball task.

    The field *order* is fixed here rather than per robot, so two robots differ
    only in the widths of their own blocks.  A reader of two reports can then
    line them up field by field instead of guessing at offsets.
    """
    joints = tuple(str(name) for name in joint_names)
    dof = len(joints)
    if dof < 1:
        raise ObservationLayoutError("a joint-space layout needs at least one joint")
    low = tuple(float(value) for value in joint_position_low)
    high = tuple(float(value) for value in joint_position_high)
    if len(low) != dof or len(high) != dof:
        raise ObservationLayoutError("joint bounds must hold one value per joint")
    linear = float(linear_velocity_limit)
    angular = float(angular_velocity_limit)
    rate = float(joint_velocity_limit)
    return ObservationLayout(
        (
            ObservationField(
                BALL_POSITION,
                3,
                tuple(float(value) for value in ball_position_low),
                tuple(float(value) for value in ball_position_high),
                "task",
            ),
            ObservationField(
                BALL_LINEAR_VELOCITY, 3, (-linear,) * 3, (linear,) * 3, "task"
            ),
            ObservationField(
                BALL_ANGULAR_VELOCITY, 3, (-angular,) * 3, (angular,) * 3, "task"
            ),
            ObservationField(JOINT_POSITION, dof, low, high, "robot", joints),
            ObservationField(
                JOINT_VELOCITY, dof, (-rate,) * dof, (rate,) * dof, "robot", joints
            ),
            ObservationField(
                EFFECTOR_POSITION,
                3,
                tuple(float(value) for value in effector_position_low),
                tuple(float(value) for value in effector_position_high),
                "robot",
            ),
            ObservationField(
                EFFECTOR_QUATERNION, 4, (-1.0,) * 4, (1.0,) * 4, "robot"
            ),
            ObservationField(
                EFFECTOR_LINEAR_VELOCITY, 3, (-linear,) * 3, (linear,) * 3, "robot"
            ),
        )
    )


__all__ = [
    "BALL_ANGULAR_VELOCITY",
    "BALL_LINEAR_VELOCITY",
    "BALL_POSITION",
    "EFFECTOR_LINEAR_VELOCITY",
    "EFFECTOR_POSITION",
    "EFFECTOR_QUATERNION",
    "JOINT_POSITION",
    "JOINT_VELOCITY",
    "ROBOT_INDEPENDENT_FIELDS",
    "ObservationField",
    "ObservationLayout",
    "ObservationLayoutError",
    "joint_space_layout",
]
