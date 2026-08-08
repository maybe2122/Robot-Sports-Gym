"""Backend-neutral, JSON-friendly types for robot shot benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Literal, Mapping, Sequence, TypeAlias, cast


Vec2: TypeAlias = tuple[float, float]
Vec3: TypeAlias = tuple[float, float, float]
FailureReason: TypeAlias = Literal[
    "miss",
    "net",
    "own_side",
    "out",
    "floor",
    "timeout",
    "safety",
    "numerical",
]

_FAILURE_REASONS = frozenset(
    {"miss", "net", "own_side", "out", "floor", "timeout", "safety", "numerical"}
)


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _vector(value: object, *, size: int, field: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{field} must contain exactly {size} finite numbers")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{field} must contain exactly {size} finite numbers") from exc
    if len(items) != size:
        raise ValueError(f"{field} must contain exactly {size} finite numbers")
    return tuple(
        _finite_float(item, field=f"{field}[{index}]") for index, item in enumerate(items)
    )


def _non_empty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _tags(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("tags must be a sequence of non-empty strings")
    result = tuple(_non_empty_text(item, field="tags item") for item in value)
    if len(set(result)) != len(result):
        raise ValueError("tags must not contain duplicates")
    return result


@dataclass(frozen=True)
class SemanticContact:
    """A backend contact normalized to two semantic participants.

    Equality includes the optional contact sample. Event detectors deliberately key
    contacts by :attr:`pair`, so a moving persistent contact does not create a new hit.
    """

    pair: frozenset[str]
    position: Vec3 | None = None
    normal: Vec3 | None = None

    def __post_init__(self) -> None:
        if isinstance(self.pair, (str, bytes)):
            raise ValueError("contact pair must contain exactly two semantic names")
        try:
            pair = frozenset(self.pair)
        except TypeError as exc:
            raise ValueError("contact pair must contain exactly two semantic names") from exc
        if len(pair) != 2:
            raise ValueError("contact pair must contain exactly two distinct semantic names")
        for item in pair:
            _non_empty_text(item, field="contact semantic name")
        object.__setattr__(self, "pair", pair)
        if self.position is not None:
            object.__setattr__(
                self,
                "position",
                cast(Vec3, _vector(self.position, size=3, field="contact position")),
            )
        if self.normal is not None:
            object.__setattr__(
                self,
                "normal",
                cast(Vec3, _vector(self.normal, size=3, field="contact normal")),
            )

    @classmethod
    def between(
        cls,
        first: str,
        second: str,
        *,
        position: Vec3 | None = None,
        normal: Vec3 | None = None,
    ) -> SemanticContact:
        return cls(frozenset((first, second)), position=position, normal=normal)

    def involves(self, *categories: str) -> bool:
        """Return whether every requested semantic category participates."""
        return bool(categories) and all(category in self.pair for category in categories)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": sorted(self.pair),
            "position": None if self.position is None else list(self.position),
            "normal": None if self.normal is None else list(self.normal),
        }


@dataclass(frozen=True)
class TargetSpec:
    """A circular landing target in task-local table coordinates."""

    center_xy: Vec2
    radius_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "center_xy",
            cast(Vec2, _vector(self.center_xy, size=2, field="center_xy")),
        )
        radius = _finite_float(self.radius_m, field="radius_m")
        if radius <= 0.0:
            raise ValueError("radius_m must be greater than zero")
        object.__setattr__(self, "radius_m", radius)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TargetSpec:
        unknown = set(value) - {"center_xy", "radius_m"}
        if unknown:
            raise ValueError(f"unknown target fields: {', '.join(sorted(unknown))}")
        try:
            return cls(center_xy=value["center_xy"], radius_m=value["radius_m"])
        except KeyError as exc:
            raise ValueError(f"missing target field: {exc.args[0]}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"center_xy": list(self.center_xy), "radius_m": self.radius_m}


@dataclass(frozen=True)
class ShotSpec:
    """One versionable ball launch and its optional placement target."""

    shot_id: str
    sport: str
    level: str
    position: Vec3
    linear_velocity: Vec3
    angular_velocity: Vec3
    tags: tuple[str, ...] = ()
    target: TargetSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shot_id", _non_empty_text(self.shot_id, field="shot_id"))
        object.__setattr__(self, "sport", _non_empty_text(self.sport, field="sport"))
        object.__setattr__(self, "level", _non_empty_text(self.level, field="level"))
        object.__setattr__(
            self, "position", cast(Vec3, _vector(self.position, size=3, field="position"))
        )
        object.__setattr__(
            self,
            "linear_velocity",
            cast(Vec3, _vector(self.linear_velocity, size=3, field="linear_velocity")),
        )
        object.__setattr__(
            self,
            "angular_velocity",
            cast(Vec3, _vector(self.angular_velocity, size=3, field="angular_velocity")),
        )
        object.__setattr__(self, "tags", _tags(self.tags))
        if self.target is not None and not isinstance(self.target, TargetSpec):
            if not isinstance(self.target, Mapping):
                raise ValueError("target must be a TargetSpec, mapping, or null")
            object.__setattr__(self, "target", TargetSpec.from_mapping(self.target))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ShotSpec:
        fields = {
            "shot_id",
            "sport",
            "level",
            "position",
            "linear_velocity",
            "angular_velocity",
            "tags",
            "target",
        }
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown shot fields: {', '.join(sorted(unknown))}")
        required = fields - {"tags", "target"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"missing shot fields: {', '.join(sorted(missing))}")
        raw_target = value.get("target")
        if raw_target is not None and not isinstance(raw_target, (TargetSpec, Mapping)):
            raise ValueError("target must be a mapping or null")
        target = (
            raw_target
            if isinstance(raw_target, TargetSpec)
            else TargetSpec.from_mapping(raw_target)
            if raw_target is not None
            else None
        )
        return cls(
            shot_id=value["shot_id"],
            sport=value["sport"],
            level=value["level"],
            position=value["position"],
            linear_velocity=value["linear_velocity"],
            angular_velocity=value["angular_velocity"],
            tags=value.get("tags", ()),
            target=target,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "sport": self.sport,
            "level": self.level,
            "position": list(self.position),
            "linear_velocity": list(self.linear_velocity),
            "angular_velocity": list(self.angular_velocity),
            "tags": list(self.tags),
            "target": None if self.target is None else self.target.to_dict(),
        }


@dataclass(frozen=True)
class BallState:
    """Finite world/task-frame ball state exposed by a simulation adapter."""

    position: Vec3
    linear_velocity: Vec3
    angular_velocity: Vec3

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", cast(Vec3, _vector(self.position, size=3, field="position"))
        )
        object.__setattr__(
            self,
            "linear_velocity",
            cast(Vec3, _vector(self.linear_velocity, size=3, field="linear_velocity")),
        )
        object.__setattr__(
            self,
            "angular_velocity",
            cast(Vec3, _vector(self.angular_velocity, size=3, field="angular_velocity")),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position": list(self.position),
            "linear_velocity": list(self.linear_velocity),
            "angular_velocity": list(self.angular_velocity),
        }


@dataclass
class EpisodeResult:
    """Raw, interpretable result for one shot; tuples serialize as JSON arrays."""

    shot_id: str
    level: str | None = None
    tags: tuple[str, ...] = ()
    incoming_valid: bool = False
    hit: bool = False
    valid_return: bool = False
    target_hit: bool = False
    crossed_net: bool = False
    net_touch: bool = False
    contact_time_s: float | None = None
    landing_xy: Vec2 | None = None
    target_error_m: float | None = None
    incoming_speed_mps: float | None = None
    outgoing_speed_mps: float | None = None
    incoming_spin_radps: Vec3 | None = None
    episode_time_s: float | None = None
    failure_reason: FailureReason | None = None

    def __post_init__(self) -> None:
        self.shot_id = _non_empty_text(self.shot_id, field="shot_id")
        if self.level is not None:
            self.level = _non_empty_text(self.level, field="level")
        self.tags = _tags(self.tags)
        for field in (
            "contact_time_s",
            "target_error_m",
            "incoming_speed_mps",
            "outgoing_speed_mps",
            "episode_time_s",
        ):
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, _finite_float(value, field=field))
        if self.landing_xy is not None:
            self.landing_xy = cast(
                Vec2, _vector(self.landing_xy, size=2, field="landing_xy")
            )
        if self.incoming_spin_radps is not None:
            self.incoming_spin_radps = cast(
                Vec3,
                _vector(self.incoming_spin_radps, size=3, field="incoming_spin_radps"),
            )
        if self.failure_reason is not None and self.failure_reason not in _FAILURE_REASONS:
            raise ValueError(f"unknown failure_reason: {self.failure_reason!r}")
        self.validate()

    def validate(self) -> None:
        for field in (
            "incoming_valid",
            "hit",
            "valid_return",
            "target_hit",
            "crossed_net",
            "net_touch",
        ):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be a boolean")
        for field in (
            "contact_time_s",
            "target_error_m",
            "incoming_speed_mps",
            "outgoing_speed_mps",
            "episode_time_s",
        ):
            value = getattr(self, field)
            if value is not None:
                finite_value = _finite_float(value, field=field)
                if finite_value < 0.0:
                    raise ValueError(f"{field} must be non-negative")
        if self.landing_xy is not None:
            _vector(self.landing_xy, size=2, field="landing_xy")
        if self.incoming_spin_radps is not None:
            _vector(self.incoming_spin_radps, size=3, field="incoming_spin_radps")
        if self.failure_reason is not None and self.failure_reason not in _FAILURE_REASONS:
            raise ValueError(f"unknown failure_reason: {self.failure_reason!r}")
        if self.valid_return and not self.hit:
            raise ValueError("valid_return requires hit")
        if self.target_hit and not self.valid_return:
            raise ValueError("target_hit requires valid_return")
        if self.target_error_m is not None and not self.valid_return:
            raise ValueError("target_error_m requires valid_return")
        if self.failure_reason is not None and self.valid_return:
            raise ValueError("a valid return cannot have a failure_reason")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "shot_id": self.shot_id,
            "level": self.level,
            "tags": list(self.tags),
            "incoming_valid": self.incoming_valid,
            "hit": self.hit,
            "valid_return": self.valid_return,
            "target_hit": self.target_hit,
            "crossed_net": self.crossed_net,
            "net_touch": self.net_touch,
            "contact_time_s": self.contact_time_s,
            "landing_xy": None if self.landing_xy is None else list(self.landing_xy),
            "target_error_m": self.target_error_m,
            "incoming_speed_mps": self.incoming_speed_mps,
            "outgoing_speed_mps": self.outgoing_speed_mps,
            "incoming_spin_radps": (
                None if self.incoming_spin_radps is None else list(self.incoming_spin_radps)
            ),
            "episode_time_s": self.episode_time_s,
            "failure_reason": self.failure_reason,
        }
