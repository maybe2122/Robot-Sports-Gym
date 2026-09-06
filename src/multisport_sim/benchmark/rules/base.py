"""Contracts shared by every sport's rule engine.

A judge turns a stream of semantic contacts and ball states into one
:class:`~multisport_sim.benchmark.types.EpisodeResult`.  Judges never read
backend geometry names, so the same instance scores a MuJoCo episode and an
Isaac episode identically.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Protocol, runtime_checkable

from ..types import BallState, EpisodeResult, FailureReason, SemanticContact, ShotSpec, TargetSpec


def finite(value: object, name: str) -> float:
    """Validate one finite number, the way every rule module needs it."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@runtime_checkable
class PlayingSurface(Protocol):
    """A rectangular playing surface split into a robot and an opponent half."""

    top_height_m: float
    net_plane_x_m: float
    height_tolerance_m: float
    minimum_normal_z: float

    def side_at(self, x: float, y: float) -> str | None:
        """Classify a surface point as ``robot``, ``opponent`` or outside."""
        ...


@dataclass(frozen=True)
class RectangularSurface:
    """Playing surface geometry the rules need, independent of render geometry.

    ``height_tolerance_m`` and ``minimum_normal_z`` keep a contact with the side
    or leg of the furniture from being scored as a landing on its top face.
    """

    length_m: float
    width_m: float
    top_height_m: float
    net_plane_x_m: float = 0.0
    center_y_m: float = 0.0
    height_tolerance_m: float = 0.05
    minimum_normal_z: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "length_m",
            "width_m",
            "top_height_m",
            "net_plane_x_m",
            "center_y_m",
            "height_tolerance_m",
            "minimum_normal_z",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if self.length_m <= 0.0 or self.width_m <= 0.0:
            raise ValueError("surface length and width must be greater than zero")
        if self.height_tolerance_m < 0.0:
            raise ValueError("height_tolerance_m must be non-negative")
        if not 0.0 <= self.minimum_normal_z <= 1.0:
            raise ValueError("minimum_normal_z must be between zero and one")

    def side_at(self, x: float, y: float) -> str | None:
        """Classify a top-surface point as ``robot``, ``opponent`` or outside."""
        relative_x = x - self.net_plane_x_m
        relative_y = y - self.center_y_m
        if abs(relative_y) > self.width_m / 2.0:
            return None
        if 0.0 < relative_x <= self.length_m / 2.0:
            return "opponent"
        if -self.length_m / 2.0 <= relative_x < 0.0:
            return "robot"
        return None


@runtime_checkable
class ShotJudge(Protocol):
    """Scores exactly one shot; reset before each episode, then update per step."""

    @property
    def done(self) -> bool:
        ...

    @property
    def result(self) -> EpisodeResult:
        ...

    def reset(
        self,
        shot: ShotSpec | str,
        *,
        target: TargetSpec | None = None,
        start_time_s: float = 0.0,
    ) -> EpisodeResult:
        ...

    def update(
        self,
        *,
        time_s: float,
        ball: BallState,
        contacts: Iterable[SemanticContact | frozenset[str]],
    ) -> EpisodeResult:
        ...

    def abort(
        self, *, failure_reason: FailureReason, time_s: float | None = None
    ) -> EpisodeResult:
        """Terminate an adapter-owned numerical or safety failure explicitly."""
        ...
