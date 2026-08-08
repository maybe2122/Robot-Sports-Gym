"""Backend contract shared by benchmark runners."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import BallState, SemanticContact, ShotSpec


@runtime_checkable
class ShotBackend(Protocol):
    """One single-projectile simulator with backend-owned action semantics."""

    @property
    def time(self) -> float:
        ...

    @property
    def timestep(self) -> float:
        ...

    def reset(self) -> None:
        ...

    def launch_ball(self, shot: ShotSpec) -> None:
        ...

    def get_ball_state(self) -> BallState:
        ...

    def semantic_contacts(self) -> tuple[SemanticContact, ...]:
        ...

    def observe(self) -> object:
        ...

    def apply_action(self, action: object | None) -> None:
        ...

    def step(self, action: object | None = None) -> None:
        """Advance exactly one physics tick, optionally applying an action first."""
        ...


# Terminology retained from the design document for downstream integrations.
SportSimAdapter = ShotBackend
