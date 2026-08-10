"""Backend contract shared by benchmark runners."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..task_config import TaskFrame
from ..types import BallState, SemanticContact, ShotSpec


@runtime_checkable
class ShotBackend(Protocol):
    """One single-projectile simulator with backend-owned action semantics.

    Every quantity crossing this boundary -- shot launches, ball states and
    contact samples -- is expressed in the task frame, so the rule engine never
    sees a backend-specific placement.  A backend whose simulator world frame
    differs from the task frame declares that placement in :attr:`task_frame`
    and converts internally.
    """

    @property
    def task_frame(self) -> TaskFrame:
        ...

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
