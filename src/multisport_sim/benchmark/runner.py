"""Deterministic single-shot execution shared by CLI and future environments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite
from numbers import Real
from typing import Sequence

from .backends.base import ShotBackend
from .controllers import Controller
from .rules.table_tennis import TableTennisReturnJudge
from .task_config import TABLE_TENNIS_RETURN_V0, TableTennisReturnTaskConfig
from .types import EpisodeResult, ShotSpec


@dataclass(frozen=True)
class RunConfig:
    """Execution settings that do not alter a frozen shot distribution.

    ``control_hz`` and ``timeout_s`` default to the shared task configuration
    and may be overridden for diagnostics; ``task`` supplies everything else the
    run and the judge must agree on with other backends.
    """

    seed: int = 0
    control_hz: float = TABLE_TENNIS_RETURN_V0.control_hz
    timeout_s: float = TABLE_TENNIS_RETURN_V0.timeout_s
    task: TableTennisReturnTaskConfig = TABLE_TENNIS_RETURN_V0

    @classmethod
    def from_task_config(
        cls, task: TableTennisReturnTaskConfig, *, seed: int = 0
    ) -> RunConfig:
        """Run exactly the published task settings, overriding nothing."""
        if not isinstance(task, TableTennisReturnTaskConfig):
            raise TypeError("task must be a TableTennisReturnTaskConfig")
        return cls(
            seed=seed,
            control_hz=task.control_hz,
            timeout_s=task.timeout_s,
            task=task,
        )

    @property
    def effective_task(self) -> TableTennisReturnTaskConfig:
        """The shared task configuration with this run's overrides applied."""
        return replace(
            self.task, control_hz=self.control_hz, timeout_s=self.timeout_s
        )

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.task, TableTennisReturnTaskConfig):
            raise TypeError("task must be a TableTennisReturnTaskConfig")
        if (
            isinstance(self.control_hz, bool)
            or not isinstance(self.control_hz, Real)
            or not isfinite(float(self.control_hz))
            or self.control_hz <= 0.0
        ):
            raise ValueError("control_hz must be greater than zero")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, Real)
            or not isfinite(float(self.timeout_s))
            or self.timeout_s <= 0.0
        ):
            raise ValueError("timeout_s must be greater than zero")


@dataclass(frozen=True)
class RunOutput:
    results: tuple[EpisodeResult, ...]
    physics_steps: int
    control_decimation: int
    episode_seeds: tuple[int, ...]


def _episode_seeds(seed: int, count: int) -> tuple[int, ...]:
    """Derive cross-version 63-bit seeds from a documented SHA-256 namespace."""
    values: list[int] = []
    for index in range(count):
        payload = f"multisport-shot-seed-v0:{seed}:{index}".encode()
        values.append(int.from_bytes(sha256(payload).digest()[:8], "big") & ((1 << 63) - 1))
    return tuple(values)


def run_shots(
    backend: ShotBackend,
    controller: Controller,
    shots: Sequence[ShotSpec],
    *,
    config: RunConfig | None = None,
) -> RunOutput:
    """Execute every supplied fixed shot once and preserve its input ordering."""
    settings = config or RunConfig()
    if not shots:
        raise ValueError("at least one shot is required")
    if not isfinite(backend.timestep) or backend.timestep <= 0.0:
        raise ValueError("backend timestep must be a positive finite number")

    task = settings.effective_task
    decimation = task.decimation(backend.timestep)
    episode_seeds = _episode_seeds(settings.seed, len(shots))
    total_steps = 0
    results: list[EpisodeResult] = []

    for shot, episode_seed in zip(shots, episode_seeds, strict=True):
        backend.reset()
        controller.reset(shot, seed=episode_seed)
        backend.launch_ball(shot)

        judge = TableTennisReturnJudge(
            timeout_s=task.timeout_s, table_spec=task.table
        )
        judge.reset(shot)

        # The extra steps allow floating-point time to reach the exact timeout.
        maximum_steps = task.max_physics_steps(backend.timestep)
        for episode_step in range(maximum_steps):
            if episode_step % decimation == 0:
                backend.apply_action(controller.act(backend.observe()))
            try:
                backend.step()
            except FloatingPointError:
                judge.abort(
                    failure_reason="numerical",
                    time_s=(episode_step + 1) * backend.timestep,
                )
                break
            total_steps += 1
            try:
                current_time = backend.time
                if not isfinite(current_time):
                    raise ValueError("backend time is not finite")
                judge.update(
                    time_s=current_time,
                    ball=backend.get_ball_state(),
                    contacts=backend.semantic_contacts(),
                )
            except (FloatingPointError, ValueError):
                judge.abort(
                    failure_reason="numerical",
                    time_s=(episode_step + 1) * backend.timestep,
                )
                break
            if judge.done:
                break
        else:  # pragma: no cover - defensive guard around third-party adapters
            raise RuntimeError(
                f"judge did not terminate shot {shot.shot_id!r} within "
                f"{settings.timeout_s:.3f} s"
            )

        results.append(judge.result)

    return RunOutput(tuple(results), total_steps, decimation, episode_seeds)
