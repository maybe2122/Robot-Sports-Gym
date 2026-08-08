"""Deterministic single-shot execution shared by CLI and future environments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from numbers import Real
from typing import Sequence

from .backends.base import ShotBackend
from .controllers import Controller
from .rules.table_tennis import TableTennisReturnJudge
from .types import EpisodeResult, ShotSpec


@dataclass(frozen=True)
class RunConfig:
    """Execution settings that do not alter a frozen shot distribution."""

    seed: int = 0
    control_hz: float = 200.0
    timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
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

    decimation = max(1, round(1.0 / (settings.control_hz * backend.timestep)))
    episode_seeds = _episode_seeds(settings.seed, len(shots))
    total_steps = 0
    results: list[EpisodeResult] = []

    for shot, episode_seed in zip(shots, episode_seeds, strict=True):
        backend.reset()
        controller.reset(shot, seed=episode_seed)
        backend.launch_ball(shot)

        judge = TableTennisReturnJudge(timeout_s=settings.timeout_s)
        judge.reset(shot)

        # The extra steps allow floating-point time to reach the exact timeout.
        maximum_steps = int(settings.timeout_s / backend.timestep) + 2
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
