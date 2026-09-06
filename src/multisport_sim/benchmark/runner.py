"""Deterministic single-shot execution shared by CLI and future environments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite, sqrt
from numbers import Real
from time import perf_counter

from .backends.base import ShotBackend
from .controllers import Controller
from .robot import RobotObservation
from .rules.base import ShotJudge
from .task_config import TABLE_TENNIS_RETURN_V0, ShotTaskConfig
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
    task: ShotTaskConfig = TABLE_TENNIS_RETURN_V0

    @classmethod
    def from_task_config(cls, task: ShotTaskConfig, *, seed: int = 0) -> RunConfig:
        """Run exactly the published task settings, overriding nothing."""
        if not isinstance(task, ShotTaskConfig):
            raise TypeError("task must be a ShotTaskConfig")
        return cls(
            seed=seed,
            control_hz=task.control_hz,
            timeout_s=task.timeout_s,
            task=task,
        )

    @property
    def effective_task(self) -> ShotTaskConfig:
        """The shared task configuration with this run's overrides applied."""
        return replace(
            self.task, control_hz=self.control_hz, timeout_s=self.timeout_s
        )

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.task, ShotTaskConfig):
            raise TypeError("task must be a ShotTaskConfig")
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
    """Per-episode results plus the execution facts a report has to carry.

    ``safety_violations`` and ``energy_joule`` are empty for a backend without a
    robot.  They are not zero: a mocap fixture has no safety envelope and no
    actuators, and reporting zero would claim it passed a check that was never
    run.
    """

    results: tuple[EpisodeResult, ...]
    physics_steps: int
    control_decimation: int
    episode_seeds: tuple[int, ...]
    safety_violations: tuple[int, ...] = ()
    energy_joule: tuple[float, ...] = ()
    # One entry per episode; ``None`` where the episode produced no blade
    # contact, which is different from a contact that happened to be centred.
    contact_offset_m: tuple[float | None, ...] = ()
    contact_speed_mps: tuple[float | None, ...] = ()
    # Wall-clock cost of the policy's own ``act`` call, in milliseconds.  It is
    # a property of this machine, not of the task, and the report says so.
    inference_latency_ms: tuple[float, ...] = ()


def _blade_frame_offset(
    contact_position: Sequence[float],
    blade_position: Sequence[float],
    blade_quaternion: Sequence[float],
) -> float:
    """Distance from the blade's centre to a contact, across the strike face.

    Only the in-face components count.  The third component is the ball's
    radius plus the blade's half thickness -- a constant that says nothing
    about aim -- while the in-face distance is exactly how far from the sweet
    spot the strike landed.  The blade site's z axis is its face normal, which
    is why this is a two-component norm in the site frame.
    """
    w, x, y, z = (float(value) for value in blade_quaternion)
    # Rows of the transpose: world vector -> blade frame.
    face_x = (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w))
    face_y = (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w))
    delta = tuple(
        float(contact) - float(centre)
        for contact, centre in zip(contact_position, blade_position, strict=True)
    )
    across = sum(axis * value for axis, value in zip(face_x, delta, strict=True))
    along = sum(axis * value for axis, value in zip(face_y, delta, strict=True))
    return sqrt(across * across + along * along)


def _judge_for(task: ShotTaskConfig) -> ShotJudge:
    """The judge the registry publishes for this task.

    Going through the registry rather than naming a judge here is what lets one
    runner score every sport: a task that adds a sport adds an entry, not a
    branch in this file.
    """
    from . import tasks  # noqa: F401  (importing publishes the built-in tasks)
    from .registry import get_task

    return get_task(task.task_id).make_judge(task)


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
    # An embodied backend can report envelope breaches and actuator work; a
    # mocap fixture cannot, and must not be made to look as if it could.
    embodied = hasattr(backend, "safety_violations") and hasattr(
        backend, "robot_observation"
    )
    safety_counts: list[int] = []
    energies: list[float] = []
    contact_offsets: list[float | None] = []
    contact_speeds: list[float | None] = []
    latencies: list[float] = []

    # A backend that perturbs what the policy sees draws from the episode seed,
    # so a bad episode can be replayed exactly rather than described.
    seedable = hasattr(backend, "seed_episode")

    for shot, episode_seed in zip(shots, episode_seeds, strict=True):
        backend.reset()
        if seedable:
            backend.seed_episode(episode_seed)
        controller.reset(shot, seed=episode_seed)
        backend.launch_ball(shot)
        episode_safety = 0
        episode_energy = 0.0
        episode_offset: float | None = None
        episode_speed: float | None = None
        episode_latency: list[float] = []

        judge = _judge_for(task)
        judge.reset(shot)

        # The extra steps allow floating-point time to reach the exact timeout.
        maximum_steps = task.max_physics_steps(backend.timestep)
        for episode_step in range(maximum_steps):
            if episode_step % decimation == 0:
                observation = backend.observe()
                started = perf_counter()
                action = controller.act(observation)
                episode_latency.append((perf_counter() - started) * 1000.0)
                backend.apply_action(action)
            try:
                backend.step()
            except FloatingPointError:
                judge.abort(
                    failure_reason="numerical",
                    time_s=(episode_step + 1) * backend.timestep,
                )
                break
            total_steps += 1
            if embodied:
                # Safety is evaluated every physics step, not once per control
                # step: a limit broken mid-decimation is still broken.
                violations = backend.safety_violations()
                observation: RobotObservation = backend.robot_observation()
                episode_energy += observation.mechanical_power_w() * backend.timestep
                if violations:
                    episode_safety += len(violations)
                    judge.abort(failure_reason="safety", time_s=backend.time)
                    break
            try:
                current_time = backend.time
                if not isfinite(current_time):
                    raise ValueError("backend time is not finite")
                contacts = backend.semantic_contacts()
                if embodied and episode_offset is None:
                    # The first blade contact is the strike; later ones are the
                    # ball leaving, and scoring those would flatter the aim.
                    strike = next(
                        (
                            contact
                            for contact in contacts
                            if "robot_racket" in contact.pair and contact.position is not None
                        ),
                        None,
                    )
                    if strike is not None:
                        blade = backend.robot_observation()
                        episode_offset = _blade_frame_offset(
                            strike.position,
                            blade.effector_position,
                            blade.effector_quaternion,
                        )
                        episode_speed = sqrt(
                            sum(value * value for value in blade.effector_linear_velocity)
                        )
                judge.update(
                    time_s=current_time,
                    ball=backend.get_ball_state(),
                    contacts=contacts,
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
        safety_counts.append(episode_safety)
        energies.append(episode_energy)
        contact_offsets.append(episode_offset)
        contact_speeds.append(episode_speed)
        latencies.extend(episode_latency)

    return RunOutput(
        tuple(results),
        total_steps,
        decimation,
        episode_seeds,
        safety_violations=tuple(safety_counts) if embodied else (),
        energy_joule=tuple(energies) if embodied else (),
        contact_offset_m=tuple(contact_offsets) if embodied else (),
        contact_speed_mps=tuple(contact_speeds) if embodied else (),
        inference_latency_ms=tuple(latencies),
    )
