"""Declared perturbations: observation noise, latency and domain randomization.

``return-v0`` labels its hardest level "Generalization" and then admits, in the
manifest, that ``domain_randomization``, ``observation_noise`` and ``latency``
are ``not_implemented_in_v0``.  Held-out trajectories alone do not test
generalization -- they test interpolation over a slightly wider distribution.
This module supplies the three missing conditions.

Three rules shape the design:

* **A perturbation is part of the task, not of the runner.**  What is applied
  comes from the shot bank's manifest, so two runs of the same level of the
  same bank are perturbed identically, and a report can state exactly what the
  policy was subjected to.
* **It is reproducible from the episode seed.**  Every draw comes from a
  generator seeded per episode, so a failure can be replayed.
* **It perturbs what the policy sees, never what the judge sees.**  Noise on an
  observation must not change whether the ball crossed the net.  The judge
  reads the backend directly; only :meth:`PerturbedBackend.observe` is affected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass, replace
from math import isfinite
from numbers import Real
from typing import Any

import numpy as np

from .robot import RobotObservation
from .sensors import CameraFrame, SensorReadings
from .types import BallState


def _non_negative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite, non-negative number")
    return result


def _steps(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _scale_range(value: object, *, field: str) -> tuple[float, float]:
    try:
        low, high = (float(item) for item in value)  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be two finite numbers") from exc
    if not (isfinite(low) and isfinite(high)) or low <= 0.0 or high < low:
        raise ValueError(f"{field} must be a positive, ordered range")
    return (low, high)


@dataclass(frozen=True)
class ObservationNoise:
    """Zero-mean Gaussian noise on what the policy reads.

    The ball terms apply to the state track; ``camera_pixel`` applies to the
    vision track.  A track therefore cannot be quietly spared: whichever
    observation a policy uses, the declared noise reaches it.
    """

    ball_position_m: float = 0.0
    ball_velocity_mps: float = 0.0
    ball_spin_radps: float = 0.0
    joint_position_rad: float = 0.0
    joint_velocity_radps: float = 0.0
    camera_pixel: float = 0.0

    def __post_init__(self) -> None:
        for field in (
            "ball_position_m",
            "ball_velocity_mps",
            "ball_spin_radps",
            "joint_position_rad",
            "joint_velocity_radps",
            "camera_pixel",
        ):
            object.__setattr__(self, field, _non_negative(getattr(self, field), field=field))

    @property
    def is_identity(self) -> bool:
        return not any(
            (
                self.ball_position_m,
                self.ball_velocity_mps,
                self.ball_spin_radps,
                self.joint_position_rad,
                self.joint_velocity_radps,
                self.camera_pixel,
            )
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "ball_position_m": self.ball_position_m,
            "ball_velocity_mps": self.ball_velocity_mps,
            "ball_spin_radps": self.ball_spin_radps,
            "joint_position_rad": self.joint_position_rad,
            "joint_velocity_radps": self.joint_velocity_radps,
            "camera_pixel": self.camera_pixel,
        }


@dataclass(frozen=True)
class Latency:
    """Delay, in control steps, on the way in and on the way out.

    Both directions exist because they fail differently.  An observation delay
    means the policy is aiming at where the ball *was*; an action delay means
    the arm starts moving after the policy decided it should.  A controller
    that compensates for one does not automatically survive the other.
    """

    observation_steps: int = 0
    action_steps: int = 0

    def __post_init__(self) -> None:
        for field in ("observation_steps", "action_steps"):
            object.__setattr__(self, field, _steps(getattr(self, field), field=field))

    @property
    def is_identity(self) -> bool:
        return self.observation_steps == 0 and self.action_steps == 0

    def to_dict(self) -> dict[str, int]:
        return {
            "observation_steps": self.observation_steps,
            "action_steps": self.action_steps,
        }


@dataclass(frozen=True)
class DomainRandomization:
    """Per-episode physical variation, drawn once at reset.

    Only quantities a real deployment genuinely varies in are randomized: ball
    mass within the ITTF tolerance band, air drag, and -- on the vision track --
    the camera mounting, which no installation reproduces to the millimetre.
    Table geometry is *not* randomized: it is regulated equipment, and moving it
    would change what the judge means by a legal return.
    """

    ball_mass_scale: tuple[float, float] = (1.0, 1.0)
    drag_scale: tuple[float, float] = (1.0, 1.0)
    camera_position_m: float = 0.0

    def __post_init__(self) -> None:
        for field in ("ball_mass_scale", "drag_scale"):
            object.__setattr__(self, field, _scale_range(getattr(self, field), field=field))
        object.__setattr__(
            self, "camera_position_m", _non_negative(self.camera_position_m, field="camera_position_m")
        )

    @property
    def is_identity(self) -> bool:
        return (
            self.ball_mass_scale == (1.0, 1.0)
            and self.drag_scale == (1.0, 1.0)
            and self.camera_position_m == 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ball_mass_scale": list(self.ball_mass_scale),
            "drag_scale": list(self.drag_scale),
            "camera_position_m": self.camera_position_m,
        }


@dataclass(frozen=True)
class PerturbationSpec:
    """Everything one level applies, as the manifest declares it."""

    observation_noise: ObservationNoise = ObservationNoise()
    latency: Latency = Latency()
    domain_randomization: DomainRandomization = DomainRandomization()

    @property
    def is_identity(self) -> bool:
        return (
            self.observation_noise.is_identity
            and self.latency.is_identity
            and self.domain_randomization.is_identity
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_noise": self.observation_noise.to_dict(),
            "latency": self.latency.to_dict(),
            "domain_randomization": self.domain_randomization.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> PerturbationSpec:
        """Build from a manifest block, treating an unimplemented level as identity.

        ``return-v0`` states ``"not_implemented_in_v0"`` for each condition.
        That string is honoured as "nothing is applied" rather than rejected, so
        the frozen bank keeps scoring exactly as it always has.
        """
        if not value:
            return cls()
        blocks: dict[str, Any] = {}
        for name, factory in (
            ("observation_noise", ObservationNoise),
            ("latency", Latency),
            ("domain_randomization", DomainRandomization),
        ):
            block = value.get(name)
            if isinstance(block, Mapping):
                blocks[name] = factory(**block)
            elif block not in (None, "not_implemented_in_v0"):
                raise ValueError(
                    f"manifest perturbation {name!r} must be an object or "
                    f"'not_implemented_in_v0', got {block!r}"
                )
        return cls(**blocks)


def for_level(manifest: Mapping[str, Any], level: str) -> PerturbationSpec:
    """The perturbations one level of one bank declares.

    Only levels that name their conditions get any: a bank whose manifest is
    silent perturbs nothing, which is what keeps ``return-v0`` frozen.
    """
    declared = manifest.get("perturbations")
    if isinstance(declared, Mapping):
        return PerturbationSpec.from_mapping(declared.get(level))
    if level == "L5":
        # The v0 manifest spells the L5 conditions out under their own key.
        return PerturbationSpec.from_mapping(manifest.get("l5_conditions"))
    return PerturbationSpec()


def _noisy_ball(ball: BallState, noise: ObservationNoise, rng: np.random.Generator) -> BallState:
    def jitter(values: tuple[float, float, float], sigma: float) -> tuple[float, float, float]:
        if sigma <= 0.0:
            return values
        return tuple(float(value + rng.normal(0.0, sigma)) for value in values)  # type: ignore[return-value]

    return BallState(
        position=jitter(ball.position, noise.ball_position_m),
        linear_velocity=jitter(ball.linear_velocity, noise.ball_velocity_mps),
        angular_velocity=jitter(ball.angular_velocity, noise.ball_spin_radps),
    )


def _noisy_robot(
    robot: RobotObservation, noise: ObservationNoise, rng: np.random.Generator
) -> RobotObservation:
    def jitter(values: tuple[float, ...], sigma: float) -> tuple[float, ...]:
        if sigma <= 0.0:
            return values
        return tuple(float(value + rng.normal(0.0, sigma)) for value in values)

    return replace(
        robot,
        joint_positions=jitter(robot.joint_positions, noise.joint_position_rad),
        joint_velocities=jitter(robot.joint_velocities, noise.joint_velocity_radps),
    )


def _noisy_sensors(
    readings: SensorReadings, noise: ObservationNoise, rng: np.random.Generator
) -> SensorReadings:
    if noise.camera_pixel <= 0.0:
        return readings
    perturbed = []
    for reading in readings.values():
        if isinstance(reading, CameraFrame):
            grain = rng.normal(0.0, noise.camera_pixel, size=reading.rgb.shape)
            rgb = np.clip(reading.rgb.astype(np.int16) + grain, 0, 255).astype(np.uint8)
            perturbed.append(replace(reading, rgb=rgb))
        else:
            perturbed.append(reading)
    return SensorReadings(perturbed)


def perturb_observation(
    observation: Any, noise: ObservationNoise, rng: np.random.Generator
) -> Any:
    """Return a copy of an observation with the declared noise applied.

    It works on both tracks by touching only the fields that exist: the state
    track's ``ball``, the vision track's ``sensors``, and the proprioception
    both carry.
    """
    if noise.is_identity:
        return observation
    if not is_dataclass(observation) or isinstance(observation, type):
        raise TypeError(
            "perturb_observation needs a dataclass observation; both tracks "
            f"publish one, got {type(observation).__name__}"
        )
    changes: dict[str, Any] = {}
    ball = getattr(observation, "ball", None)
    if ball is not None:
        changes["ball"] = _noisy_ball(ball, noise, rng)
    robot = getattr(observation, "robot", None)
    if robot is not None:
        changes["robot"] = _noisy_robot(robot, noise, rng)
    readings = getattr(observation, "sensors", None)
    if readings is not None:
        changes["sensors"] = _noisy_sensors(readings, noise, rng)
    return replace(observation, **changes) if changes else observation


class PerturbedBackend:
    """Wrap a backend so the policy sees a degraded, delayed view of it.

    Everything the judge reads -- contacts, ball state, safety, time -- is
    delegated untouched.  Only ``observe`` and ``apply_action`` are intercepted,
    which is the difference between "the task is harder" and "the scoring is
    wrong".
    """

    def __init__(self, backend: Any, spec: PerturbationSpec) -> None:
        self.backend = backend
        self.spec = spec
        self._rng = np.random.default_rng(0)
        self._observations: list[Any] = []
        self._actions: list[Any] = []
        self._applied: Any = None

    # -- episode lifecycle --------------------------------------------------

    def seed_episode(self, seed: int) -> None:
        """Reseed so an episode's perturbations can be replayed exactly."""
        self._rng = np.random.default_rng(seed)
        self._observations.clear()
        self._actions.clear()
        self._applied = None
        self._randomize()

    def reset(self) -> None:
        self.backend.reset()
        self._observations.clear()
        self._actions.clear()
        self._applied = None

    def _randomize(self) -> None:
        randomization = self.spec.domain_randomization
        if randomization.is_identity:
            return
        model = getattr(self.backend, "model", None)
        if model is None:
            return
        ball_geom = getattr(self.backend, "_ball_geom_id", None)
        if ball_geom is not None:
            body = int(model.geom_bodyid[ball_geom])
            scale = float(self._rng.uniform(*randomization.ball_mass_scale))
            # Store the nominal mass once; scaling a scaled mass would drift.
            nominal = getattr(self, "_nominal_ball_mass", None)
            if nominal is None:
                nominal = float(model.body_mass[body])
                self._nominal_ball_mass = nominal
            model.body_mass[body] = nominal * scale
        aerodynamics = getattr(self.backend, "aerodynamics", None)
        if aerodynamics is not None and randomization.drag_scale != (1.0, 1.0):
            # Drag scales with air density, so density is the physically honest
            # knob: it moves drag and Magnus together, the way weather does.
            nominal_density = getattr(self, "_nominal_density", None)
            if nominal_density is None:
                nominal_density = float(aerodynamics.atmosphere.density)
                self._nominal_density = nominal_density
            scale = float(self._rng.uniform(*randomization.drag_scale))
            aerodynamics.atmosphere = replace(
                aerodynamics.atmosphere, density=nominal_density * scale
            )
        if randomization.camera_position_m > 0.0 and model is not None and model.ncam:
            nominal_cameras = getattr(self, "_nominal_cam_pos", None)
            if nominal_cameras is None:
                nominal_cameras = np.array(model.cam_pos, copy=True)
                self._nominal_cam_pos = nominal_cameras
            model.cam_pos[:] = nominal_cameras + self._rng.normal(
                0.0, randomization.camera_position_m, size=nominal_cameras.shape
            )

    # -- the two intercepted calls -----------------------------------------

    def observe(self) -> Any:
        current = perturb_observation(
            self.backend.observe(), self.spec.observation_noise, self._rng
        )
        delay = self.spec.latency.observation_steps
        if delay <= 0:
            return current
        self._observations.append(current)
        # Before the buffer fills, the oldest reading available is the honest
        # answer: a real system starts with nothing and catches up.
        if len(self._observations) > delay:
            return self._observations.pop(0)
        return self._observations[0]

    def apply_action(self, action: Any) -> None:
        delay = self.spec.latency.action_steps
        if delay <= 0:
            self.backend.apply_action(action)
            return
        self._actions.append(action)
        if len(self._actions) > delay:
            self._applied = self._actions.pop(0)
        # Until the pipeline fills, the arm holds whatever it last received.
        self.backend.apply_action(self._applied)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)


__all__ = [
    "DomainRandomization",
    "Latency",
    "ObservationNoise",
    "PerturbationSpec",
    "PerturbedBackend",
    "for_level",
    "perturb_observation",
]
