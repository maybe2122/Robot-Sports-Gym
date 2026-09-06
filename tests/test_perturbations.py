"""Declared perturbations: noise, latency and randomization.

The property that matters most is the one that is easiest to get wrong: a
perturbation must change what the *policy* sees and nothing else.  If noise
reached the judge, a level would score differently for reasons that have
nothing to do with the policy, and the two tracks would stop being comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from multisport_sim.benchmark.perturbations import (
    DomainRandomization,
    Latency,
    ObservationNoise,
    PerturbationSpec,
    PerturbedBackend,
    for_level,
    perturb_observation,
)
from multisport_sim.benchmark.robot import RobotObservation
from multisport_sim.benchmark.sensors import CameraFrame, SensorReadings
from multisport_sim.benchmark.shot_bank import ShotBank
from multisport_sim.benchmark.types import BallState

BALL = BallState(
    position=(1.0, 0.2, 1.1),
    linear_velocity=(-5.0, 0.1, 2.0),
    angular_velocity=(0.0, 30.0, 0.0),
)
ROBOT = RobotObservation(
    time_s=0.1,
    joint_positions=(0.1,) * 7,
    joint_velocities=(0.0,) * 7,
    applied_torque=(0.0,) * 7,
    effector_position=(-1.6, 0.0, 1.0),
    effector_quaternion=(1.0, 0.0, 0.0, 0.0),
)


@dataclass(frozen=True)
class _Observation:
    """A stand-in with the fields both tracks share.

    It is a dataclass because the real observations are: perturbation rebuilds
    an observation rather than mutating one, so that a controller holding an
    earlier reading still sees what it was actually given.
    """

    ball: BallState | None = None
    robot: RobotObservation = ROBOT
    sensors: SensorReadings | None = None


def test_a_spec_refuses_values_that_cannot_mean_anything() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ObservationNoise(ball_position_m=-0.1)
    with pytest.raises(ValueError, match="non-negative integer"):
        Latency(observation_steps=-1)
    with pytest.raises(ValueError, match="positive, ordered range"):
        DomainRandomization(ball_mass_scale=(1.2, 0.8))
    with pytest.raises(ValueError, match="positive, ordered range"):
        DomainRandomization(drag_scale=(0.0, 1.0))


def test_an_empty_or_unimplemented_declaration_perturbs_nothing() -> None:
    """v0 says "not_implemented_in_v0"; that must read as identity, not as an error."""
    assert PerturbationSpec.from_mapping(None).is_identity
    assert PerturbationSpec.from_mapping({}).is_identity
    assert PerturbationSpec.from_mapping(
        {
            "domain_randomization": "not_implemented_in_v0",
            "observation_noise": "not_implemented_in_v0",
            "latency": "not_implemented_in_v0",
        }
    ).is_identity
    with pytest.raises(ValueError, match="must be an object"):
        PerturbationSpec.from_mapping({"latency": "later"})


def test_the_frozen_bank_still_perturbs_nothing_at_any_level() -> None:
    manifest = ShotBank.from_resource(split="dev").manifest

    for level in ("L0", "L1", "L2", "L3", "L4", "L5"):
        assert for_level(manifest, level).is_identity, level


def test_the_new_bank_perturbs_only_its_generalization_level() -> None:
    manifest = ShotBank.from_resource(split="dev", task="table_tennis/return-v1").manifest

    for level in ("L0", "L1", "L2", "L3", "L4"):
        assert for_level(manifest, level).is_identity, level
    generalization = for_level(manifest, "L5")
    assert not generalization.is_identity
    assert generalization.latency.observation_steps > 0
    assert generalization.observation_noise.ball_position_m > 0.0
    assert generalization.domain_randomization.ball_mass_scale != (1.0, 1.0)


def test_noise_reaches_the_ball_the_joints_and_the_camera() -> None:
    noise = ObservationNoise(
        ball_position_m=0.01,
        ball_velocity_mps=0.2,
        joint_position_rad=0.01,
        camera_pixel=6.0,
    )
    frame = CameraFrame(name="cam", time_s=0.0, rgb=np.full((4, 4, 3), 128, np.uint8))
    observation = _Observation(ball=BALL, sensors=SensorReadings([frame]))

    perturbed = perturb_observation(
        observation, noise, np.random.default_rng(0)
    )

    assert perturbed.ball.position != BALL.position
    assert perturbed.ball.linear_velocity != BALL.linear_velocity
    # Spin was not declared noisy, so it is untouched.
    assert perturbed.ball.angular_velocity == BALL.angular_velocity
    assert perturbed.robot.joint_positions != ROBOT.joint_positions
    assert perturbed.robot.applied_torque == ROBOT.applied_torque
    grain = perturbed.sensors.camera("cam").rgb
    assert grain.dtype == np.uint8
    assert not np.array_equal(grain, frame.rgb)
    assert abs(float(grain.mean()) - 128.0) < 6.0


def test_identity_noise_returns_the_observation_untouched() -> None:
    observation = _Observation(ball=BALL)

    assert perturb_observation(observation, ObservationNoise(), np.random.default_rng(0)) is observation


def test_the_same_seed_produces_the_same_noise() -> None:
    noise = ObservationNoise(ball_position_m=0.02)
    first = perturb_observation(_Observation(ball=BALL), noise, np.random.default_rng(7))
    second = perturb_observation(_Observation(ball=BALL), noise, np.random.default_rng(7))
    third = perturb_observation(_Observation(ball=BALL), noise, np.random.default_rng(8))

    assert first.ball.position == second.ball.position
    assert first.ball.position != third.ball.position


class _CountingBackend:
    """Records what the policy was handed and what the actuators received."""

    def __init__(self) -> None:
        self.tick = 0
        self.applied: list[object] = []

    def observe(self) -> _Observation:
        self.tick += 1
        return _Observation(
            ball=BallState(
                position=(float(self.tick), 0.0, 1.0),
                linear_velocity=(-5.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
            )
        )

    def apply_action(self, action) -> None:
        self.applied.append(action)

    def reset(self) -> None:
        self.tick = 0
        self.applied.clear()

    def get_ball_state(self) -> BallState:
        return BallState(
            position=(float(self.tick), 0.0, 1.0),
            linear_velocity=(-5.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )

    @property
    def timestep(self) -> float:
        return 0.002


def test_observation_latency_hands_the_policy_an_older_reading() -> None:
    backend = PerturbedBackend(
        _CountingBackend(), PerturbationSpec(latency=Latency(observation_steps=2))
    )
    backend.seed_episode(1)

    seen = [backend.observe().ball.position[0] for _ in range(5)]

    # The buffer warms up on the oldest available reading, then runs two behind.
    assert seen == [1.0, 1.0, 1.0, 2.0, 3.0]


def test_action_latency_delays_what_reaches_the_actuators() -> None:
    inner = _CountingBackend()
    backend = PerturbedBackend(inner, PerturbationSpec(latency=Latency(action_steps=1)))
    backend.seed_episode(1)

    for command in ("a", "b", "c"):
        backend.apply_action(command)

    # The first control step commands nothing new; the arm holds what it had.
    assert inner.applied == [None, "a", "b"]


def test_the_judge_still_reads_the_unperturbed_backend() -> None:
    """Noise on an observation must not change whether the ball crossed the net."""
    inner = _CountingBackend()
    backend = PerturbedBackend(
        inner,
        PerturbationSpec(observation_noise=ObservationNoise(ball_position_m=1.0)),
    )
    backend.seed_episode(3)

    observed = backend.observe()
    truth = backend.get_ball_state()

    assert observed.ball.position != truth.position
    assert truth.position == inner.get_ball_state().position
    # Everything else is delegated untouched.
    assert backend.timestep == inner.timestep


def test_reseeding_an_episode_clears_the_pipelines() -> None:
    backend = PerturbedBackend(
        _CountingBackend(),
        PerturbationSpec(latency=Latency(observation_steps=2, action_steps=1)),
    )
    backend.seed_episode(1)
    for _ in range(4):
        backend.observe()
        backend.apply_action("x")

    backend.reset()
    backend.seed_episode(1)

    # A fresh episode starts with an empty buffer, not with the last one's tail.
    assert backend.observe().ball.position[0] == pytest.approx(1.0)


def test_domain_randomization_is_reproducible_and_does_not_drift() -> None:
    """Scaling an already-scaled mass would compound over episodes."""

    class _Model:
        def __init__(self) -> None:
            self.body_mass = np.array([0.0027])
            self.geom_bodyid = np.array([0])
            self.cam_pos = np.zeros((0, 3))
            self.ncam = 0

    class _Backend:
        def __init__(self) -> None:
            self.model = _Model()
            self._ball_geom_id = 0

        def reset(self) -> None:
            return None

    backend = _Backend()
    spec = PerturbationSpec(
        domain_randomization=DomainRandomization(ball_mass_scale=(0.9, 1.1))
    )
    wrapper = PerturbedBackend(backend, spec)

    wrapper.seed_episode(5)
    first = float(backend.model.body_mass[0])
    wrapper.seed_episode(5)
    repeated = float(backend.model.body_mass[0])
    wrapper.seed_episode(6)
    different = float(backend.model.body_mass[0])

    assert first == pytest.approx(repeated)
    assert first != pytest.approx(different)
    for value in (first, different):
        assert 0.9 * 0.0027 <= value <= 1.1 * 0.0027


def test_a_spec_round_trips_through_its_report_block() -> None:
    spec = PerturbationSpec(
        observation_noise=ObservationNoise(ball_position_m=0.005, camera_pixel=3.0),
        latency=Latency(observation_steps=2, action_steps=1),
        domain_randomization=DomainRandomization(drag_scale=(0.9, 1.1)),
    )

    assert PerturbationSpec.from_mapping(spec.to_dict()) == spec
