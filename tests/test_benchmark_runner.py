from __future__ import annotations

import pytest

from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
from multisport_sim.benchmark.controllers import NoOpController, ScriptedPaddleController
from multisport_sim.benchmark.robot import RobotObservation
from multisport_sim.benchmark.runner import RunConfig, run_shots
from multisport_sim.benchmark.shot_bank import ShotBank
from multisport_sim.benchmark.types import BallState, SemanticContact, ShotSpec

REFERENCE_SHOT = ShotSpec(
    shot_id="runner-reference",
    sport="table_tennis",
    level="L2",
    position=(1.8, 0.1, 1.1),
    linear_velocity=(-5.6, -0.35, 1.62),
    angular_velocity=(4.0, 55.0, 2.0),
    tags=("reference",),
)


def test_runner_records_a_real_return_and_a_noop_miss() -> None:
    backend = MujocoShotBackend()
    scripted = run_shots(backend, ScriptedPaddleController(), [REFERENCE_SHOT])
    noop = run_shots(backend, NoOpController(), [REFERENCE_SHOT])

    returned = scripted.results[0]
    assert returned.incoming_valid
    assert returned.hit
    assert returned.crossed_net
    assert returned.valid_return
    assert returned.landing_xy is not None and returned.landing_xy[0] > 0.0
    assert returned.incoming_speed_mps is not None and returned.incoming_speed_mps > 3.0
    assert returned.outgoing_speed_mps is not None and returned.outgoing_speed_mps > 3.0

    missed = noop.results[0]
    assert missed.incoming_valid
    assert not missed.hit
    assert missed.failure_reason == "miss"
    assert scripted.control_decimation == 5


def test_runner_replay_is_exactly_deterministic() -> None:
    backend = MujocoShotBackend()
    first = run_shots(
        backend,
        ScriptedPaddleController(),
        [REFERENCE_SHOT],
        config=RunConfig(seed=9),
    )
    second = run_shots(
        backend,
        ScriptedPaddleController(),
        [REFERENCE_SHOT],
        config=RunConfig(seed=9),
    )

    assert second.results[0].to_dict() == first.results[0].to_dict()
    assert second.physics_steps == first.physics_steps
    assert second.episode_seeds == first.episode_seeds


def test_scripted_fixture_intercepts_dev_l1_and_returns_at_least_one_l2_shot() -> None:
    bank = ShotBank.from_resource(split="dev")
    backend = MujocoShotBackend()

    l1 = run_shots(backend, ScriptedPaddleController(), bank.filter(level="L1"))
    l2 = run_shots(backend, ScriptedPaddleController(), bank.filter(level="L2"))

    assert all(result.hit for result in l1.results)
    assert any(result.valid_return for result in l2.results)


def test_runner_rejects_empty_input_and_invalid_timing() -> None:
    with pytest.raises(ValueError, match="at least one shot"):
        run_shots(MujocoShotBackend(), NoOpController(), [])
    with pytest.raises(ValueError, match="control_hz"):
        RunConfig(control_hz=0.0)
    with pytest.raises(ValueError, match="timeout_s"):
        RunConfig(timeout_s=0.0)
    with pytest.raises(ValueError, match="control_hz"):
        RunConfig(control_hz=float("nan"))
    with pytest.raises(ValueError, match="timeout_s"):
        RunConfig(timeout_s=float("inf"))
    with pytest.raises(ValueError, match="seed"):
        RunConfig(seed=True)


class _NonFiniteBackend:
    timestep = 0.001

    def __init__(self) -> None:
        self.time = 0.0

    def reset(self) -> None:
        self.time = 0.0

    def launch_ball(self, shot: ShotSpec) -> None:
        del shot

    def observe(self) -> object:
        return object()

    def apply_action(self, action: object | None) -> None:
        del action

    def step(self, action: object | None = None) -> None:
        del action
        self.time += self.timestep

    def get_ball_state(self) -> object:
        raise ValueError("ball position must be finite")

    def semantic_contacts(self) -> tuple[object, ...]:
        return ()


def test_runner_counts_non_finite_backend_state_as_an_episode_failure() -> None:
    output = run_shots(_NonFiniteBackend(), NoOpController(), [REFERENCE_SHOT])

    assert output.results[0].failure_reason == "numerical"
    assert output.results[0].episode_time_s == pytest.approx(0.001)


class _StubEmbodiedBackend:
    """A scripted embodied backend: no simulator, exact numbers, no asset.

    It exists so the runner's contact-error and latency bookkeeping can be
    checked against values worked out by hand.  A real MuJoCo run cannot do
    that: the contact position is whatever the solver produced.
    """

    timestep = 0.005

    def __init__(self, *, contact_at_step: int | None = 2) -> None:
        self.contact_at_step = contact_at_step
        self._step = 0
        self.time = 0.0

    def reset(self) -> None:
        self._step = 0
        self.time = 0.0

    def launch_ball(self, shot) -> None:
        del shot

    def observe(self) -> object:
        return object()

    def apply_action(self, action) -> None:
        del action

    def step(self) -> None:
        self._step += 1
        self.time = self._step * self.timestep

    def get_ball_state(self) -> BallState:
        return BallState(
            position=(1.0 - self._step * 0.05, 0.0, 1.0),
            linear_velocity=(-5.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )

    def semantic_contacts(self) -> tuple[SemanticContact, ...]:
        if self.contact_at_step is None or self._step < self.contact_at_step:
            return ()
        # 3 cm across the face and 4 cm along it: a 5 cm off-centre strike.
        return (
            SemanticContact.between(
                "ball", "robot_racket", position=(0.03, 0.04, 1.0), normal=(1.0, 0.0, 0.0)
            ),
        )

    def safety_violations(self) -> tuple:
        return ()

    def robot_observation(self) -> RobotObservation:
        return RobotObservation(
            time_s=self.time,
            joint_positions=(0.0,) * 7,
            joint_velocities=(0.0,) * 7,
            applied_torque=(0.0,) * 7,
            effector_position=(0.0, 0.0, 1.0),
            effector_quaternion=(1.0, 0.0, 0.0, 0.0),
            effector_linear_velocity=(3.0, 4.0, 0.0),
        )


def test_the_runner_measures_how_far_off_centre_the_strike_landed() -> None:
    backend = _StubEmbodiedBackend()
    output = run_shots(backend, NoOpController(), [REFERENCE_SHOT])

    # The face is the blade site's x-y plane, so the 1 m offset along the
    # normal is excluded and the in-face 3-4-5 triangle is what is reported.
    assert output.contact_offset_m[0] == pytest.approx(0.05)
    assert output.contact_speed_mps[0] == pytest.approx(5.0)


def test_an_episode_without_a_strike_reports_no_contact_error_rather_than_zero() -> None:
    """Zero would mean a perfectly centred hit; there was no hit at all."""
    output = run_shots(
        _StubEmbodiedBackend(contact_at_step=None), NoOpController(), [REFERENCE_SHOT]
    )

    assert output.contact_offset_m == (None,)
    assert output.contact_speed_mps == (None,)


def test_only_the_first_blade_contact_counts() -> None:
    """Later contacts are the ball leaving; scoring them would flatter the aim."""
    backend = _StubEmbodiedBackend(contact_at_step=1)
    output = run_shots(backend, NoOpController(), [REFERENCE_SHOT])

    assert output.contact_offset_m[0] == pytest.approx(0.05)


def test_the_runner_times_the_policy_and_not_the_physics() -> None:
    backend = _StubEmbodiedBackend()
    output = run_shots(backend, NoOpController(), [REFERENCE_SHOT])

    # One sample per control step, and a no-op controller is fast.
    assert len(output.inference_latency_ms) > 0
    assert all(value >= 0.0 for value in output.inference_latency_ms)
    assert max(output.inference_latency_ms) < 50.0


def test_a_mocap_backend_reports_no_contact_error_at_all() -> None:
    """It has no blade pose to measure against, so the lists stay empty."""
    output = run_shots(MujocoShotBackend(), NoOpController(), [REFERENCE_SHOT])

    assert output.contact_offset_m == ()
    assert output.contact_speed_mps == ()
    assert len(output.inference_latency_ms) > 0
