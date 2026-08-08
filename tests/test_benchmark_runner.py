from __future__ import annotations

import pytest

from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
from multisport_sim.benchmark.controllers import NoOpController, ScriptedPaddleController
from multisport_sim.benchmark.runner import RunConfig, run_shots
from multisport_sim.benchmark.shot_bank import ShotBank
from multisport_sim.benchmark.types import ShotSpec


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
