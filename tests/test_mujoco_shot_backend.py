from __future__ import annotations

import mujoco
import numpy as np
import pytest

from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
from multisport_sim.benchmark.controllers import (
    NoOpController,
    PaddleCommand,
    ScriptedPaddleController,
)
from multisport_sim.benchmark.runner import RunConfig, run_shots
from multisport_sim.benchmark.types import ShotSpec
from multisport_sim.scene import build_model, build_xml
from multisport_sim.specs import TABLE_TENNIS


REFERENCE_SHOT = ShotSpec(
    shot_id="tt-reference-real-contact",
    sport="table_tennis",
    level="L2",
    position=(1.8, 0.1, 1.1),
    linear_velocity=(-5.6, -0.35, 1.62),
    angular_velocity=(4.0, 55.0, 2.0),
    tags=("reference-fixture",),
)


def test_benchmark_blade_is_opt_in_and_default_scene_stays_uncontrolled() -> None:
    assert "table_tennis_benchmark_paddle" not in build_xml("table_tennis")
    default_model = build_model("table_tennis")
    assert default_model.nmocap == 0

    benchmark_model = build_model("table_tennis", benchmark_paddle=True)
    body_id = mujoco.mj_name2id(
        benchmark_model,
        mujoco.mjtObj.mjOBJ_BODY,
        "table_tennis_benchmark_paddle",
    )
    blade_id = mujoco.mj_name2id(
        benchmark_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "table_tennis_benchmark_paddle_blade",
    )
    assert benchmark_model.nmocap == 1
    assert body_id >= 0 and benchmark_model.body_mocapid[body_id] == 0
    assert blade_id >= 0

    with pytest.raises(ValueError, match="only available for the table_tennis"):
        build_model("campus", benchmark_paddle=True)


def test_launch_writes_the_free_joint_and_one_step_keeps_aerodynamics() -> None:
    backend = MujocoShotBackend()
    backend.launch_ball(REFERENCE_SHOT)
    assert backend.get_ball_state().position == pytest.approx(REFERENCE_SHOT.position)
    assert backend.get_ball_state().linear_velocity == pytest.approx(
        REFERENCE_SHOT.linear_velocity
    )
    assert backend.get_ball_state().angular_velocity == pytest.approx(
        REFERENCE_SHOT.angular_velocity
    )
    qpos = backend._ball_qpos_address
    np.testing.assert_allclose(backend.data.qpos[qpos + 3 : qpos + 7], (1.0, 0.0, 0.0, 0.0))

    backend.step()
    assert backend.time == pytest.approx(backend.timestep)
    # Drag opposes the negative launch X velocity and remains applied by the
    # existing Aerodynamics layer during the backend's one-tick step.
    assert backend.data.xfrc_applied[backend._ball_body_id, 0] > 0.0

    with pytest.raises(ValueError, match="cannot launch sport"):
        backend.launch_ball(
            ShotSpec(
                "wrong-sport",
                "tennis",
                "L1",
                REFERENCE_SHOT.position,
                REFERENCE_SHOT.linear_velocity,
                REFERENCE_SHOT.angular_velocity,
            )
        )


def test_backend_rejects_non_finite_wind() -> None:
    with pytest.raises(ValueError, match="wind"):
        MujocoShotBackend(wind=(float("nan"), 0.0, 0.0))


def test_backend_validates_and_normalizes_reference_fixture_actions() -> None:
    backend = MujocoShotBackend()
    backend.apply_action(PaddleCommand((-1.5, 0.1, 1.0), (2.0, 0.0, 0.0, 0.0)))
    observation = backend.observe()
    assert observation.paddle_position == pytest.approx((-1.5, 0.1, 1.0))
    assert observation.paddle_quaternion == pytest.approx((1.0, 0.0, 0.0, 0.0))

    backend.reset()
    assert backend.observe().paddle_position == pytest.approx((-1.58, 1.25, 1.0))

    with pytest.raises(ValueError, match="non-zero norm"):
        backend.apply_action(PaddleCommand((-1.5, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0)))
    with pytest.raises(TypeError, match="expects PaddleCommand"):
        backend.apply_action(object())


def _run_reference_trajectory() -> tuple[
    np.ndarray,
    tuple[tuple[frozenset[str], ...], ...],
    tuple[float, float, float] | None,
]:
    backend = MujocoShotBackend()
    controller = ScriptedPaddleController()
    backend.reset()
    controller.reset(REFERENCE_SHOT, seed=7)
    backend.launch_ball(REFERENCE_SHOT)

    states: list[tuple[float, ...]] = []
    contact_log: list[tuple[frozenset[str], ...]] = []
    previous_pairs: set[frozenset[str]] = set()
    hit = False
    first_post_hit_landing: tuple[float, float, float] | None = None
    for _ in range(1800):
        backend.apply_action(controller.act(backend.observe()))
        backend.step()
        state = backend.get_ball_state()
        states.append((*state.position, *state.linear_velocity, *state.angular_velocity))

        contacts = backend.semantic_contacts()
        active_pairs = {contact.pair for contact in contacts}
        contact_log.append(tuple(sorted(active_pairs, key=lambda pair: tuple(sorted(pair)))))
        rising_pairs = active_pairs - previous_pairs
        previous_pairs = active_pairs
        if frozenset(("ball", "robot_racket")) in rising_pairs:
            hit = True
            raw_pairs = {
                frozenset((int(contact.geom1), int(contact.geom2)))
                for contact in backend.data.contact[: backend.data.ncon]
                if float(contact.dist) <= 0.0
            }
            assert frozenset((backend._ball_geom_id, backend._paddle_geom_id)) in raw_pairs
            racket_sample = next(
                contact for contact in contacts if contact.involves("ball", "robot_racket")
            )
            assert racket_sample.position is not None
            assert racket_sample.normal is not None
            assert np.linalg.norm(racket_sample.normal) == pytest.approx(1.0)
        if hit and frozenset(("ball", "table")) in rising_pairs:
            landing = next(contact for contact in contacts if contact.involves("ball", "table"))
            assert landing.position is not None
            first_post_hit_landing = landing.position
            break

    assert hit, "reference fixture must make a real MuJoCo blade contact"
    return np.asarray(states), tuple(contact_log), first_post_hit_landing


def test_scripted_fixture_hits_and_returns_a_reference_shot_deterministically() -> None:
    first_states, first_contacts, first_landing = _run_reference_trajectory()
    second_states, second_contacts, second_landing = _run_reference_trajectory()

    np.testing.assert_allclose(second_states, first_states, rtol=0.0, atol=0.0)
    assert second_contacts == first_contacts
    assert first_landing is not None
    assert second_landing is not None
    assert second_landing == pytest.approx(first_landing)
    assert 0.0 < first_landing[0] <= TABLE_TENNIS.half_length
    assert abs(first_landing[1]) <= TABLE_TENNIS.half_width


def test_noop_controller_cannot_create_a_racket_contact() -> None:
    backend = MujocoShotBackend()
    controller = NoOpController()
    backend.reset()
    controller.reset(REFERENCE_SHOT)
    backend.launch_ball(REFERENCE_SHOT)

    for _ in range(1300):
        backend.apply_action(controller.act(backend.observe()))
        backend.step()
        assert not any(
            contact.involves("ball", "robot_racket")
            for contact in backend.semantic_contacts()
        )


def test_reference_fixture_integrates_with_runner_at_200_hz() -> None:
    output = run_shots(
        MujocoShotBackend(),
        ScriptedPaddleController(),
        [REFERENCE_SHOT],
        config=RunConfig(seed=7, control_hz=200.0, timeout_s=2.0),
    )
    result = output.results[0]
    assert output.control_decimation == 5
    assert result.incoming_valid
    assert result.hit
    assert result.crossed_net
    assert result.valid_return
    assert result.failure_reason is None
    assert result.contact_time_s is not None
    assert 0.6 < result.contact_time_s < 0.9
    assert result.landing_xy is not None
    assert 0.0 < result.landing_xy[0] <= TABLE_TENNIS.half_length
    assert abs(result.landing_xy[1]) <= TABLE_TENNIS.half_width
