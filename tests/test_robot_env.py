"""The embodied Gymnasium environment and its safety termination path.

These tests need the Panda asset and skip without it.  They check the two
things a submission depends on: that the environment obeys the Gymnasium
contract with a joint-space action, and that breaking the safety envelope ends
the episode as a scored failure rather than being absorbed.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import multisport_sim.benchmark  # noqa: F401  (registers the environments)
from multisport_sim.benchmark.assets import FRANKA_PANDA, asset_available
from multisport_sim.benchmark.task_config import (
    TABLE_TENNIS_RETURN_PANDA_V1,
    TABLE_TENNIS_RETURN_V0,
)

ENV_ID = "MultiSportRobot/TableTennisReturn-Panda-v1"

pytestmark = pytest.mark.skipif(
    not asset_available(FRANKA_PANDA),
    reason="the Franka Panda asset is not installed; see docs/ROBOT_LAYER.md",
)


@pytest.fixture
def env():
    instance = gym.make(ENV_ID, split="dev").unwrapped
    yield instance
    instance.close()


class TestRegistration:
    def test_the_robot_task_is_registered_separately_from_the_fixture(self) -> None:
        assert ENV_ID in gym.registry
        assert "MultiSportRobot/TableTennisReturn-v0" in gym.registry

    def test_the_fixture_can_replay_the_embodied_tasks_bank(self, env) -> None:
        """Comparisons opt the fixture into the embodied task's versioned bank."""
        fixture = gym.make(
            "MultiSportRobot/TableTennisReturn-v0",
            split="dev",
            config=TABLE_TENNIS_RETURN_V0.__class__(
                bank_resource=TABLE_TENNIS_RETURN_PANDA_V1.bank_resource
            ),
        ).unwrapped
        try:
            assert env.shot_bank.digest == fixture.shot_bank.digest
            assert env.config.table == fixture.config.table
            assert env.config.reward == fixture.config.reward
        finally:
            fixture.close()


class TestGymnasiumContract:
    def test_it_passes_the_gymnasium_environment_checker(self) -> None:
        instance = gym.make(ENV_ID, split="dev").unwrapped
        try:
            check_env(instance, skip_render_check=True)
        finally:
            instance.close()

    def test_spaces_match_the_declared_task_configuration(self, env) -> None:
        config = TABLE_TENNIS_RETURN_PANDA_V1
        assert env.action_space.shape == (config.ACTION_DIM,)
        assert env.observation_space.shape == (config.OBSERVATION_DIM,)
        low, high = config.action_bounds()
        assert env.action_space.low == pytest.approx(np.array(low, dtype=np.float32))
        assert env.action_space.high == pytest.approx(np.array(high, dtype=np.float32))

    def test_the_action_space_is_the_arm_own_joint_range(self, env) -> None:
        """A policy must not be able to request motion the hardware cannot do."""
        limits = env.backend.adapter.joint_limits
        low, high = env.config.action_bounds()
        assert low == pytest.approx(limits.position_low, abs=1e-6)
        assert high == pytest.approx(limits.position_high, abs=1e-6)

    def test_the_observation_starts_inside_its_declared_bounds(self, env) -> None:
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)

    def test_reset_is_reproducible_for_one_seed(self, env) -> None:
        first, first_info = env.reset(seed=3)
        second, second_info = env.reset(seed=3)
        assert first_info["shot_id"] == second_info["shot_id"]
        assert first == pytest.approx(second)

    def test_a_named_shot_can_be_selected(self, env) -> None:
        shot_id = env.shot_bank[0].shot_id
        _, info = env.reset(seed=0, options={"shot_id": shot_id})
        assert info["shot_id"] == shot_id

    def test_an_unknown_reset_option_is_refused(self, env) -> None:
        with pytest.raises(ValueError, match="unknown reset options"):
            env.reset(seed=0, options={"nudge": 1.0})

    def test_stepping_before_reset_is_an_error(self, env) -> None:
        with pytest.raises(RuntimeError, match="reset must be called"):
            env.step(np.zeros(7, dtype=np.float32))

    def test_an_out_of_range_action_is_refused_not_clipped(self, env) -> None:
        env.reset(seed=0)
        with pytest.raises(ValueError, match="declared action space"):
            env.step(np.full(7, 99.0, dtype=np.float32))

    def test_a_wrongly_shaped_action_is_refused(self, env) -> None:
        env.reset(seed=0)
        with pytest.raises(ValueError, match=r"shape \(7,\)"):
            env.step(np.zeros(3, dtype=np.float32))


class TestEpisodeLifecycle:
    def test_holding_the_ready_pose_ends_in_a_scored_failure(self, env) -> None:
        """A parked arm must produce a miss or a timeout, never an exception."""
        shot_id = next(shot.shot_id for shot in env.shot_bank if shot.level == "L1")
        env.reset(seed=0, options={"shot_id": shot_id})
        ready = np.array(env.backend.adapter.ready_qpos, dtype=np.float32)
        for _ in range(1000):
            _, _, terminated, truncated, info = env.step(ready)
            if terminated or truncated:
                break
        else:  # pragma: no cover - the judge always terminates
            pytest.fail("the episode never finished")
        result = info["episode_result"]
        assert not result["valid_return"]
        assert result["failure_reason"] in {"miss", "timeout", "safety"}
        assert env.episode_records

    def test_info_carries_the_provenance_a_submission_needs(self, env) -> None:
        _, info = env.reset(seed=0)
        assert info["task_id"] == TABLE_TENNIS_RETURN_PANDA_V1.task_id
        assert info["robot_id"] == TABLE_TENNIS_RETURN_PANDA_V1.robot_id
        assert info["shot_bank_digest"]
        assert info["control_dt"] == pytest.approx(1.0 / env.config.control_hz, rel=1e-3)


class TestSafetyTermination:
    def test_a_violent_action_ends_the_episode_as_a_safety_failure(self, env) -> None:
        """The envelope must be enforced, and the failure must be scored.

        Stepping a position actuator to the far end of its range asks for a
        speed no Panda has.  The episode has to end with
        ``failure_reason="safety"`` -- counted in the denominator like any miss,
        not silently clipped into a slower motion.
        """
        shot_id = next(shot.shot_id for shot in env.shot_bank if shot.level == "L1")
        env.reset(seed=0, options={"shot_id": shot_id})
        extreme = np.array(env.config.action_bounds()[1], dtype=np.float32)
        for _ in range(200):
            _, _, terminated, truncated, info = env.step(extreme)
            if terminated or truncated:
                break
        else:  # pragma: no cover
            pytest.fail("the safety envelope never triggered")
        assert terminated
        assert info["episode_result"]["failure_reason"] == "safety"
        assert info["safety_violations"] >= 1
        kinds = {item["kind"] for item in info["safety_violation_details"]}
        assert kinds <= {
            "joint_position",
            "joint_velocity",
            "joint_torque",
            "collision",
            "workspace",
        }
        assert kinds

    def test_the_safety_counter_resets_with_the_backend(self, env) -> None:
        env.reset(seed=0)
        extreme = np.array(env.config.action_bounds()[1], dtype=np.float32)
        for _ in range(200):
            _, _, terminated, truncated, _ = env.step(extreme)
            if terminated or truncated:
                break
        assert env.backend.safety_violation_count >= 1
        env.reset(seed=1)
        assert env.backend.safety_violation_count == 0
