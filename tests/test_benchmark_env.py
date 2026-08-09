import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from multisport_sim.benchmark import register_envs

register_envs()


def test_registered_table_tennis_environment_passes_gymnasium_check() -> None:
    env = gym.make("MultiSportRobot/TableTennisReturn-v0")
    check_env(env.unwrapped, skip_render_check=True)


def test_reset_is_seeded_and_terminal_results_are_recorded() -> None:
    env = gym.make("MultiSportRobot/TableTennisReturn-v0")
    first_observation, first_info = env.reset(seed=41)
    second_observation, second_info = env.reset(seed=41)

    assert np.array_equal(first_observation, second_observation)
    assert first_info["shot_id"] == second_info["shot_id"]
    assert first_info["shot_bank_digest"] == second_info["shot_bank_digest"]

    action = np.array([-1.7, 0.0, 1.0, 0.707, 0.0, 0.0, 0.707], dtype=np.float32)
    for _ in range(500):
        observation, _, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            break
    else:  # pragma: no cover - protects a physics or Judge regression
        raise AssertionError("episode did not finish within its declared timeout")

    assert "episode_result" in info
    assert env.unwrapped.episode_records[-1] == info["episode_result"]
