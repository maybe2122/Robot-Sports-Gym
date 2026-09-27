"""L3 targets reach the policy: through env info, a wrapper, and reset()."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from multisport_sim.benchmark import register_envs
from multisport_sim.benchmark.envs import target_info
from multisport_sim.benchmark.policy_eval import PolicyController
from multisport_sim.benchmark.shot_bank import ShotBank
from multisport_sim.benchmark.wrappers import TargetObservation


def _l3_shot_id(task: str) -> str:
    return ShotBank.from_resource(split="dev", task=task).filter(level="L3")[0].shot_id


def test_the_env_reports_the_target_in_info() -> None:
    register_envs()
    env = gym.make("MultiSportRobot/BadmintonServe-v0").unwrapped
    try:
        shot_id = _l3_shot_id("badminton/serve-v0")
        _, info = env.reset(seed=0, options={"shot_id": shot_id})
        shot = env.shot_bank.get(shot_id)
        assert info["target"] == {
            "center": list(shot.target.center_xy),
            "radius_m": shot.target.radius_m,
        }
        l2 = env.shot_bank.filter(level="L2")[0].shot_id
        _, info = env.reset(seed=0, options={"shot_id": l2})
        assert info["target"] is None
    finally:
        env.close()


def test_the_wrapper_appends_the_target_to_the_observation() -> None:
    register_envs()
    env = TargetObservation(gym.make("MultiSportRobot/FootballKick-v0"))
    try:
        shot_id = _l3_shot_id("football/kick-v0")
        observation, info = env.reset(seed=0, options={"shot_id": shot_id})
        assert observation.shape == (20,)
        assert env.observation_space.contains(observation)
        assert observation[-4] == 1.0
        assert np.allclose(observation[-3:-1], info["target"]["center"])
        action = env.action_space.sample()
        observation, *_ = env.step(action)
        assert observation[-4] == 1.0
    finally:
        env.close()


class _TargetAware:
    def __init__(self) -> None:
        self.seen = "unset"

    def reset(self, *, seed=None, target=None):
        self.seen = target

    def act(self, observation):  # pragma: no cover - not stepped here
        return observation


class _Legacy:
    def __init__(self) -> None:
        self.calls = 0

    def reset(self, *, seed=None):
        self.calls += 1

    def act(self, observation):  # pragma: no cover - not stepped here
        return observation


def test_policy_reset_gets_the_target_only_if_it_asks() -> None:
    shot = ShotBank.from_resource(split="dev", task="table_tennis/return-v1").filter(level="L3")[0]
    aware, legacy = _TargetAware(), _Legacy()
    PolicyController(aware, policy_id="aware").reset(shot, seed=1)
    PolicyController(legacy, policy_id="legacy").reset(shot, seed=1)
    assert aware.seen == target_info(shot)
    assert legacy.calls == 1
