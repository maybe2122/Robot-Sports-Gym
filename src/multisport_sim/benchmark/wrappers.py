"""Gymnasium wrappers that a learning setup typically needs.

:class:`TargetObservation` appends the shot's placement target to a flat
observation.  The environments publish the target in ``info["target"]`` --
task information, like the court -- but most RL libraries never read ``info``,
and an L3 policy that cannot see its target can only guess.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

TARGET_FIELDS = ("target.present", "target.center_u", "target.center_v", "target.radius_m")


class TargetObservation(gym.ObservationWrapper):
    """Append ``[present, centre_u, centre_v, radius]`` to a Box observation.

    ``present`` is 0 and the rest are 0 when the shot has no target, so the
    layout is the same on every level.  The centre is in the task's placement
    plane (see :func:`multisport_sim.benchmark.envs.target_info`).
    """

    def __init__(self, env: gym.Env, *, extent_m: float = 120.0) -> None:
        super().__init__(env)
        base = env.observation_space
        if not isinstance(base, spaces.Box) or len(base.shape) != 1:
            raise TypeError("TargetObservation needs a flat Box observation space")
        low = np.concatenate([base.low, np.array([0.0, -extent_m, -extent_m, 0.0])])
        high = np.concatenate([base.high, np.array([1.0, extent_m, extent_m, extent_m])])
        self.observation_space = spaces.Box(
            low=low.astype(base.dtype), high=high.astype(base.dtype), dtype=base.dtype
        )
        self._target = np.zeros(4, dtype=base.dtype)

    def _remember(self, info: dict[str, Any]) -> None:
        target = info.get("target")
        if target is None:
            self._target[:] = 0.0
        else:
            u, v = target["center"]
            self._target[:] = (1.0, u, v, target["radius_m"])

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self._remember(info)
        return self.observation(observation), info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate([observation, self._target]).astype(self.observation_space.dtype)


__all__ = ["TARGET_FIELDS", "TargetObservation"]
