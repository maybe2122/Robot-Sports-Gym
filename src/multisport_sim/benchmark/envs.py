"""Gymnasium environments backed by the versioned Shot Skill core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import sqrt
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .backends.mujoco import MujocoTableTennisBackend
from .controllers import PaddleCommand
from .rules.table_tennis import TableTennisReturnJudge
from .shot_bank import ShotBank
from .task_config import TABLE_TENNIS_RETURN_V0, TableTennisReturnTaskConfig
from .types import EpisodeResult, ShotSpec


class TableTennisReturnEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-shot MuJoCo table-tennis return task.

    The action is a task-frame mocap paddle pose ``[x, y, z, qw, qx, qy, qz]``.
    Position is measured in metres and the quaternion is normalized before it is
    sent to MuJoCo.  The observation is ``[ball position, linear velocity,
    angular velocity, paddle position, paddle quaternion]`` in the same frame.
    Spaces, control rate, timeout and reward terms all come from the shared
    :class:`~multisport_sim.benchmark.task_config.TableTennisReturnTaskConfig`,
    so an Isaac implementation of the same task cannot drift from this one.
    This fixture is deliberately marked experimental: it is an API and
    rule-engine integration point, not a robot embodiment.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        *,
        split: str | None = None,
        control_hz: float | None = None,
        timeout_s: float | None = None,
        config: TableTennisReturnTaskConfig = TABLE_TENNIS_RETURN_V0,
    ) -> None:
        super().__init__()
        if not isinstance(config, TableTennisReturnTaskConfig):
            raise TypeError("config must be a TableTennisReturnTaskConfig")
        overrides = {
            name: value
            for name, value in (
                ("split", split),
                ("control_hz", control_hz),
                ("timeout_s", timeout_s),
            )
            if value is not None
        }
        # The dataclass validates every override, including a non-finite rate.
        self.config = replace(config, **overrides) if overrides else config
        self.shot_bank = ShotBank.from_resource(split=self.config.split)
        self.backend = MujocoTableTennisBackend()
        self.control_hz = self.config.control_hz
        self.timeout_s = self.config.timeout_s
        self._control_decimation = self.config.decimation(self.backend.timestep)
        action_low, action_high = self.config.action_bounds()
        self.action_space = spaces.Box(
            low=np.array(action_low, dtype=np.float32),
            high=np.array(action_high, dtype=np.float32),
            dtype=np.float32,
        )
        observation_low, observation_high = self.config.observation_bounds()
        self.observation_space = spaces.Box(
            low=np.array(observation_low, dtype=np.float32),
            high=np.array(observation_high, dtype=np.float32),
            dtype=np.float32,
        )
        self._judge: TableTennisReturnJudge | None = None
        self._shot: ShotSpec | None = None
        self._elapsed_steps = 0
        self._episode_records: list[dict[str, Any]] = []

    @property
    def episode_records(self) -> tuple[dict[str, Any], ...]:
        """Completed episode results, ready to serialize as JSON."""
        return tuple(self._episode_records)

    def _observation(self) -> np.ndarray:
        state = self.backend.observe()
        return np.asarray(
            (*state.ball.position, *state.ball.linear_velocity, *state.ball.angular_velocity,
             *state.paddle_position, *state.paddle_quaternion),
            dtype=np.float32,
        )

    @staticmethod
    def _result_info(result: EpisodeResult) -> dict[str, Any]:
        return result.to_dict()

    def _info(self) -> dict[str, Any]:
        assert self._shot is not None
        return {
            "task_id": self.config.task_id,
            "shot_id": self._shot.shot_id,
            "shot_bank_digest": self.shot_bank.digest,
            "physics_dt": self.backend.timestep,
            "control_dt": self._control_decimation * self.backend.timestep,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = {} if options is None else options
        if not isinstance(options, Mapping):
            raise TypeError("reset options must be a mapping or None")
        unknown = set(options) - {"shot_id"}
        if unknown:
            raise ValueError(f"unknown reset options: {sorted(unknown)}")
        if "shot_id" in options:
            shot_id = options["shot_id"]
            if not isinstance(shot_id, str):
                raise ValueError("options['shot_id'] must be a string")
            shot = self.shot_bank.get(shot_id)
        else:
            shot = self.shot_bank[int(self.np_random.integers(len(self.shot_bank)))]
        self.backend.reset()
        self.backend.launch_ball(shot)
        self._judge = TableTennisReturnJudge(
            timeout_s=self.config.timeout_s, table_spec=self.config.table
        )
        self._judge.reset(shot)
        self._shot = shot
        self._elapsed_steps = 0
        return self._observation(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._judge is None or self._shot is None:
            raise RuntimeError("reset must be called before step")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite float array with shape (7,)")
        if not self.action_space.contains(action):
            raise ValueError("action is outside the declared action space")
        quaternion = tuple(float(value) for value in action[3:])
        norm = sqrt(sum(value * value for value in quaternion))
        if norm <= 1e-12:
            raise ValueError("action quaternion must have non-zero norm")
        command = PaddleCommand(
            position=tuple(float(value) for value in action[:3]),
            quaternion=tuple(value / norm for value in quaternion),
        )
        previously_hit = self._judge.result.hit
        numerical_failure = False
        for _ in range(self._control_decimation):
            self.backend.apply_action(command)
            try:
                self.backend.step()
                self._judge.update(
                    time_s=self.backend.time,
                    ball=self.backend.get_ball_state(),
                    contacts=self.backend.semantic_contacts(),
                )
            except (FloatingPointError, ValueError):
                self._judge.abort(failure_reason="numerical", time_s=self.backend.time)
                numerical_failure = True
            if self._judge.done:
                break
        self._elapsed_steps += 1
        result = self._judge.result
        terminated = self._judge.done and result.failure_reason != "timeout"
        truncated = self._judge.done and result.failure_reason == "timeout"
        reward = self.config.reward_for(result, previously_hit=previously_hit)
        info = self._info()
        if terminated or truncated:
            info["episode_result"] = self._result_info(result)
            info["numerical_failure"] = numerical_failure
            self._episode_records.append(info["episode_result"])
        return self._observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self._judge = None
        self._shot = None


def register_envs() -> None:
    """Register supported environments, allowing repeated imports safely."""
    environment_id = TABLE_TENNIS_RETURN_V0.env_id
    if environment_id not in gym.registry:
        gym.register(environment_id, entry_point="multisport_sim.benchmark.envs:TableTennisReturnEnv")
