"""Gymnasium environments backed by the versioned Shot Skill core."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, sqrt
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .backends.mujoco import MujocoTableTennisBackend
from .controllers import PaddleCommand
from .rules.table_tennis import TableTennisReturnJudge
from .shot_bank import ShotBank
from .types import EpisodeResult, ShotSpec


class TableTennisReturnEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-shot MuJoCo table-tennis return task.

    The action is a world-frame mocap paddle pose ``[x, y, z, qw, qx, qy, qz]``.
    Position is measured in metres and the quaternion is normalized before it is
    sent to MuJoCo.  The observation is ``[ball position, linear velocity,
    angular velocity, paddle position, paddle quaternion]`` in the same world
    frame.  This fixture is deliberately marked experimental: it is an API and
    rule-engine integration point, not a robot embodiment.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        *,
        split: str = "dev",
        control_hz: float = 200.0,
        timeout_s: float = 2.0,
    ) -> None:
        super().__init__()
        if not isfinite(control_hz) or control_hz <= 0.0:
            raise ValueError("control_hz must be a positive finite number")
        if not isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be a positive finite number")
        self.shot_bank = ShotBank.from_resource(split=split)
        self.backend = MujocoTableTennisBackend()
        self.control_hz = float(control_hz)
        self.timeout_s = float(timeout_s)
        self._control_decimation = max(
            1, round(1.0 / (self.control_hz * self.backend.timestep))
        )
        self.action_space = spaces.Box(
            low=np.array([-2.0, -1.5, 0.5, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([0.0, 1.5, 1.8, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array(
                [-5.0, -5.0, -1.0, -50.0, -50.0, -50.0, -2000.0, -2000.0, -2000.0,
                 -2.0, -1.5, 0.5, -1.0, -1.0, -1.0, -1.0],
                dtype=np.float32,
            ),
            high=np.array(
                [5.0, 5.0, 5.0, 50.0, 50.0, 50.0, 2000.0, 2000.0, 2000.0,
                 0.0, 1.5, 1.8, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
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
        self._judge = TableTennisReturnJudge(timeout_s=self.timeout_s)
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
        reward = float(result.hit and not previously_hit)
        if result.valid_return:
            reward += 3.0
        if result.target_hit:
            reward += 1.0
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
    environment_id = "MultiSportRobot/TableTennisReturn-v0"
    if environment_id not in gym.registry:
        gym.register(environment_id, entry_point="multisport_sim.benchmark.envs:TableTennisReturnEnv")
