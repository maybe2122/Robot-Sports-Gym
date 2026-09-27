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
from .backends.mujoco_robot import (
    MujocoG1TableTennisBackend,
    MujocoPandaTableTennisBackend,
)
from .backends.mujoco_standing import MujocoStandingG1TableTennisBackend
from .controllers import PaddleCommand
from .robot import ControlMode, JointCommand, SafetyViolation, WorkspaceBox
from .rules.badminton import BadmintonServeJudge
from .rules.basketball import BasketballShootJudge
from .rules.football import FootballKickJudge
from .rules.net_return import NetReturnJudge
from .rules.table_tennis import TableTennisReturnJudge
from .rules.tennis import TennisReturnJudge
from .shot_bank import ShotBank
from .task_config import (
    BADMINTON_SERVE_V0,
    BASKETBALL_SHOOT_V0,
    FOOTBALL_KICK_V0,
    TABLE_TENNIS_RETURN_G1_V1,
    TABLE_TENNIS_RETURN_PANDA_V1,
    TABLE_TENNIS_RETURN_STANDING_G1_V2,
    TABLE_TENNIS_RETURN_V0,
    TENNIS_RETURN_V0,
    BadmintonServeTaskConfig,
    BasketballShootTaskConfig,
    EmbodiedTableTennisReturnTaskConfig,
    FootballKickTaskConfig,
    TableTennisReturnG1TaskConfig,
    TableTennisReturnStandingG1TaskConfig,
    TableTennisReturnTaskConfig,
    TennisReturnTaskConfig,
)
from .types import EpisodeResult, ShotSpec
from .vision import (
    VisionTrackBackend,
    vision_observation_dict,
    vision_observation_space_shapes,
    vision_sensors_for,
)


def robot_observation_vector(
    state: Any, config: Any = TABLE_TENNIS_RETURN_PANDA_V1
) -> np.ndarray:
    """Pack one embodied backend observation, in the task's published layout.

    Evaluation and training must read the same numbers in the same order, so
    both the Gymnasium environment and the offline scoring path come through
    here instead of each laying out the vector for itself.  The order is not
    written here either: it comes from
    :meth:`~...task_config.TableTennisReturnPandaTaskConfig.observation_layout`,
    which is what makes the same function correct for a robot with a different
    number of joints.
    """
    return config.observation_layout().build(state)


def panda_observation_vector(state: Any) -> np.ndarray:
    """The Panda's 33-vector.  Retained under its released name."""
    return robot_observation_vector(state, TABLE_TENNIS_RETURN_PANDA_V1)


def target_info(shot: ShotSpec) -> dict[str, Any] | None:
    """The shot's placement target as task information, or ``None``.

    An L3 target is part of the *task*, like the court: a policy cannot be asked
    to land a ball in a circle it is never told about.  It was, until
    2026-09-28 -- no observation, info field or reset argument carried it, so
    only the privileged reference fixtures could score L3.  The coordinates are
    in the task's placement plane (``manifest.targets``): court (x, y) for the
    net sports and the serve, goal mouth (y, z) for the kick, rim plane (x, y)
    for the shot.
    """
    if shot.target is None:
        return None
    return {"center": list(shot.target.center_xy), "radius_m": shot.target.radius_m}


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
    # What a subclass changes to become another sport.  Everything else --
    # spaces, decimation, reward, episode bookkeeping -- is identical, because
    # the task is identical; only the geometry and the implement differ.
    config_type: ClassVar[type] = TableTennisReturnTaskConfig
    default_config: ClassVar[Any] = TABLE_TENNIS_RETURN_V0
    judge_type: ClassVar[type] = TableTennisReturnJudge

    def __init__(
        self,
        *,
        split: str | None = None,
        control_hz: float | None = None,
        timeout_s: float | None = None,
        config: Any = None,
    ) -> None:
        super().__init__()
        config = self.default_config if config is None else config
        if not isinstance(config, self.config_type):
            raise TypeError(f"config must be a {self.config_type.__name__}")
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
        self.shot_bank = ShotBank.from_resource(
            split=self.config.split, task=self.config.bank_resource
        )
        self.backend = MujocoTableTennisBackend(sport=self.config.sport)
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
        self._judge: NetReturnJudge | None = None
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
            "target": target_info(self._shot),
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
        self._judge = self._make_judge()
        self._judge.reset(shot)
        self._shot = shot
        self._elapsed_steps = 0
        return self._observation(), self._info()

    def _make_judge(self) -> NetReturnJudge:
        """The sport's judge, built from the sport's own surface geometry."""
        return self.judge_type(timeout_s=self.config.timeout_s, **self.judge_surface_kwarg())

    def judge_surface_kwarg(self) -> dict[str, Any]:
        return {"table_spec": self.config.table}

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


class TennisReturnEnv(TableTennisReturnEnv):
    """Single-shot MuJoCo tennis return, on the regulation singles court.

    Same action, same observation layout, same judge engine, same reward terms
    and the same L0-L5 thresholds as table tennis.  What differs is what the
    numbers describe: a 23.77 m court instead of a 2.74 m table, a ball arriving
    at 20-30 m/s instead of 4-8, and a three-second episode because the ball has
    twenty metres to travel.
    """

    config_type: ClassVar[type] = TennisReturnTaskConfig
    default_config: ClassVar[Any] = TENNIS_RETURN_V0
    judge_type: ClassVar[type] = TennisReturnJudge

    def judge_surface_kwarg(self) -> dict[str, Any]:
        return {"court_spec": self.config.court}


class BadmintonServeEnv(TableTennisReturnEnv):
    """Single-shot MuJoCo badminton serve, into the BWF singles service court.

    The action and observation layouts are the return tasks'.  The shuttle is
    released on the robot's side at reset and falls until it is struck; the
    episode ends when it first lands, is faulted, or times out.  The physics
    step is 0.5 ms (see ``scene.BENCHMARK_TIMESTEPS``), so one 200 Hz control
    step holds ten physics steps, over which the face pose is interpolated.
    """

    config_type: ClassVar[type] = BadmintonServeTaskConfig
    default_config: ClassVar[Any] = BADMINTON_SERVE_V0
    judge_type: ClassVar[type] = BadmintonServeJudge

    def judge_surface_kwarg(self) -> dict[str, Any]:
        return {"court_spec": self.config.court}


class FootballKickEnv(TableTennisReturnEnv):
    """Single-shot MuJoCo football kick at the IFAB goal.

    The task frame's origin is the centre of the goal line; the ball rests or
    rolls on the pitch at x < 0 and the episode ends when the whole ball crosses
    the line, leaves the pitch, or times out.
    """

    config_type: ClassVar[type] = FootballKickTaskConfig
    default_config: ClassVar[Any] = FOOTBALL_KICK_V0
    judge_type: ClassVar[type] = FootballKickJudge

    def judge_surface_kwarg(self) -> dict[str, Any]:
        return {"goal_spec": self.config.goal}


class BasketballShootEnv(TableTennisReturnEnv):
    """Single-shot MuJoCo basketball shot at the FIBA basket.

    The task frame's origin is the floor below the rim's centre; the ball is
    released at x < 0 and the episode ends when it comes down through the ring,
    reaches the floor, or times out.
    """

    config_type: ClassVar[type] = BasketballShootTaskConfig
    default_config: ClassVar[Any] = BASKETBALL_SHOOT_V0
    judge_type: ClassVar[type] = BasketballShootJudge

    def judge_surface_kwarg(self) -> dict[str, Any]:
        return {"basket_spec": self.config.basket}


class TableTennisReturnPandaEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-shot MuJoCo table-tennis return performed by a Franka Panda.

    The action is a seven-dimensional joint-position setpoint in radians, held
    for one control step and clamped to the arm's own limits.  The observation
    concatenates the ball state, the joint state, and the blade's task-frame
    pose and linear velocity -- 33 numbers, none of which name a geom or a
    simulator handle.

    Unlike :class:`TableTennisReturnEnv`, this environment has a safety
    envelope.  A joint, velocity, torque, collision or workspace breach ends the
    episode with ``terminated=True`` and ``failure_reason="safety"``, and the
    violation records appear in ``info`` so a submission can see what it broke.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}
    # What a subclass changes to become another embodiment.  Everything else --
    # spaces, decimation, safety handling, episode bookkeeping -- is identical,
    # because the task is identical; only the robot differs.  The spaces are
    # built from the config's published layout, so a robot with a different
    # joint count needs no code here at all.
    config_type: ClassVar[type] = EmbodiedTableTennisReturnTaskConfig
    default_config: ClassVar[Any] = TABLE_TENNIS_RETURN_PANDA_V1
    backend_type: ClassVar[Any] = MujocoPandaTableTennisBackend

    def __init__(
        self,
        *,
        split: str | None = None,
        control_hz: float | None = None,
        timeout_s: float | None = None,
        config: Any = None,
    ) -> None:
        super().__init__()
        config = self.default_config if config is None else config
        if not isinstance(config, self.config_type):
            raise TypeError(f"config must be a {self.config_type.__name__}")
        overrides = {
            name: value
            for name, value in (
                ("split", split),
                ("control_hz", control_hz),
                ("timeout_s", timeout_s),
            )
            if value is not None
        }
        self.config = replace(config, **overrides) if overrides else config
        self.shot_bank = ShotBank.from_resource(
            split=self.config.split, task=self.config.bank_resource
        )
        self.backend = self.backend_type(
            workspace=WorkspaceBox(
                self.config.workspace.position_low, self.config.workspace.position_high
            )
        )
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
        return tuple(self._episode_records)

    def _observation(self) -> np.ndarray:
        return robot_observation_vector(self.backend.observe(), self.config)

    def _info(self) -> dict[str, Any]:
        assert self._shot is not None
        return {
            "task_id": self.config.task_id,
            "robot_id": self.config.robot_id,
            "shot_id": self._shot.shot_id,
            "shot_bank_digest": self.shot_bank.digest,
            "physics_dt": self.backend.timestep,
            "control_dt": self._control_decimation * self.backend.timestep,
            "safety_violations": self.backend.safety_violation_count,
            "target": target_info(self._shot),
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
        if action.shape != (self.config.ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError(
                f"action must be a finite float array with shape "
                f"({self.config.ACTION_DIM},)"
            )
        if not self.action_space.contains(action):
            raise ValueError("action is outside the declared action space")
        command = JointCommand(
            tuple(float(value) for value in action), ControlMode.JOINT_POSITION
        )

        previously_hit = self._judge.result.hit
        numerical_failure = False
        violations: tuple[SafetyViolation, ...] = ()
        for _ in range(self._control_decimation):
            self.backend.apply_action(command)
            try:
                self.backend.step()
                # Safety is checked every physics step, not once per control
                # step: a limit broken mid-decimation is still broken.
                violations = self.backend.safety_violations()
                if violations:
                    self._judge.abort(failure_reason="safety", time_s=self.backend.time)
                    break
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
        if violations:
            info["safety_violation_details"] = [item.to_dict() for item in violations]
        if terminated or truncated:
            info["episode_result"] = result.to_dict()
            info["numerical_failure"] = numerical_failure
            self._episode_records.append(info["episode_result"])
        return self._observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self._judge = None
        self._shot = None


class TableTennisReturnPandaVisionEnv(TableTennisReturnPandaEnv):
    """The embodied return task with cameras in place of privileged state.

    Same arm, same action, same judge, same shot bank and the same reward as
    :class:`TableTennisReturnPandaEnv` -- so a score here is comparable with a
    state-track score, and the difference between them is perception.

    The observation is a dictionary rather than a flat vector: the two camera
    frames are ``uint8`` images, and flattening them would decide the policy's
    architecture on its behalf.  ``frame_age_s`` says how stale the newest
    frame is, because the cameras run at 120 Hz under a 200 Hz control loop.
    """

    sensor_robot: ClassVar[str] = "panda"

    def __init__(self, *, perception: str = "stereo", **kwargs: Any) -> None:
        if perception not in {"stereo", "rgbd", "depth"}:
            raise ValueError("perception must be stereo, rgbd, or depth")
        super().__init__(**kwargs)
        self.perception = perception
        self.sensor_specs = vision_sensors_for(self.sensor_robot, depth=perception != "stereo")
        # Rebuild the backend with the declared suite: the cameras have to
        # exist in the compiled model, so they cannot be added afterwards.
        self.backend = self.backend_type(
            workspace=WorkspaceBox(
                self.config.workspace.position_low, self.config.workspace.position_high
            ),
            sensors=self.sensor_specs,
        )
        self._vision = VisionTrackBackend(self.backend, depth_only=perception == "depth")
        image_spaces = {
            name: spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
            for name, shape in vision_observation_space_shapes(self.sensor_specs).items()
        }
        if perception == "depth":
            image_spaces.clear()
        from .sensors import CameraSpec
        for spec in self.sensor_specs:
            if isinstance(spec, CameraSpec) and spec.depth:
                image_spaces[f"{spec.name}_depth"] = spaces.Box(
                    low=0., high=np.inf, shape=(spec.height, spec.width), dtype=np.float32
                )
        if any(s.name == "balance_imu" for s in self.sensor_specs):
            image_spaces["base_imu"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
            )
        joint_low, joint_high = self.config.action_bounds()
        self.observation_space = spaces.Dict(
            {
                **image_spaces,
                "joint_positions": spaces.Box(
                    low=np.array(joint_low, dtype=np.float32),
                    high=np.array(joint_high, dtype=np.float32),
                    dtype=np.float32,
                ),
                "joint_velocities": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.config.ACTION_DIM,), dtype=np.float32
                ),
                "blade_pose": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
                "frame_age_s": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )

    def _observation(self) -> dict[str, np.ndarray]:  # type: ignore[override]
        return vision_observation_dict(self._vision.observe(), sensors=self.sensor_specs,
                                       include_rgb=self.perception != "depth")

    def _info(self) -> dict[str, Any]:
        return {**super()._info(), "track": "vision", "privileged_ball_state": False}

    def close(self) -> None:
        if self.backend.sensors is not None:
            self.backend.sensors.close()
        super().close()


class TableTennisReturnG1Env(TableTennisReturnPandaEnv):
    """The same return task, performed by a fixed-base Unitree G1.

    Ten joint setpoints instead of seven, so a 39-number observation instead of
    33.  Neither width appears in this class: both spaces are built from the
    config's published observation layout, which is the whole reason this
    environment is nine lines rather than three hundred.

    Same judge, same frozen shot bank, same thresholds, same report schema.  A
    G1 result and a Panda result belong in one table with a robot column.
    """

    config_type: ClassVar[type] = TableTennisReturnG1TaskConfig
    default_config: ClassVar[Any] = TABLE_TENNIS_RETURN_G1_V1
    backend_type: ClassVar[Any] = MujocoG1TableTennisBackend


class TableTennisReturnG1VisionEnv(TableTennisReturnPandaVisionEnv):
    """G1 joint control from stereo RGB or aligned RGB-D and proprioception."""

    config_type: ClassVar[type] = TableTennisReturnG1TaskConfig
    default_config: ClassVar[Any] = TABLE_TENNIS_RETURN_G1_V1
    backend_type: ClassVar[Any] = MujocoG1TableTennisBackend
    sensor_robot: ClassVar[str] = "g1"


class TableTennisReturnStandingG1Env(TableTennisReturnG1Env):
    config_type: ClassVar[type] = TableTennisReturnStandingG1TaskConfig
    default_config: ClassVar[Any] = TABLE_TENNIS_RETURN_STANDING_G1_V2
    backend_type: ClassVar[Any] = MujocoStandingG1TableTennisBackend


class TableTennisReturnStandingG1VisionEnv(TableTennisReturnG1VisionEnv):
    config_type: ClassVar[type] = TableTennisReturnStandingG1TaskConfig
    default_config: ClassVar[Any] = TABLE_TENNIS_RETURN_STANDING_G1_V2
    backend_type: ClassVar[Any] = MujocoStandingG1TableTennisBackend
    sensor_robot: ClassVar[str] = "g1-standing"


def register_envs() -> None:
    """Register every task in the registry, allowing repeated imports safely."""
    from . import tasks  # noqa: F401  (importing publishes the built-in tasks)
    from .registry import iter_tasks

    for entry in iter_tasks():
        if entry.config.env_id not in gym.registry:
            gym.register(entry.config.env_id, entry_point=entry.env_entry_point)
        vision_id = getattr(entry.config, "vision_env_id", None)
        if entry.vision_env_entry_point is not None and vision_id not in gym.registry:
            gym.register(vision_id, entry_point=entry.vision_env_entry_point)
