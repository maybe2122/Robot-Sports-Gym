"""Learned baselines for the launch tasks: a policy over one swing primitive.

The scripted launch fixtures are good because they *know* things: the serve's
measured carry table, the boot's impulse gain, the drag model.  A learned
baseline has to find those out by trying.  This module gives it the smallest
honest problem that still requires it:

* a **primitive** that swings the face along a straight line, along a normal
  set by an elevation and a yaw offset from the aim point, at a face speed --
  and knows nothing about what that does to the ball;
* a **policy** that sees only observable geometry -- where the contact will
  be relative to the aim point, how the ball is moving, what the target is --
  and outputs the three primitive parameters;
* a one-decision **environment** (:class:`PrimitiveLaunchEnv`) that runs the
  real MuJoCo episode, scored by the real judge, for each decision.

What the primitive does know is observation geometry: the ball's observed
state extrapolated to the contact time, and the aim point (the L3 target when
there is one -- task information, see ``envs.target_info``).  Nothing in it is
calibrated against the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, cos, radians, sin, sqrt
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .controllers import ControllerObservation, PaddleCommand, _quaternion_taking_y_to
from .envs import target_info
from .runner import RunConfig, run_shots
from .shot_bank import ShotBank
from .types import EpisodeResult, ShotSpec

GRAVITY = 9.81


@dataclass(frozen=True)
class PrimitiveSpec:
    """Parameter ranges and geometry of one sport's swing primitive."""

    sport: str
    task_id: str
    bank: str
    speed_mps: tuple[float, float]
    elevation_degrees: tuple[float, float]
    yaw_offset_degrees: tuple[float, float]
    standoff_m: float
    default_aim: tuple[float, float]
    # Airborne objects are struck after falling to this height (badminton) or
    # after this lead (basketball, football).
    strike_height_m: float | None = None
    lead_s: float = 0.15
    backswing_m: float = 0.4
    follow_through_m: float = 0.10
    aim_is_goal_mouth: bool = False
    # Typical magnitude of each feature, so the policy sees numbers near one.
    feature_scale: tuple[float, ...] = (10.0, 1.0, 3.0, 3.0, 3.0, 1.0, 1.0, 0.5)


SPECS: dict[str, PrimitiveSpec] = {
    "badminton": PrimitiveSpec(
        sport="badminton",
        task_id="badminton-serve-v0",
        bank="badminton/serve-v0",
        speed_mps=(6.0, 26.0),
        elevation_degrees=(25.0, 65.0),
        yaw_offset_degrees=(-10.0, 10.0),
        standoff_m=0.025 + 0.0135,
        # The middle of the receiving court; the side is mirrored at use.
        default_aim=(4.3, 1.3),
        strike_height_m=0.85,
        backswing_m=0.5,
        feature_scale=(7.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5),
    ),
    "football": PrimitiveSpec(
        sport="football",
        task_id="football-kick-v0",
        bank="football/kick-v0",
        speed_mps=(8.0, 24.0),
        elevation_degrees=(0.0, 30.0),
        yaw_offset_degrees=(-6.0, 6.0),
        standoff_m=0.03 + 0.11,
        # Goal mouth (y, z): the middle of the goal at 1 m.
        default_aim=(0.0, 1.0),
        lead_s=0.30,
        aim_is_goal_mouth=True,
        feature_scale=(20.0, 0.1, 3.0, 3.0, 1.0, 1.5, 1.0, 0.5),
    ),
    "basketball": PrimitiveSpec(
        sport="basketball",
        task_id="basketball-shoot-v0",
        bank="basketball/shoot-v0",
        speed_mps=(3.0, 12.0),
        elevation_degrees=(40.0, 75.0),
        yaw_offset_degrees=(-5.0, 5.0),
        standoff_m=0.03 + 0.12,
        # Rim plane (x, y): the rim's centre.
        default_aim=(0.0, 0.0),
        lead_s=0.15,
        feature_scale=(5.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.06),
    ),
}

FEATURES = (
    "aim_distance_m",
    "contact_height_m",
    "ball_vx_mps",
    "ball_vy_mps",
    "ball_vz_mps",
    "aim_height_m",
    "target_present",
    "target_radius_m",
)


def _scale(value: float, bounds: tuple[float, float]) -> float:
    """Map an action component in [-1, 1] onto ``bounds``."""
    low, high = bounds
    return low + (float(np.clip(value, -1.0, 1.0)) + 1.0) * 0.5 * (high - low)


class SwingPrimitive:
    """Execute one straight-line swing chosen by :meth:`choose` at the first step.

    ``plan_features`` and ``choose`` are split so that a policy can look at the
    features and hand back the three parameters; :class:`LearnedLaunchController`
    wires a trained policy into that loop.
    """

    def __init__(self, spec: PrimitiveSpec, *, control_dt: float = 0.005) -> None:
        self.spec = spec
        self.control_dt = float(control_dt)
        self._target: dict[str, Any] | None = None
        self._shot_y = 0.0
        self._idle = False
        self._contact: tuple[tuple[float, float, float], tuple[float, float, float], float] | None = None
        self._plan: tuple[Any, ...] | None = None

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        del seed
        self._target = target_info(shot)
        self._shot_y = shot.position[1]
        self._idle = shot.level == "L0"
        self._contact = None
        self._plan = None

    # -- geometry the policy may see ----------------------------------------

    def _predict_contact(self, observation: ControllerObservation):
        x, y, z = observation.ball.position
        vx, vy, vz = observation.ball.linear_velocity
        spec = self.spec
        if spec.strike_height_m is not None:
            drop = max(z - 0.012 - spec.strike_height_m, 0.0)
            dt = (vz + sqrt(max(vz * vz + 2.0 * GRAVITY * drop, 0.0))) / GRAVITY
            contact = (x + vx * dt, y + vy * dt, spec.strike_height_m)
            velocity = (vx, vy, vz - GRAVITY * dt)
        elif spec.sport == "football":
            dt = spec.lead_s
            contact = (x + vx * dt, y + vy * dt, z)
            velocity = (vx, vy, 0.0)
        else:
            dt = spec.lead_s
            contact = (x + vx * dt, y + vy * dt, z + vz * dt - 0.5 * GRAVITY * dt * dt)
            velocity = (vx, vy, vz - GRAVITY * dt)
        return contact, velocity, observation.time_s + dt

    def _aim(self) -> tuple[float, float, float]:
        """Horizontal aim point (x, y) and, for the goal mouth, the aim height."""
        spec = self.spec
        if self._target is not None:
            u, v = self._target["center"]
        else:
            u, v = spec.default_aim
            if spec.sport == "badminton":
                v = (-1.0 if self._shot_y > 0.0 else 1.0) * abs(v)
        if spec.aim_is_goal_mouth:
            return 0.0, u, v
        return u, v, 0.0

    def features(self, observation: ControllerObservation) -> np.ndarray:
        contact, velocity, _ = self._predict_contact(observation)
        aim_x, aim_y, aim_z = self._aim()
        distance = sqrt((aim_x - contact[0]) ** 2 + (aim_y - contact[1]) ** 2)
        target = self._target
        raw = np.array(
            (
                distance,
                contact[2],
                *velocity,
                aim_z,
                0.0 if target is None else 1.0,
                0.0 if target is None else target["radius_m"],
            ),
            dtype=np.float32,
        )
        return raw / np.asarray(self.spec.feature_scale, dtype=np.float32)

    # -- the primitive --------------------------------------------------------

    def choose(self, observation: ControllerObservation, action: np.ndarray) -> None:
        spec = self.spec
        contact, _, contact_time = self._predict_contact(observation)
        aim_x, aim_y, _ = self._aim()
        speed = _scale(action[0], spec.speed_mps)
        elevation = radians(_scale(action[1], spec.elevation_degrees))
        yaw = atan2(aim_y - contact[1], aim_x - contact[0]) + radians(
            _scale(action[2], spec.yaw_offset_degrees)
        )
        normal = (cos(elevation) * cos(yaw), cos(elevation) * sin(yaw), sin(elevation))
        face = tuple(contact[axis] - normal[axis] * spec.standoff_m for axis in range(3))
        self._plan = (face, normal, contact_time, speed, _quaternion_taking_y_to(normal))

    def act(self, observation: ControllerObservation) -> PaddleCommand | None:
        if self._idle or self._plan is None:
            return None
        face, normal, contact_time, speed, quaternion = self._plan
        when = observation.time_s + self.control_dt
        travel = min(
            max(speed * (when - contact_time), -self.spec.backswing_m),
            self.spec.follow_through_m,
        )
        return PaddleCommand(
            position=tuple(face[axis] + normal[axis] * travel for axis in range(3)),
            quaternion=quaternion,
        )


def load_learned_controller(sport: str, path: str) -> LearnedLaunchController:
    """Load a Stable-Baselines3 PPO policy saved by ``scripts/train_launch_policies.py``."""
    from stable_baselines3 import PPO

    model = PPO.load(path, device="cpu")
    return LearnedLaunchController(sport, model, policy_id=f"learned-primitive-ppo:{path}")


@dataclass(frozen=True)
class ReturnPrimitiveSpec:
    """Parameter ranges of the return fixture's swing, for one net sport.

    The return primitive is the reference fixture's own swing -- track the
    ball laterally, drive the blade forward once the ball passes a trigger --
    with the three numbers the reference hard-codes left to the policy:
    blade speed, face tilt and where the swing starts.
    """

    sport: str
    task_id: str
    bank: str
    speed_mps: tuple[float, float]
    tilt_degrees: tuple[float, float]
    trigger_x_m: tuple[float, float]
    feature_scale: tuple[float, ...]


RETURN_SPECS: dict[str, ReturnPrimitiveSpec] = {
    "table_tennis": ReturnPrimitiveSpec(
        sport="table_tennis",
        task_id="table-tennis-return-v0",
        bank="table_tennis/return-v1",
        speed_mps=(1.0, 6.0),
        tilt_degrees=(5.0, 45.0),
        trigger_x_m=(-1.4, -0.4),
        feature_scale=(1.5, 0.5, 0.5, 5.0, 1.0, 3.0, 50.0, 100.0, 50.0, 1.0, 1.5, 0.5, 0.2),
    ),
    "tennis": ReturnPrimitiveSpec(
        sport="tennis",
        task_id="tennis-return-v0",
        bank="tennis/return-v0",
        speed_mps=(6.0, 24.0),
        tilt_degrees=(0.0, 30.0),
        trigger_x_m=(-9.0, -2.0),
        feature_scale=(12.0, 3.0, 1.0, 20.0, 5.0, 5.0, 200.0, 300.0, 200.0, 1.0, 10.0, 4.0, 1.0),
    ),
}

RETURN_FEATURES = (
    "ball_x_m",
    "ball_y_m",
    "ball_z_m",
    "ball_vx_mps",
    "ball_vy_mps",
    "ball_vz_mps",
    "ball_wx_radps",
    "ball_wy_radps",
    "ball_wz_radps",
    "target_present",
    "target_u_m",
    "target_v_m",
    "target_radius_m",
)


class ReturnPrimitive:
    """The reference return swing with three parameters chosen by a policy."""

    def __init__(self, spec: ReturnPrimitiveSpec) -> None:
        self.spec = spec
        self._target: dict[str, Any] | None = None
        self._controller: Any = None
        self._idle = False

    @property
    def _plan(self) -> Any:
        # The launch controller's contract: planned once the swing is chosen.
        return self._controller

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        del seed
        self._target = target_info(shot)
        self._controller = None
        self._idle = shot.level == "L0"

    def features(self, observation: ControllerObservation) -> np.ndarray:
        target = self._target
        raw = np.array(
            (
                *observation.ball.position,
                *observation.ball.linear_velocity,
                *observation.ball.angular_velocity,
                0.0 if target is None else 1.0,
                *((0.0, 0.0) if target is None else target["center"]),
                0.0 if target is None else target["radius_m"],
            ),
            dtype=np.float32,
        )
        return raw / np.asarray(self.spec.feature_scale, dtype=np.float32)

    def choose(self, observation: ControllerObservation, action: np.ndarray) -> None:
        from .controllers import SCRIPTED_DEFAULTS, ScriptedPaddleController

        defaults = dict(SCRIPTED_DEFAULTS[self.spec.sport])
        defaults["swing_speed_mps"] = _scale(action[0], self.spec.speed_mps)
        defaults["upward_tilt_degrees"] = _scale(action[1], self.spec.tilt_degrees)
        defaults["swing_trigger_x"] = _scale(action[2], self.spec.trigger_x_m)
        self._controller = ScriptedPaddleController(**defaults)

    def act(self, observation: ControllerObservation) -> PaddleCommand | None:
        if self._idle or self._controller is None:
            return None
        return self._controller.act(observation)


def primitive_for(sport: str) -> SwingPrimitive | ReturnPrimitive:
    """The primitive a sport's learned baseline acts through."""
    if sport in RETURN_SPECS:
        return ReturnPrimitive(RETURN_SPECS[sport])
    return SwingPrimitive(SPECS[sport])


class LearnedLaunchController:
    """A trained policy driving :class:`SwingPrimitive`; the benchmark controller."""

    def __init__(self, sport: str, policy: Any, *, policy_id: str) -> None:
        self.primitive = primitive_for(sport)
        self.policy = policy
        self.controller_id = policy_id

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        self.primitive.reset(shot, seed=seed)

    def act(self, observation: ControllerObservation) -> PaddleCommand | None:
        if self.primitive._plan is None and not self.primitive._idle:
            features = self.primitive.features(observation)
            action, _ = self.policy.predict(features, deterministic=True)
            self.primitive.choose(observation, np.asarray(action, dtype=np.float32))
        return self.primitive.act(observation)


class _FixedAction:
    """Feed a preset action to :class:`SwingPrimitive` inside the one-step env."""

    def __init__(self, primitive: SwingPrimitive, action: np.ndarray) -> None:
        self.primitive = primitive
        self.action = action
        self.controller_id = "primitive-rollout"

    def reset(self, shot: ShotSpec, *, seed: int | None = None) -> None:
        self.primitive.reset(shot, seed=seed)

    def act(self, observation: ControllerObservation) -> PaddleCommand | None:
        if self.primitive._plan is None and not self.primitive._idle:
            self.primitive.choose(observation, self.action)
        return self.primitive.act(observation)


def launch_reward(result: EpisodeResult, *, target_radius_m: float | None = None) -> float:
    """Success first, placement second, contact as a small shaping term.

    With a target, a successful launch also earns up to 0.5 for landing close
    to it, so a near miss is better than a far one -- the binary ``target_hit``
    alone gives the policy nothing to climb.
    """
    reward = 0.1 * float(result.hit) + float(result.valid_return) + float(result.target_hit)
    if target_radius_m and result.valid_return and result.target_error_m is not None:
        reward += 0.5 * float(np.exp(-((result.target_error_m / target_radius_m) ** 2)))
    return reward


class PrimitiveLaunchEnv(gym.Env[np.ndarray, np.ndarray]):
    """One decision per episode: pick the swing, watch the real rollout, get scored.

    Observations are :data:`FEATURES`; actions are three numbers in [-1, 1]
    for face speed, elevation and yaw offset.  Shots are drawn from the train
    split of the task's bank, from the levels given.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        sport: str,
        *,
        split: str = "train",
        levels: tuple[str, ...] = ("L1", "L2", "L3", "L4"),
    ) -> None:
        from . import tasks  # noqa: F401  (publishes the built-in tasks)
        from .backends.mujoco import MujocoShotBackend
        from .registry import get_task

        super().__init__()
        self.spec_ = RETURN_SPECS[sport] if sport in RETURN_SPECS else SPECS[sport]
        entry = get_task(self.spec_.task_id)
        bank = ShotBank.from_resource(split=split, task=self.spec_.bank)
        # The bank's own timeout: the table-tennis fixture learns on return-v1.
        self.task = replace(
            entry.config,
            split=split,
            bank_resource=self.spec_.bank,
            timeout_s=float(bank.manifest["episode"]["timeout_s"]),
        )
        self.shots = [shot for shot in bank if shot.level in levels]
        self.backend = MujocoShotBackend(sport=sport)
        self.primitive = primitive_for(sport)
        width = len(RETURN_FEATURES) if sport in RETURN_SPECS else len(FEATURES)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(width,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self._shot: ShotSpec | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._shot = self.shots[int(self.np_random.integers(len(self.shots)))]
        self.backend.reset()
        self.backend.launch_ball(self._shot)
        self.primitive.reset(self._shot)
        return self.primitive.features(self.backend.observe()), {"shot_id": self._shot.shot_id}

    def step(self, action: np.ndarray):
        assert self._shot is not None
        output = run_shots(
            self.backend,
            _FixedAction(self.primitive, np.asarray(action, dtype=np.float32)),
            [self._shot],
            config=RunConfig.from_task_config(self.task),
        )
        result = output.results[0]
        features = np.zeros(self.observation_space.shape, dtype=np.float32)
        radius = None if self._shot.target is None else self._shot.target.radius_m
        reward = launch_reward(result, target_radius_m=radius)
        return features, reward, True, False, {"episode_result": result.to_dict()}


__all__ = [
    "FEATURES",
    "RETURN_FEATURES",
    "RETURN_SPECS",
    "SPECS",
    "LearnedLaunchController",
    "PrimitiveLaunchEnv",
    "PrimitiveSpec",
    "ReturnPrimitive",
    "ReturnPrimitiveSpec",
    "SwingPrimitive",
    "launch_reward",
    "load_learned_controller",
    "primitive_for",
]
