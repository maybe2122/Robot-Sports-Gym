"""Isaac Lab vectorized implementation of the table-tennis return task.

Import this module only after :class:`isaaclab.app.AppLauncher` has created the
SimulationApp, exactly like :mod:`multisport_sim.isaac_backend`.

The environment reuses the backend-neutral core rather than re-deriving it: the
shot bank, the semantic-contact vocabulary, :class:`TableTennisReturnJudge` and
the shared :class:`TableTennisReturnTaskConfig` are the same objects the MuJoCo
reference environment uses, so the two backends can only differ in physics, not
in task definition, rule interpretation or reward.

Two deliberate differences from the MuJoCo fixture are documented rather than
hidden:

* the scene contains only the bodies the rules can observe (table, net, floor,
  blade) -- decorative geometry is skipped because a vectorized training scene
  gains nothing from it;
* the blade is an axis-aligned box rather than MuJoCo's ellipsoid, so contact
  timing near the blade rim is not bit-comparable across backends.

Status: experimental.  Validated on CPU PhysX against the MuJoCo reference by
``scripts/backend_parity.py``: ball flight agrees to millimetres, ball-surface
contact does not yet (see ``reports/table-tennis-backend-parity.md``), so return
scores from the two backends are not directly comparable.  CPU and GPU PhysX
agree with each other to within a millimetre of flight.  The repository's CI
has no Isaac runtime.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from ...isaac_backend import primitive_spawn_cfg
from ...isaac_scene import IsaacPrimitive
from ...physics import Atmosphere
from ...specs import AIR_DENSITY, BALLS, Sport
from ...specs import TABLE_TENNIS as TABLE
from ..events import BALL, FLOOR, NET, ROBOT_RACKET
from ..events import TABLE as TABLE_CATEGORY
from ..rules.table_tennis import TableTennisReturnJudge
from ..shot_bank import ShotBank
from ..task_config import TABLE_TENNIS_RETURN_V0, TableTennisReturnTaskConfig
from ..types import BallState, SemanticContact, ShotSpec

BALL_SPEC = BALLS[Sport.TABLE_TENNIS]
PHYSICS_DT = 1.0 / 1000.0
"""Matches the MuJoCo model timestep so both backends resolve the same bounces."""

BLADE_SIZE = (0.17, 0.016, 0.20)
"""Box stand-in for the MuJoCo ellipsoid blade, using its bounding dimensions."""

EPISODE_RECORD_LIMIT = 100_000
"""Cap on retained episode records; see :attr:`IsaacTableTennisReturnEnv.episode_records`."""

# Contact-sensor filter order.  The judge consumes semantic names only, so this
# is the single place where an Isaac body becomes a benchmark category.
SEMANTIC_FILTERS: tuple[tuple[str, str], ...] = (
    (TABLE_CATEGORY, "{ENV_REGEX_NS}/Table"),
    (NET, "{ENV_REGEX_NS}/Net"),
    (FLOOR, "{ENV_REGEX_NS}/Floor"),
    (ROBOT_RACKET, "{ENV_REGEX_NS}/Blade"),
)


def _static_body(
    name: str,
    prim_path: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    friction: tuple[float, float] = (0.7, 0.6),
    restitution: float = 0.0,
) -> RigidObjectCfg:
    """Spawn a fixed collider as a kinematic body so contacts can be filtered.

    PhysX reports filtered contact pairs only between rigid bodies, so the table,
    net and floor are kinematic rigid bodies rather than static colliders.
    """
    primitive = IsaacPrimitive(
        name=name,
        kind="cuboid",
        position=(0.0, 0.0, 0.0),
        size=size,
        color=color,
        static_friction=friction[0],
        dynamic_friction=friction[1],
        restitution=restitution,
    )
    spawn = primitive_spawn_cfg(primitive)
    spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    spawn.mass_props = sim_utils.MassPropertiesCfg(mass=1.0)
    spawn.activate_contact_sensors = True
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


def _blade_body() -> RigidObjectCfg:
    primitive = IsaacPrimitive(
        name="blade",
        kind="cuboid",
        position=(0.0, 0.0, 0.0),
        size=BLADE_SIZE,
        color=(0.82, 0.05, 0.035),
        static_friction=0.85,
        dynamic_friction=0.70,
        restitution=0.80,
    )
    spawn = primitive_spawn_cfg(primitive)
    spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.17)
    spawn.activate_contact_sensors = True
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Blade",
        spawn=spawn,
        # Parked outside the playing width, as in the MuJoCo scene.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.58, 1.25, 1.0)),
    )


def _ball_body() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_SPEC.radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                linear_damping=0.0,
                angular_damping=max(0.001, BALL_SPEC.rolling_friction * 2.0),
                max_linear_velocity=120.0,
                max_angular_velocity=3000.0,
                max_depenetration_velocity=20.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_SPEC.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=min(0.004, BALL_SPEC.radius * 0.20),
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=BALL_SPEC.color[:3], roughness=0.72
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.58,
                dynamic_friction=0.48,
                restitution=BALL_SPEC.restitution,
                friction_combine_mode="average",
                restitution_combine_mode="max",
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, 0.0, 1.25)),
    )


@configclass
class TableTennisReturnSceneCfg(InteractiveSceneCfg):
    """Per-environment table, net, floor, blade and ball in the task frame.

    Every prim is authored relative to the environment origin, which *is* the
    task-frame origin, so the judge needs no per-backend offset.
    """

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2200.0, color=(0.78, 0.84, 0.95)),
    )
    floor = _static_body(
        "floor",
        "{ENV_REGEX_NS}/Floor",
        (0.0, 0.0, -0.04),
        (8.0, 5.0, 0.08),
        (0.34, 0.16, 0.08),
        friction=(0.9, 0.8),
    )
    table = _static_body(
        "table",
        "{ENV_REGEX_NS}/Table",
        (0.0, 0.0, TABLE.table_height - 0.02),
        (TABLE.table_length, TABLE.table_width, 0.04),
        (0.025, 0.27, 0.52),
        friction=(0.35, 0.25),
        restitution=BALL_SPEC.restitution,
    )
    net = _static_body(
        "net",
        "{ENV_REGEX_NS}/Net",
        (TABLE.net_plane_x, 0.0, TABLE.table_height + TABLE.net_height / 2.0),
        (0.012, TABLE.net_span, TABLE.net_height),
        (0.04, 0.07, 0.11),
        friction=(0.30, 0.25),
    )
    blade = _blade_body()
    ball = _ball_body()
    ball_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        update_period=0.0,
        history_length=0,
        force_threshold=1e-3,
        filter_prim_paths_expr=[path for _, path in SEMANTIC_FILTERS],
    )


class PaddlePoseAction(ActionTerm):
    """Write the commanded blade pose straight onto a kinematic body.

    The action is the task-frame pose ``[x, y, z, qw, qx, qy, qz]`` declared by
    the shared task config.  It is clipped to the published workspace and its
    quaternion is normalized, matching the MuJoCo mocap fixture; a robot adapter
    replaces this term with joint or end-effector commands.
    """

    cfg: PaddlePoseActionCfg

    def __init__(self, cfg: PaddlePoseActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._asset: RigidObject = env.scene[cfg.asset_name]
        task = env.cfg.task
        low, high = task.action_bounds()
        self._low = torch.tensor(low, device=self.device, dtype=torch.float32)
        self._high = torch.tensor(high, device=self.device, dtype=torch.float32)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = self._raw_actions.clone()
        self._processed_actions[:, 3] = 1.0

    @property
    def action_dim(self) -> int:
        return TableTennisReturnTaskConfig.ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        clamped = torch.clamp(actions, self._low, self._high)
        quaternion = clamped[:, 3:]
        norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        # A degenerate command keeps the identity rotation instead of NaNs.
        safe = torch.where(norm > 1e-6, quaternion / norm, self._processed_actions[:, 3:])
        self._processed_actions = torch.cat((clamped[:, :3], safe), dim=-1)

    def apply_actions(self) -> None:
        pose = self._processed_actions.clone()
        pose[:, :3] += self._env.scene.env_origins
        self._asset.write_root_pose_to_sim(pose)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._processed_actions[env_ids, 3] = 1.0


@configclass
class PaddlePoseActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PaddlePoseAction
    asset_name: str = "blade"


def observe_task_state(env: IsaacTableTennisReturnEnv) -> torch.Tensor:
    """Return the 16-value task observation shared with the MuJoCo environment."""
    return env.task_observation()


def judge_reward(env: IsaacTableTennisReturnEnv) -> torch.Tensor:
    return env.consume_reward()


def judge_terminated(env: IsaacTableTennisReturnEnv) -> torch.Tensor:
    return env.terminated_buffer()


def judge_timeout(env: IsaacTableTennisReturnEnv) -> torch.Tensor:
    return env.truncated_buffer()


def launch_shot(env: IsaacTableTennisReturnEnv, env_ids: torch.Tensor) -> None:
    env.launch_shots(env_ids)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        task_state = ObsTerm(func=observe_task_state)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    blade_pose: PaddlePoseActionCfg = PaddlePoseActionCfg()


@configclass
class EventCfg:
    launch = EventTerm(func=launch_shot, mode="reset")


@configclass
class RewardsCfg:
    shot_events = RewTerm(func=judge_reward, weight=1.0)


@configclass
class TerminationsCfg:
    judged = DoneTerm(func=judge_terminated)
    timeout = DoneTerm(func=judge_timeout, time_out=True)


@configclass
class TableTennisReturnEnvCfg(ManagerBasedRLEnvCfg):
    """Isaac Lab configuration whose task semantics come from the shared config."""

    task: TableTennisReturnTaskConfig = TABLE_TENNIS_RETURN_V0
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)

    scene: TableTennisReturnSceneCfg = TableTennisReturnSceneCfg(
        num_envs=1024, env_spacing=6.0, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.sim = SimulationCfg(
            dt=PHYSICS_DT,
            render_interval=self.task.decimation(PHYSICS_DT),
            gravity=(0.0, 0.0, -9.81),
            physx=PhysxCfg(
                solver_type=1,
                enable_ccd=True,
                bounce_threshold_velocity=0.05,
                min_position_iteration_count=4,
                min_velocity_iteration_count=1,
            ),
        )
        self.decimation = self.task.decimation(PHYSICS_DT)
        self.episode_length_s = self.task.timeout_s


class IsaacTableTennisReturnEnv(ManagerBasedRLEnv):
    """Vectorized return task judged by the shared backend-neutral rules.

    Rules need every physics step: a table-tennis bounce lasts well under one
    control period, so sampling contacts once per control step would drop most
    of them.  The judge therefore runs from a physics callback, and the manager
    terms only publish what the judges already decided.
    """

    cfg: TableTennisReturnEnvCfg

    def __init__(self, cfg: TableTennisReturnEnvCfg, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self.task = cfg.task
        self.shot_bank = ShotBank.from_resource(
            split=self.task.split, task=self.task.bank_resource
        )
        # Shots queued here are launched before any sampled one, in order.  A
        # cross-backend comparison needs to know exactly which shot each
        # environment is playing; training leaves the queue empty.
        self.shot_queue: deque[ShotSpec] = deque()
        self._semantic_names = tuple(name for name, _ in SEMANTIC_FILTERS)
        self._judges = [
            TableTennisReturnJudge(
                timeout_s=self.task.timeout_s, table_spec=self.task.table
            )
            for _ in range(self.num_envs)
        ]
        self._shots: list[ShotSpec | None] = [None] * self.num_envs
        self._episode_time_s = torch.zeros(self.num_envs, device="cpu")
        self._reward = torch.zeros(self.num_envs, device=self.device)
        self._terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Shot selection is seeded here rather than left to the global torch RNG
        # so a run reproduces regardless of what else draws from it.
        self._shot_rng = np.random.default_rng(0 if cfg.seed is None else cfg.seed)
        # Bounded because a vectorized run can finish millions of episodes; drain
        # it between rollouts to keep every record.
        self.episode_records: deque[dict[str, Any]] = deque(maxlen=EPISODE_RECORD_LIMIT)

        atmosphere = Atmosphere(wind=cfg.wind)
        self._wind = torch.tensor(atmosphere.wind, device=self.device, dtype=torch.float32)
        self._magnus_scale = atmosphere.magnus_scale
        self._judging = False
        self.sim.add_physics_callback("shot_skill_judge", self._on_physics_step)

    # -- shared-core plumbing -------------------------------------------------

    @property
    def ball(self) -> RigidObject:
        return self.scene["ball"]

    @property
    def blade(self) -> RigidObject:
        return self.scene["blade"]

    @property
    def current_shots(self) -> tuple[ShotSpec | None, ...]:
        """The shot each environment is playing, by environment index."""
        return tuple(self._shots)

    def _task_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Convert world positions to the per-environment task frame."""
        return positions - self.scene.env_origins

    def task_observation(self) -> torch.Tensor:
        ball = self.ball.data
        blade = self.blade.data
        return torch.cat(
            (
                self._task_positions(ball.root_link_pos_w),
                ball.root_com_lin_vel_w,
                ball.root_com_ang_vel_w,
                self._task_positions(blade.root_link_pos_w),
                blade.root_link_quat_w,
            ),
            dim=-1,
        )

    def consume_reward(self) -> torch.Tensor:
        """Return and clear the reward the judges accumulated this control step."""
        reward = self._reward.clone()
        self._reward.zero_()
        return reward

    def terminated_buffer(self) -> torch.Tensor:
        return self._terminated

    def truncated_buffer(self) -> torch.Tensor:
        return self._truncated

    # -- episode lifecycle ----------------------------------------------------

    def launch_shots(self, env_ids: torch.Tensor) -> None:
        """Reset judges and write one fixed shot per environment (reset event)."""
        indices = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
        if not indices:
            return
        sampled = self._shot_rng.integers(len(self.shot_bank), size=len(indices))
        root_state = self.ball.data.default_root_state[env_ids].clone()
        for row, env_index in enumerate(indices):
            if self.shot_queue:
                shot = self.shot_queue.popleft()
            else:
                shot = self.shot_bank[int(sampled[row])]
            self._shots[env_index] = shot
            self._judges[env_index].reset(shot)
            self._episode_time_s[env_index] = 0.0
            root_state[row, :3] = torch.tensor(shot.position, device=root_state.device)
            root_state[row, 3:7] = torch.tensor(
                (1.0, 0.0, 0.0, 0.0), device=root_state.device
            )
            root_state[row, 7:10] = torch.tensor(
                shot.linear_velocity, device=root_state.device
            )
            root_state[row, 10:13] = torch.tensor(
                shot.angular_velocity, device=root_state.device
            )
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.ball.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
        self._reward[env_ids] = 0.0
        self._terminated[env_ids] = False
        self._truncated[env_ids] = False

    # -- per-physics-step judging --------------------------------------------

    def _semantic_contacts(self) -> list[tuple[SemanticContact, ...]]:
        """Map filtered contact forces onto the shared semantic vocabulary.

        PhysX filtered pairs carry no witness point, so contacts are reported
        without a position; the judge falls back to the ball position, which it
        already supports for adapters that cannot supply one.
        """
        sensor = self.scene.sensors["ball_contacts"]
        matrix = sensor.data.force_matrix_w
        if matrix is None:  # pragma: no cover - requires a live Isaac stage
            return [() for _ in range(self.num_envs)]
        # (num_envs, bodies, filters, 3) -> (num_envs, filters)
        magnitude = torch.linalg.vector_norm(matrix, dim=-1).sum(dim=1)
        active = (magnitude > sensor.cfg.force_threshold).cpu()
        contacts: list[tuple[SemanticContact, ...]] = []
        for env_index in range(self.num_envs):
            row = active[env_index]
            contacts.append(
                tuple(
                    SemanticContact.between(BALL, name)
                    for column, name in enumerate(self._semantic_names)
                    if bool(row[column])
                )
            )
        return contacts

    def _apply_aerodynamics(self) -> None:
        """Add the same quadratic drag and bounded Magnus lift as MuJoCo."""
        ball = self.ball
        velocity = ball.data.root_com_lin_vel_w - self._wind
        spin = ball.data.root_com_ang_vel_w
        speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
        area = BALL_SPEC.cross_section
        force = -0.5 * AIR_DENSITY * BALL_SPEC.drag_coefficient * area * speed * velocity

        spin_axis = torch.linalg.cross(spin, velocity, dim=-1)
        axis_norm = torch.linalg.vector_norm(spin_axis, dim=-1, keepdim=True)
        spin_rate = torch.linalg.vector_norm(spin, dim=-1, keepdim=True)
        # Spin parameter S=r*omega/v with the same 0.35 bound as physics.py.
        lift = torch.clamp(
            self._magnus_scale * BALL_SPEC.radius * spin_rate / speed.clamp_min(1e-9),
            max=0.35,
        )
        magnus = (
            0.5 * AIR_DENSITY * area * speed.pow(2) * lift * spin_axis / axis_norm.clamp_min(1e-9)
        )
        force = force + torch.where(
            (speed > 0.1) & (axis_norm > 1e-9), magnus, torch.zeros_like(magnus)
        )
        # Only buffered here.  ManagerBasedRLEnv writes every asset's wrench
        # before each physics step, and PhysX accumulates applied forces until
        # the step runs: writing it here as well doubled the drag.  Found by
        # scripts/backend_parity.py -- 32 cm of flight divergence at 0.3 s.
        ball.set_external_force_and_torque(
            force.unsqueeze(1), torch.zeros_like(force).unsqueeze(1), is_global=True
        )

    def _on_physics_step(self, dt: float) -> None:
        if self._judging:  # pragma: no cover - guards re-entrant Kit callbacks
            return
        self._judging = True
        try:
            self.scene.sensors["ball_contacts"].update(dt, force_recompute=True)
            self.ball.update(dt)
            self._apply_aerodynamics()
            contacts = self._semantic_contacts()
            positions = self._task_positions(self.ball.data.root_link_pos_w).cpu()
            linear = self.ball.data.root_com_lin_vel_w.cpu()
            angular = self.ball.data.root_com_ang_vel_w.cpu()
            self._episode_time_s += dt
            for env_index, judge in enumerate(self._judges):
                if self._shots[env_index] is None or judge.done:
                    continue
                previously_hit = judge.result.hit
                try:
                    judge.update(
                        time_s=float(self._episode_time_s[env_index]),
                        ball=BallState(
                            position=tuple(positions[env_index].tolist()),
                            linear_velocity=tuple(linear[env_index].tolist()),
                            angular_velocity=tuple(angular[env_index].tolist()),
                        ),
                        contacts=contacts[env_index],
                    )
                except (FloatingPointError, ValueError):
                    judge.abort(
                        failure_reason="numerical",
                        time_s=float(self._episode_time_s[env_index]),
                    )
                self._reward[env_index] += self.task.reward_for(
                    judge.result, previously_hit=previously_hit
                )
                if judge.done:
                    self._record(env_index, judge)
        finally:
            self._judging = False

    def _record(self, env_index: int, judge: TableTennisReturnJudge) -> None:
        result = judge.result
        timed_out = result.failure_reason == "timeout"
        self._terminated[env_index] = not timed_out
        self._truncated[env_index] = timed_out
        self.episode_records.append(result.to_dict())


def make_env_cfg(
    *,
    num_envs: int = 1024,
    device: str = "cuda:0",
    task: TableTennisReturnTaskConfig = TABLE_TENNIS_RETURN_V0,
    split: str | None = None,
) -> TableTennisReturnEnvCfg:
    """Build a ready-to-run configuration for ``num_envs`` parallel returns."""
    cfg = TableTennisReturnEnvCfg()
    cfg.task = task if split is None else replace(task, split=split)
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    # Re-derive rate settings in case the caller supplied a different task.
    cfg.decimation = cfg.task.decimation(PHYSICS_DT)
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = cfg.task.timeout_s
    return cfg
