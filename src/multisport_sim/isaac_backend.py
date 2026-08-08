"""Isaac Sim 5 / Isaac Lab runtime backend.

Import this module only after :class:`isaaclab.app.AppLauncher` has created the
SimulationApp. Isaac/Omniverse extensions are unavailable before that point.
"""

from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
from isaaclab.utils.math import quat_apply
from isaacsim.core.utils.stage import get_current_stage

from .isaac_scene import IsaacPrimitive, IsaacSceneSpec, build_isaac_scene_spec
from .specs import (
    AIR_DENSITY,
    BALLS,
    SHUTTLE_CENTER_OF_PRESSURE_OFFSET,
    SHUTTLE_SKIRT_RADIUS,
    Sport,
)


def _visual(primitive: IsaacPrimitive) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(
        diffuse_color=primitive.color,
        opacity=primitive.opacity,
        roughness=0.62,
    )


def _physics_material(primitive: IsaacPrimitive) -> sim_utils.RigidBodyMaterialCfg | None:
    if not primitive.collision:
        return None
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=primitive.static_friction,
        dynamic_friction=primitive.dynamic_friction,
        restitution=primitive.restitution,
        friction_combine_mode="average",
        restitution_combine_mode="max",
    )


def _spawn_primitive(root: str, primitive: IsaacPrimitive) -> None:
    common = {
        "collision_props": (
            sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)
            if primitive.collision
            else None
        ),
        "visual_material": _visual(primitive),
        "physics_material": _physics_material(primitive),
    }
    if primitive.kind == "cuboid":
        cfg = sim_utils.CuboidCfg(size=primitive.size, **common)
    elif primitive.kind == "sphere":
        cfg = sim_utils.SphereCfg(radius=primitive.size[0], **common)
    elif primitive.kind == "cylinder":
        cfg = sim_utils.CylinderCfg(radius=primitive.size[0], height=primitive.size[1], **common)
    elif primitive.kind == "capsule":
        cfg = sim_utils.CapsuleCfg(radius=primitive.size[0], height=primitive.size[1], **common)
    else:  # pragma: no cover - protected by backend-neutral tests
        raise ValueError(f"unsupported Isaac primitive kind: {primitive.kind}")
    cfg.func(
        f"{root}/{primitive.name}",
        cfg,
        translation=primitive.position,
        orientation=primitive.orientation,
    )


def _ball_material(sport: Sport) -> sim_utils.RigidBodyMaterialCfg:
    spec = BALLS[sport]
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=0.58,
        dynamic_friction=0.48,
        restitution=spec.restitution,
        friction_combine_mode="average",
        restitution_combine_mode="max",
    )


def _spawn_ball(root: str, sport: Sport, position: tuple[float, float, float]) -> RigidObject:
    spec = BALLS[sport]
    cfg = RigidObjectCfg(
        prim_path=f"{root}/{sport.value}/ball",
        spawn=sim_utils.SphereCfg(
            radius=spec.radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                linear_damping=0.0,
                angular_damping=max(0.001, spec.rolling_friction * 2.0),
                max_linear_velocity=120.0,
                max_angular_velocity=3000.0,
                max_depenetration_velocity=20.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=min(0.004, spec.radius * 0.20),
                rest_offset=0.0,
                torsional_patch_radius=spec.radius * 0.15,
                min_torsional_patch_radius=spec.radius * 0.05,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=spec.color[:3], roughness=0.72
            ),
            physics_material=_ball_material(sport),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )
    rigid_object = RigidObject(cfg)
    if sport is Sport.BADMINTON:
        _spawn_shuttle_visuals(cfg.prim_path)
    elif sport is Sport.FOOTBALL:
        _spawn_football_visuals(cfg.prim_path, spec.radius)
    elif sport is Sport.BASKETBALL:
        _spawn_basketball_visuals(cfg.prim_path, spec.radius)
    return rigid_object


def _visual_capsule(
    path: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    color: tuple[float, float, float],
) -> None:
    from .isaac_scene import _capsule_between

    primitive = _capsule_between(path.rsplit("/", 1)[-1], start, end, radius, color, collision=False)
    parent = path.rsplit("/", 1)[0]
    _spawn_primitive(parent, primitive)


def _spawn_shuttle_visuals(ball_path: str) -> None:
    for index in range(16):
        angle = 2 * pi * index / 16
        _visual_capsule(
            f"{ball_path}/feather_{index}",
            (0.010 * cos(angle), 0.010 * sin(angle), 0.0),
            (0.031 * cos(angle), 0.031 * sin(angle), 0.066),
            0.0013,
            (0.97, 0.97, 0.93),
        )


def _spawn_football_visuals(ball_path: str, radius: float) -> None:
    distance = radius * 0.87
    for index, position in enumerate(
        (
            (distance, 0.0, 0.0),
            (-distance, 0.0, 0.0),
            (0.0, distance, 0.0),
            (0.0, -distance, 0.0),
            (0.0, 0.0, distance),
            (0.0, 0.0, -distance),
        )
    ):
        primitive = IsaacPrimitive(
            f"patch_{index}", "sphere", position, (radius * 0.29,), (0.035, 0.04, 0.045), collision=False
        )
        _spawn_primitive(ball_path, primitive)


def _spawn_basketball_visuals(ball_path: str, radius: float) -> None:
    seam_radius = radius * 1.006
    for plane in range(3):
        for index in range(24):
            a0, a1 = 2 * pi * index / 24, 2 * pi * (index + 1) / 24
            p0 = (seam_radius * cos(a0), seam_radius * sin(a0))
            p1 = (seam_radius * cos(a1), seam_radius * sin(a1))
            if plane == 0:
                start, end = (*p0, 0.0), (*p1, 0.0)
            elif plane == 1:
                start, end = (p0[0], 0.0, p0[1]), (p1[0], 0.0, p1[1])
            else:
                start, end = (0.0, *p0), (0.0, *p1)
            _visual_capsule(
                f"{ball_path}/seam_{plane}_{index}",
                start,
                end,
                0.0018,
                (0.08, 0.035, 0.015),
            )


class IsaacSportsSimulation:
    """Regulation multi-sport USD scene driven by PhysX and Isaac Lab tensors."""

    def __init__(
        self,
        scene: str = "campus",
        *,
        device: str = "cuda:0",
        wind: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt: float = 1.0 / 240.0,
    ) -> None:
        self.scene_spec: IsaacSceneSpec = build_isaac_scene_spec(scene)
        self.dt = dt
        self.sim = SimulationContext(
            SimulationCfg(
                dt=dt,
                render_interval=4,
                device=device,
                gravity=(0.0, 0.0, -9.81),
                physx=PhysxCfg(
                    solver_type=1,
                    enable_ccd=True,
                    bounce_threshold_velocity=0.05,
                    min_position_iteration_count=4,
                    min_velocity_iteration_count=1,
                ),
            )
        )
        self.sim.set_camera_view(
            eye=self.scene_spec.camera_eye,
            target=self.scene_spec.camera_target,
        )
        sim_utils.DomeLightCfg(intensity=2200.0, color=(0.78, 0.84, 0.95)).func(
            "/World/Light",
            sim_utils.DomeLightCfg(intensity=2200.0, color=(0.78, 0.84, 0.95)),
        )
        root = "/World/Sports"
        for primitive in self.scene_spec.primitives:
            _spawn_primitive(root, primitive)
        self.balls = {
            ball.sport: _spawn_ball(root, ball.sport, ball.position)
            for ball in self.scene_spec.balls
        }
        self.wind = torch.tensor(wind, dtype=torch.float32, device=self.sim.device)
        self.sim.reset()
        for ball in self.balls.values():
            ball.update(self.dt)

    def launch(self) -> None:
        for sport, ball in self.balls.items():
            velocity = torch.zeros((1, 6), device=self.sim.device)
            velocity[0, :3] = torch.tensor(BALLS[sport].launch_velocity, device=self.sim.device)
            if sport in (Sport.TENNIS, Sport.TABLE_TENNIS):
                velocity[0, 3:] = torch.tensor((0.0, 28.0, 6.0), device=self.sim.device)
            elif sport is Sport.FOOTBALL:
                velocity[0, 3:] = torch.tensor((0.0, 8.0, 0.0), device=self.sim.device)
            elif sport is Sport.BASKETBALL:
                velocity[0, 3:] = torch.tensor((0.0, 3.0, 0.0), device=self.sim.device)
            else:
                velocity[0, 3:] = torch.tensor((0.0, 0.0, 12.0), device=self.sim.device)
            ball.write_root_velocity_to_sim(velocity)

    def reset(self) -> None:
        for ball in self.balls.values():
            state = ball.data.default_root_state.clone()
            ball.write_root_pose_to_sim(state[:, :7])
            ball.write_root_velocity_to_sim(state[:, 7:])
            ball.reset()

    def _aerodynamic_loads(self, sport: Sport, ball: RigidObject) -> tuple[torch.Tensor, torch.Tensor]:
        spec = BALLS[sport]
        velocity = ball.data.root_com_lin_vel_w[0] - self.wind
        angular_velocity = ball.data.root_com_ang_vel_w[0]
        speed = torch.linalg.vector_norm(velocity)
        zeros = torch.zeros(3, device=self.sim.device)
        if speed.item() < 1e-7:
            return zeros, zeros

        if sport is Sport.BADMINTON:
            local_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.sim.device)
            axis = quat_apply(ball.data.root_link_quat_w, local_axis)[0]
            axial = torch.abs(torch.dot(axis, velocity / speed))
            area = pi * SHUTTLE_SKIRT_RADIUS**2 * (0.35 + 0.65 * axial)
            force = -0.5 * AIR_DENSITY * spec.drag_coefficient * area * speed * velocity
            torque = torch.linalg.cross(
                axis * SHUTTLE_CENTER_OF_PRESSURE_OFFSET,
                force,
            ) - 1.5e-5 * angular_velocity
            return force, torque

        force = -0.5 * AIR_DENSITY * spec.drag_coefficient * spec.cross_section * speed * velocity
        spin_axis = torch.linalg.cross(angular_velocity, velocity)
        spin_norm = torch.linalg.vector_norm(spin_axis)
        if spin_norm.item() > 1e-7:
            lift_coefficient = min(
                0.35,
                0.20 * spec.radius * torch.linalg.vector_norm(angular_velocity).item() / speed.item(),
            )
            force += (
                0.5
                * AIR_DENSITY
                * spec.cross_section
                * speed**2
                * lift_coefficient
                * spin_axis
                / spin_norm
            )
        return force, zeros

    def step(self, *, render: bool = True) -> None:
        for sport, ball in self.balls.items():
            force, torque = self._aerodynamic_loads(sport, ball)
            ball.set_external_force_and_torque(
                force.view(1, 1, 3),
                torque.view(1, 1, 3),
                is_global=True,
            )
            ball.write_data_to_sim()
        self.sim.step(render=render)
        for ball in self.balls.values():
            ball.update(self.dt)

    def run(
        self,
        simulation_app: object,
        duration: float,
        *,
        launch: bool = True,
        render: bool = True,
    ) -> int:
        if launch:
            self.launch()
        max_steps = 0 if duration <= 0.0 else round(duration / self.dt)
        step = 0
        while simulation_app.is_running() and (max_steps == 0 or step < max_steps):
            self.step(render=render)
            step += 1
        return step

    def regulation_drop_test(self, sport: Sport, *, render: bool = False) -> float:
        """Measure the first rebound clearance using the sport's regulation drop."""
        if sport not in self.balls:
            raise ValueError(f"{sport.value} ball is not present in scene {self.scene_spec.name}")
        drop_height, surface_height = {
            Sport.TENNIS: (2.54, 0.0),
            Sport.TABLE_TENNIS: (0.30, 0.76),
            Sport.FOOTBALL: (2.00, 0.0),
            Sport.BADMINTON: (1.80, 0.0),
            Sport.BASKETBALL: (1.80, 0.0),
        }[sport]
        ball = self.balls[sport]
        self.reset()
        pose = ball.data.default_root_state[:, :7].clone()
        pose[0, 2] = surface_height + drop_height + BALLS[sport].radius
        ball.write_root_pose_to_sim(pose)
        ball.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.sim.device))

        rising = False
        previous_vz = 0.0
        apex = 0.0
        for _ in range(round(4.0 / self.dt)):
            self.step(render=render)
            vertical_velocity = ball.data.root_com_lin_vel_w[0, 2].item()
            if not rising and previous_vz < 0.0 < vertical_velocity:
                rising = True
            if rising:
                clearance = (
                    ball.data.root_link_pos_w[0, 2].item()
                    - surface_height
                    - BALLS[sport].radius
                )
                apex = max(apex, clearance)
            if rising and previous_vz > 0.0 >= vertical_velocity:
                return apex
            previous_vz = vertical_velocity
        raise RuntimeError(f"{sport.value} did not complete a first rebound within 4 seconds")

    def export_usd(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        get_current_stage().GetRootLayer().Export(str(output))
        return output

    def state_summary(self) -> str:
        fields = []
        for sport, ball in self.balls.items():
            height = ball.data.root_link_pos_w[0, 2].item()
            speed = torch.linalg.vector_norm(ball.data.root_com_lin_vel_w[0]).item()
            fields.append(f"{sport.value}:z={height:.3f}m,v={speed:.3f}m/s")
        return ";".join(fields)
