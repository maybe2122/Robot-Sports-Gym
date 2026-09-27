#!/usr/bin/env python3
"""The Panda table-tennis return on Isaac Lab, driven by the MuJoCo baseline's controller.

"The same robot on two backends" means the same arm, the same shots, the same
judge and the same controller.  This script runs ``ScriptedInterceptController``
-- the reference baseline of ``table-tennis-return-panda-v1``, whose IK uses the
MuJoCo Panda model -- against a Franka articulation simulated by PhysX, and
writes one judged result per shot.  ``scripts/robot_kinematic_parity.py`` is
what makes this legitimate: the two arms share one kinematic chain.

    # Isaac side (Isaac Sim Python; MULTISPORT_FRANKA_USD points at franka.usd)
    PYTHONPATH=src $ISAAC_PYTHON scripts/isaac_panda_rollout.py isaac \\
        --levels L1 L2 --out generated/panda-isaac.json
    # MuJoCo side and comparison (project venv)
    PYTHONPATH=src python scripts/isaac_panda_rollout.py compare generated/panda-isaac.json \\
        --report reports/panda-backend-parity.json --markdown reports/panda-backend-parity.md

What differs between the two, and is reported rather than hidden:

* the blade is a box in PhysX and an ellipsoid in MuJoCo, and PhysX combines
  the ball's and blade's restitution rather than using the calibrated MuJoCo
  ball-rubber contact pair;
* the joint servos use Menagerie's gains as PhysX implicit PD drives, which are
  not the same discretisation as MuJoCo's affine position actuators;
* the judge sees filtered PhysX contacts, which carry no contact point.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

BANK = "table_tennis/return-v1"
MOUNT = (-1.95, 0.0, 0.75)
BLADE_OFFSET = (0.0, 0.0, 0.13)
BLADE_SIZE = (0.17, 0.016, 0.20)
READY = (-1.50953, 1.51280, 1.65599, -2.62808, -0.94185, 1.87744, 2.89730)
STIFFNESS = (4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0)
DAMPING = (450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0)
EFFORT = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
PHYSICS_DT = 0.001
DECIMATION = 5


def _shots(split: str, levels: list[str], limit: int | None):
    from multisport_sim.benchmark.shot_bank import ShotBank

    bank = ShotBank.from_resource(split=split, task=BANK)
    selected = [shot for level in levels for shot in bank.filter(level=level)]
    return bank, selected if limit is None else selected[:limit]


def run_isaac(args: argparse.Namespace) -> None:
    from isaaclab.app import AppLauncher

    AppLauncher(argparse.Namespace(headless=True, device="cpu"))
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveScene
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils import configclass
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from multisport_sim.benchmark.backends import isaac_lab as fixture
    from multisport_sim.benchmark.events import BALL, FLOOR, NET, ROBOT_RACKET, TABLE
    from multisport_sim.benchmark.robot_controllers import ScriptedInterceptController
    from multisport_sim.benchmark.rules.table_tennis import TableTennisReturnJudge
    from multisport_sim.benchmark.task_config import TABLE_TENNIS_RETURN_PANDA_V1
    from multisport_sim.benchmark.types import BallState, SemanticContact

    usd = os.environ.get("MULTISPORT_FRANKA_USD")
    if not usd:
        raise SystemExit("set MULTISPORT_FRANKA_USD to Franka's franka.usd")
    filters = (
        (TABLE, "{ENV_REGEX_NS}/Table"),
        (NET, "{ENV_REGEX_NS}/Net"),
        (FLOOR, "{ENV_REGEX_NS}/Floor"),
        (ROBOT_RACKET, "{ENV_REGEX_NS}/Robot/panda_hand"),
    )
    @configclass
    class PandaSceneCfg(fixture.TableTennisReturnSceneCfg):
        # The kinematic fixture blade is replaced by the arm's own blade.
        blade = None
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd,
                activate_contact_sensors=True,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True, solver_position_iteration_count=16
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=MOUNT,
                joint_pos={
                    **{f"panda_joint{i + 1}": value for i, value in enumerate(READY)},
                    "panda_finger_joint.*": 0.04,
                },
            ),
            actuators={
                f"j{i + 1}": ImplicitActuatorCfg(
                    joint_names_expr=[f"panda_joint{i + 1}"],
                    stiffness=STIFFNESS[i],
                    damping=DAMPING[i],
                    effort_limit_sim=EFFORT[i],
                    armature=0.1,
                )
                for i in range(7)
            }
            | {
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=["panda_finger_joint.*"], stiffness=400.0, damping=40.0
                )
            },
        )
        ball_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            update_period=0.0,
            history_length=0,
            force_threshold=1e-3,
            filter_prim_paths_expr=[path for _, path in filters],
        )

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=PHYSICS_DT,
            device="cpu",
            physx=sim_utils.PhysxCfg(solver_type=1, enable_ccd=True, bounce_threshold_velocity=0.05),
        )
    )
    scene = InteractiveScene(PandaSceneCfg(num_envs=1, env_spacing=6.0))
    # The blade becomes a collider of the hand's rigid body: authored before
    # the physics scene is parsed, it is welded to the flange by construction.
    stage = sim_utils.get_current_stage()
    blade_path = "/World/envs/env_0/Robot/panda_hand/benchmark_blade"
    blade = UsdGeom.Cube.Define(stage, blade_path)
    blade.GetSizeAttr().Set(1.0)
    xform = UsdGeom.Xformable(blade.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*BLADE_OFFSET))
    xform.AddScaleOp().Set(Gf.Vec3f(*BLADE_SIZE))
    UsdPhysics.CollisionAPI.Apply(blade.GetPrim())
    UsdPhysics.MassAPI.Apply(blade.GetPrim()).CreateMassAttr(0.17)
    # MuJoCo's calibrated ball-rubber pair: restitution ~0.84, friction 0.85.
    # The ball averages restitution, so the blade states 2 * 0.84 - 0.937;
    # "max" friction outranks the ball's "average".
    rubber = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.85,
        dynamic_friction=0.85,
        restitution=2 * 0.84 - fixture.BALL_SPEC.restitution,
        friction_combine_mode="max",
        restitution_combine_mode="average",
    )
    rubber.func("/World/Materials/blade_rubber", rubber)
    sim_utils.bind_physics_material(blade_path, "/World/Materials/blade_rubber")
    # The MuJoCo task mounts the blade on panda_nohand.xml: there is no gripper
    # to collide with.  Franka's USD has one, so every collider under the hand
    # and fingers except the blade is switched off -- otherwise the ball can
    # strike the gripper housing and be scored as a hit.
    disabled = 0
    roots = [f"/World/envs/env_0/Robot/{name}"
             for name in ("panda_hand", "panda_leftfinger", "panda_rightfinger")]
    # The asset instances its link geometry; an instance cannot be edited, so
    # the gripper's instances are made concrete first.
    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if prim.IsInstance():
                prim.SetInstanceable(False)
    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if prim.GetPath().pathString.startswith(blade_path):
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                disabled += 1
    print(f"disabled {disabled} gripper colliders", flush=True)
    sim.reset()

    robot, ball, sensor = scene["robot"], scene["ball"], scene.sensors["ball_contacts"]
    names = [name for name, _ in filters]
    arm = [robot.joint_names.index(f"panda_joint{i + 1}") for i in range(7)]
    origin = scene.env_origins[0]
    task = TABLE_TENNIS_RETURN_PANDA_V1
    _, shots = _shots(args.split, args.levels, args.limit)
    controller = ScriptedInterceptController(control_dt=PHYSICS_DT * DECIMATION)
    aero = _AeroShim(ball)
    records = []
    for shot in shots:
        ready = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(ready, torch.zeros_like(ready))
        robot.set_joint_position_target(ready)
        state = ball.data.default_root_state.clone()
        state[0, :3] = torch.tensor(shot.position) + origin
        state[0, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0))
        state[0, 7:10] = torch.tensor(shot.linear_velocity)
        state[0, 10:13] = torch.tensor(shot.angular_velocity)
        ball.write_root_pose_to_sim(state[:, :7])
        ball.write_root_velocity_to_sim(state[:, 7:])
        scene.write_data_to_sim()
        sim.forward()
        scene.update(0.0)
        judge = TableTennisReturnJudge(timeout_s=task.timeout_s, table_spec=task.table)
        judge.reset(shot)
        controller.reset(shot)
        steps = int(task.timeout_s / PHYSICS_DT) + 2
        hand_index = robot.body_names.index("panda_hand")
        contact_blade_velocity = None
        hit_step = -1000
        ball_velocity_after_hit = None
        for step in range(steps):
            time_s = step * PHYSICS_DT
            if step % DECIMATION == 0:
                position = (ball.data.root_link_pos_w[0] - origin).tolist()
                observation = SimpleNamespace(
                    time_s=time_s,
                    ball=BallState(
                        position=tuple(position),
                        linear_velocity=tuple(ball.data.root_com_lin_vel_w[0].tolist()),
                        angular_velocity=tuple(ball.data.root_com_ang_vel_w[0].tolist()),
                    ),
                    robot=SimpleNamespace(
                        joint_positions=tuple(robot.data.joint_pos[0, arm].tolist())
                    ),
                )
                command = controller.act(observation)
                target = robot.data.joint_pos_target.clone()
                target[0, arm] = torch.tensor(command.targets)
                robot.set_joint_position_target(target)
            fixture.IsaacTableTennisReturnEnv._apply_aerodynamics(aero)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(PHYSICS_DT)
            matrix = sensor.data.force_matrix_w
            active = (torch.linalg.vector_norm(matrix, dim=-1).sum(dim=1) > 1e-3)[0]
            contacts = tuple(
                SemanticContact.between(BALL, name)
                for column, name in enumerate(names)
                if bool(active[column])
            )
            judge.update(
                time_s=(step + 1) * PHYSICS_DT,
                ball=BallState(
                    position=tuple((ball.data.root_link_pos_w[0] - origin).tolist()),
                    linear_velocity=tuple(ball.data.root_com_lin_vel_w[0].tolist()),
                    angular_velocity=tuple(ball.data.root_com_ang_vel_w[0].tolist()),
                ),
                contacts=contacts,
            )
            if contact_blade_velocity is None and judge.result.hit:
                contact_blade_velocity = robot.data.body_com_lin_vel_w[0, hand_index].tolist()
                hit_step = step
            if contact_blade_velocity is not None and step == hit_step + 20:
                ball_velocity_after_hit = ball.data.root_com_lin_vel_w[0].tolist()
            if judge.done:
                break
        records.append(
            {
                **judge.result.to_dict(),
                "diagnostic_blade_velocity_at_contact": contact_blade_velocity,
                "diagnostic_ball_velocity_20ms_after_hit": ball_velocity_after_hit,
            }
        )
        print(shot.shot_id, judge.result.hit, judge.result.valid_return,
              judge.result.failure_reason, flush=True)
    payload = {
        "schema": "multisport-panda-rollout-v0",
        "backend": "isaac_lab",
        "bank": BANK,
        "split": args.split,
        "levels": args.levels,
        "controller": controller.controller_id,
        "results": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload) + "\n")
    print(f"wrote {len(records)} episodes to {args.out}", flush=True)
    sys.stdout.flush()
    os._exit(0)


class _AeroShim:
    """Borrow the fixture environment's aerodynamics for a bare ball asset."""

    def __init__(self, ball: Any) -> None:
        import torch

        from multisport_sim.physics import Atmosphere

        self.ball = ball
        self.device = ball.device
        self._wind = torch.zeros(3, device=ball.device)
        self._magnus_scale = Atmosphere().magnus_scale


def compare(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from multisport_sim.benchmark.backends.mujoco_robot import MujocoPandaTableTennisBackend
    from multisport_sim.benchmark.robot_controllers import ScriptedInterceptController
    from multisport_sim.benchmark.runner import RunConfig, run_shots
    from multisport_sim.benchmark.task_config import TABLE_TENNIS_RETURN_PANDA_V1

    isaac = json.loads(args.isaac.read_text())
    _, shots = _shots(isaac["split"], isaac["levels"], None)
    by_id = {record["shot_id"]: record for record in isaac["results"]}
    shots = [shot for shot in shots if shot.shot_id in by_id]
    task = replace(TABLE_TENNIS_RETURN_PANDA_V1, split=isaac["split"])
    backend = MujocoPandaTableTennisBackend()
    controller = ScriptedInterceptController(control_dt=PHYSICS_DT * DECIMATION)
    output = run_shots(backend, controller, shots, config=RunConfig.from_task_config(task))
    rows, summary = [], {}
    for level in isaac["levels"]:
        pairs = [
            (result.to_dict(), by_id[result.shot_id])
            for result in output.results
            if result.level == level
        ]
        n = len(pairs)
        summary[level] = {
            "episodes": n,
            "hit_rate": {"mujoco": sum(a["hit"] for a, _ in pairs) / n,
                         "isaac": sum(b["hit"] for _, b in pairs) / n},
            "valid_return_rate": {"mujoco": sum(a["valid_return"] for a, _ in pairs) / n,
                                  "isaac": sum(b["valid_return"] for _, b in pairs) / n},
            "hit_agreement": sum(a["hit"] == b["hit"] for a, b in pairs) / n,
            "valid_return_agreement": sum(
                a["valid_return"] == b["valid_return"] for a, b in pairs
            ) / n,
        }
        rows += [
            {"shot_id": a["shot_id"], "mujoco": a["failure_reason"] or "valid",
             "isaac": b["failure_reason"] or "valid"}
            for a, b in pairs
        ]
    report = {
        "schema": "multisport-panda-backend-parity-v0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bank": BANK,
        "split": isaac["split"],
        "controller": isaac["controller"],
        "summary": summary,
        "episodes": rows,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Panda on two backends: the same controller, the same shots",
        "",
        (f"`{isaac['controller']}` on `{BANK}` split `{isaac['split']}`. Reference MuJoCo "
        "(Menagerie Panda); candidate Isaac Lab / PhysX (Franka USD). Regenerate with "
        "`scripts/isaac_panda_rollout.py`."),
        "",
        "| Level | n | Hit MuJoCo | Hit Isaac | Hit agreement | Return MuJoCo | Return Isaac | Return agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level, entry in summary.items():
        lines.append(
            f"| {level} | {entry['episodes']} | {entry['hit_rate']['mujoco']:.0%} | "
            f"{entry['hit_rate']['isaac']:.0%} | {entry['hit_agreement']:.0%} | "
            f"{entry['valid_return_rate']['mujoco']:.0%} | {entry['valid_return_rate']['isaac']:.0%} | "
            f"{entry['valid_return_agreement']:.0%} |"
        )
    lines.append("")
    text = "\n".join(lines)
    if args.markdown:
        args.markdown.write_text(text)
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    isaac = sub.add_parser("isaac")
    isaac.add_argument("--split", default="dev")
    isaac.add_argument("--levels", nargs="+", default=["L1", "L2"])
    isaac.add_argument("--limit", type=int)
    isaac.add_argument("--out", type=Path, required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("isaac", type=Path)
    cmp.add_argument("--report", type=Path)
    cmp.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    if args.command == "isaac":
        run_isaac(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
