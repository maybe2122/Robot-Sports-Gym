#!/usr/bin/env python3
"""Is the Isaac Franka the same arm as the MuJoCo Panda?

"The same robot on two backends" starts with the same kinematics.  The MuJoCo
task loads the Panda from MuJoCo Menagerie; Isaac Lab loads Franka's USD.  This
script poses both at the same joint angles and compares the hand frame:

    # 1. Isaac side (Isaac Sim Python; set MULTISPORT_FRANKA_USD to franka.usd)
    PYTHONPATH=src $ISAAC_PYTHON scripts/robot_kinematic_parity.py isaac --out generated/fk-isaac.json
    # 2. compare against MuJoCo (project venv, Menagerie installed)
    PYTHONPATH=src python scripts/robot_kinematic_parity.py compare generated/fk-isaac.json \\
        --report reports/panda-kinematic-parity.json

Joint angles are drawn uniformly from the middle 80% of each joint's range with a
fixed seed, so both sides pose the identical set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
READY = (-1.50953, 1.51280, 1.65599, -2.62808, -0.94185, 1.87744, 2.89730)


def configurations(count: int, seed: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    low, high = np.asarray(LOWER), np.asarray(UPPER)
    return (low + (high - low) * rng.uniform(0.1, 0.9, size=(count, 7))).tolist()


def run_isaac(args: argparse.Namespace) -> None:
    from isaaclab.app import AppLauncher

    app = AppLauncher(argparse.Namespace(headless=True, device="cpu")).app
    del app
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg

    usd = os.environ.get("MULTISPORT_FRANKA_USD")
    if not usd:
        raise SystemExit("set MULTISPORT_FRANKA_USD to Franka's franka.usd")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.001, device="cpu"))
    ready = {f"panda_joint{i + 1}": value for i, value in enumerate(READY)}
    ready["panda_finger_joint.*"] = 0.04
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos=ready),
            actuators={
                "arm": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)
            },
        )
    )
    sim.reset()
    hand = robot.body_names.index("panda_hand")
    poses = []
    for q in configurations(args.count, args.seed):
        position = torch.tensor([[*q, 0.04, 0.04]])
        robot.write_joint_state_to_sim(position, torch.zeros_like(position))
        sim.forward()
        robot.update(0.0)
        poses.append(
            robot.data.body_link_pos_w[0, hand].tolist()
            + robot.data.body_link_quat_w[0, hand].tolist()
        )
    limits = robot.data.joint_pos_limits[0, :7].tolist()
    payload = {"joint_names": robot.joint_names[:7], "joint_limits": limits, "hand_poses": poses,
               "count": args.count, "seed": args.seed, "usd": Path(usd).name}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload) + "\n")
    print(f"wrote {len(poses)} poses to {args.out}", flush=True)
    sys.stdout.flush()
    os._exit(0)


def compare(args: argparse.Namespace) -> int:
    import mujoco

    from multisport_sim.benchmark.assets import FRANKA_PANDA, require_asset

    isaac = json.loads(args.isaac.read_text())
    model = mujoco.MjModel.from_xml_path(str(require_asset(FRANKA_PANDA)))
    data = mujoco.MjData(model)
    # The task's asset is panda_nohand.xml: the flange frame is 'attachment',
    # where the paddle is mounted; Isaac's USD calls the same frame 'panda_hand'.
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "attachment")
    if hand < 0:
        raise SystemExit("the Menagerie Panda has no 'attachment' body")
    position_error, offsets = [], []
    for q, pose in zip(configurations(isaac["count"], isaac["seed"]), isaac["hand_poses"]):
        data.qpos[:7] = q
        mujoco.mj_kinematics(model, data)
        position_error.append(float(np.linalg.norm(data.xpos[hand] - np.asarray(pose[:3]))))
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(pose[3:]))
        offsets.append(data.xmat[hand].reshape(3, 3).T @ rotation.reshape(3, 3))
    offsets = np.asarray(offsets)
    offset = offsets.mean(axis=0)
    mujoco_limits = [list(model.jnt_range[j]) for j in range(7)]
    limit_difference = max(
        abs(a - b)
        for row_a, row_b in zip(isaac["joint_limits"], mujoco_limits, strict=True)
        for a, b in zip(row_a, row_b, strict=True)
    )
    report = {
        "schema": "multisport-robot-kinematic-parity-v0",
        "robot": "franka-panda",
        "reference": "mujoco_menagerie/franka_emika_panda/panda_nohand.xml (body 'attachment')",
        "candidate": f"Isaac Lab articulation from {isaac['usd']} (body 'panda_hand')",
        "configurations": isaac["count"],
        "seed": isaac["seed"],
        "hand_position_difference_m": {"median": float(np.median(position_error)),
                                       "max": float(np.max(position_error))},
        "hand_frame_offset": {
            "rotation_mujoco_to_isaac": np.round(offset, 6).tolist(),
            "max_deviation_across_configurations": float(np.abs(offsets - offset).max()),
            "reading": "a constant rotation is a frame convention, not a kinematic difference",
        },
        "joint_limit_max_difference_rad": float(limit_difference),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    isaac = sub.add_parser("isaac")
    isaac.add_argument("--count", type=int, default=50)
    isaac.add_argument("--seed", type=int, default=0)
    isaac.add_argument("--out", type=Path, required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("isaac", type=Path)
    cmp.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "isaac":
        run_isaac(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
