#!/usr/bin/env python3
"""Replay a scored MuJoCo episode inside the Isaac Sim GUI.

**This is a viewer, not a second physics backend.**  Every number it shows was
computed by MuJoCo on the scored path -- the frozen shot bank, the registered
task, the reference controller and the judge that writes the report.  Isaac
draws it.  PhysX integrates nothing here: the robot's joints and the ball's pose
are written to the stage every frame, which is why the prims live under
``/World/Playback`` and say so in the stage tree.

That distinction is the whole reason this file is not called a backend.  A real
Isaac implementation of the embodied task would have PhysX produce the contact
that the judge then scores, and its numbers would have to be compared against
MuJoCo's rather than copied from them.  That work is TODO #4/#5.  What this
gives you is the thing you can actually look at, plus the first evidence that
the G1 asset, the joint naming and the scene compose correctly on the Isaac
side -- which every real backend would need anyway.

Two steps, two interpreters, on purpose: the recorder never imports Isaac and
the player never imports MuJoCo.

    # 1. record, in the project's own environment
    python scripts/replay_isaac.py record --robot g1 --level L2 --shots 3 \\
        --out generated/g1-l2.json

    # 2. play, in Isaac Sim's python
    PYTHONPATH=src:$ISAACLAB/source/isaaclab $ISAACSIM/python.sh \\
        scripts/replay_isaac.py play --trace generated/g1-l2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ is None and __name__ == "__main__":  # pragma: no cover - script use
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The 29-DoF G1, which is the model the MuJoCo side uses: a fixed rubber hand,
# no dexterous fingers.  The with-hands USD in the Isaac Lab asset layout has 43
# joints and would put fourteen finger joints on a robot that, in this task,
# has a paddle bolted to its wrist.  Overridable: the asset is not vendored.
DEFAULT_G1_USD = os.environ.get("MULTISPORT_G1_USD")
WRIST_LINK = "right_wrist_yaw_link"
# Where the blade sits in the USD's wrist frame.  FITTED, not copied.
#
# The MJCF bolts the paddle at (0.145, -0.003, 0.0) in its own wrist frame, and
# using that number here put the paddle 75.6 mm from where MuJoCo had it --
# most of a blade radius.  The two descriptions of "the same" G1 do not share a
# wrist frame: with identical joint angles the USD puts the wrist link itself
# 18.1 mm (max 28.9 mm) from where the MJCF puts it, and the offset that would
# reproduce MuJoCo's blade exactly is (0.138 +- 0.013, -0.029 +- 0.009,
# -0.070 +- 0.013) -- a spread, not a constant, so no single number makes them
# agree.  The median is used, and ``--check-frames`` reports what is left over.
#
# That residual is the honest headline of this viewer: an Isaac *backend*,
# unlike this playback, would have to reconcile those two descriptions rather
# than fit between them.
PADDLE_LOCAL_POS = (0.1379, -0.0293, -0.0701)
TRACE_SCHEMA = "multisport-isaac-playback-v1"


# --------------------------------------------------------------------------
# record: MuJoCo only
# --------------------------------------------------------------------------


def record(args: argparse.Namespace) -> int:
    """Run the scored path and write one frame per ``--rate`` seconds."""
    import mujoco
    import numpy as np

    from multisport_sim.benchmark.backends.mujoco_robot import (
        MujocoG1TableTennisBackend,
        MujocoPandaTableTennisBackend,
    )
    from multisport_sim.benchmark.robot import WorkspaceBox
    from multisport_sim.benchmark.robot_controllers import (
        SWING_ROBOTS,
        HoldPoseController,
        ScriptedInterceptController,
    )
    from multisport_sim.benchmark.rules.table_tennis import TableTennisReturnJudge
    from multisport_sim.benchmark.shot_bank import ShotBank
    from multisport_sim.benchmark.task_config import (
        TABLE_TENNIS_RETURN_G1_V1,
        TABLE_TENNIS_RETURN_PANDA_V1,
    )

    robots = {
        "g1": (TABLE_TENNIS_RETURN_G1_V1, MujocoG1TableTennisBackend, "g1_"),
        "panda": (TABLE_TENNIS_RETURN_PANDA_V1, MujocoPandaTableTennisBackend, "rb_"),
    }
    task, backend_type, prefix = robots[args.robot]
    bank = ShotBank.from_resource(split=args.split, task=task.bank_resource)
    shots = [shot for shot in bank if shot.level == args.level][: args.shots]
    if not shots:
        raise SystemExit(f"no {args.level} shots in split {args.split!r}")

    backend = backend_type(
        workspace=WorkspaceBox(
            task.workspace.position_low, task.workspace.position_high
        )
    )
    swing = SWING_ROBOTS[args.robot]
    control_dt = task.control_dt(backend.timestep)
    controller = (
        HoldPoseController(swing.ready_qpos)
        if args.controller == "hold"
        else ScriptedInterceptController(control_dt=control_dt, robot=swing)
    )
    decimation = task.decimation(backend.timestep)
    stride = max(1, round(args.rate / backend.timestep))

    # Record every robot joint by NAME, not by index.  Isaac orders an
    # articulation's joints its own way, and a positional dump would replay a
    # convincing but wrong pose -- the exact failure this benchmark's
    # observation layout exists to prevent elsewhere.
    model = backend.model
    joint_names = []
    joint_adr = []
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if name and name.startswith(prefix):
            joint_names.append(name.removeprefix(prefix))
            joint_adr.append(int(model.jnt_qposadr[joint]))
    # The blade's own pose, so the viewer can draw the paddle without needing
    # the USD's wrist frame to agree with the MJCF's.  It is the same site the
    # judge's contact is measured against, so the paddle a viewer sees is the
    # paddle that did the hitting.
    blade_site = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}benchmark_paddle_center")
    )
    if blade_site < 0:  # pragma: no cover - model contract
        raise SystemExit("the robot model has no benchmark paddle site")
    # The wrist body itself, so a viewer can ask the sharper question: given the
    # same joint angles, does the other simulator put this link in the same
    # place?  That separates "the paddle is bolted wrong" from "the two models
    # disagree about the arm", which the blade pose alone conflates.
    wrist_body = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}right_wrist_yaw_link")
    )
    ball_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "table_tennis_ball_free"
    )
    ball_adr = (
        int(model.jnt_qposadr[ball_joint])
        if ball_joint >= 0
        else int(model.jnt_qposadr[_first_free_joint(model, mujoco)])
    )

    episodes = []
    for shot in shots:
        backend.reset()
        controller.reset(shot)
        backend.launch_ball(shot)
        judge = TableTennisReturnJudge(timeout_s=task.timeout_s, table_spec=task.table)
        judge.reset(shot)
        frames: list[dict[str, object]] = []
        command = None
        for step in range(round(task.timeout_s / backend.timestep)):
            if step % decimation == 0:
                command = controller.act(backend.observe())
            backend.step(command)
            if backend.safety_violations():
                judge.abort(failure_reason="safety", time_s=backend.time)
            else:
                judge.update(
                    time_s=backend.time,
                    ball=backend.get_ball_state(),
                    contacts=backend.semantic_contacts(),
                )
            if step % stride == 0:
                qpos = backend.data.qpos
                quaternion = np.empty(4)
                mujoco.mju_mat2Quat(quaternion, backend.data.site_xmat[blade_site])
                frames.append(
                    {
                        "t": round(float(backend.time), 5),
                        "joints": [round(float(qpos[adr]), 6) for adr in joint_adr],
                        "ball": [round(float(v), 6) for v in qpos[ball_adr : ball_adr + 7]],
                        "blade": [
                            *(round(float(v), 6) for v in backend.data.site_xpos[blade_site]),
                            *(round(float(v), 6) for v in quaternion),
                        ],
                        **(
                            {
                                "wrist": [
                                    *(
                                        round(float(v), 6)
                                        for v in backend.data.xpos[wrist_body]
                                    ),
                                    *(
                                        round(float(v), 6)
                                        for v in backend.data.xquat[wrist_body]
                                    ),
                                ]
                            }
                            if wrist_body >= 0
                            else {}
                        ),
                    }
                )
            if judge.done:
                break
        result = judge.result
        episodes.append(
            {
                "shot_id": shot.shot_id,
                "level": shot.level,
                "hit": bool(result.hit),
                "valid_return": bool(result.valid_return),
                "failure_reason": result.failure_reason,
                "frames": frames,
            }
        )
        print(
            f"{shot.shot_id}: hit={result.hit} return={result.valid_return} "
            f"fail={result.failure_reason} frames={len(frames)}",
            file=sys.stderr,
        )

    if args.hits_only:
        kept = [item for item in episodes if item["hit"]]
        if not kept:
            raise SystemExit(
                f"none of the {len(episodes)} recorded shots produced a hit; "
                "raise --shots or drop --hits-only"
            )
        print(f"keeping {len(kept)}/{len(episodes)} episodes that hit", file=sys.stderr)
        episodes = kept

    mount = tuple(float(v) for v in backend.mount.position)
    trace = {
        "schema": TRACE_SCHEMA,
        "source": "mujoco",
        "note": (
            "State computed by MuJoCo on the scored path. Isaac playback of this "
            "trace draws it and integrates nothing."
        ),
        "task": task.task_id,
        "robot": task.robot_id,
        "shot_bank": task.bank_resource,
        "split": args.split,
        "controller": controller.controller_id,
        "mount": list(mount),
        "root_offset_z": 0.793 if args.robot == "g1" else 0.0,
        "frame_rate_hz": round(1.0 / args.rate, 3),
        "joint_names": joint_names,
        "blade_half_extents": [0.085, 0.008, 0.10],
        "episodes": episodes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=1) + "\n", encoding="utf-8")
    total = sum(len(item["frames"]) for item in episodes)
    print(f"{out}  ({len(episodes)} episodes, {total} frames, {len(joint_names)} joints)")
    return 0


def _first_free_joint(model: object, mujoco: object) -> int:  # pragma: no cover
    for joint in range(model.njnt):  # type: ignore[attr-defined]
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:  # type: ignore[attr-defined]
            return joint
    raise RuntimeError("the scene has no free joint for the ball")


# --------------------------------------------------------------------------
# play: Isaac only
# --------------------------------------------------------------------------


def play(args: argparse.Namespace) -> int:
    """Open the Isaac GUI and drive the stage from a recorded trace."""
    from isaaclab.app import AppLauncher

    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    if trace.get("schema") != TRACE_SCHEMA:
        raise SystemExit(f"{args.trace} is not a {TRACE_SCHEMA} trace")

    launcher_args = argparse.Namespace(
        headless=args.headless,
        device=args.device,
        # A Camera prim refuses to spawn without this, and the flag is what
        # turns on the render pipeline the capture reads from.
        enable_cameras=bool(args.capture),
    )
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors.camera import Camera, CameraCfg

    from multisport_sim.isaac_backend import IsaacSportsSimulation

    banner = (
        "\n"
        "=" * 78 + "\n"
        "  PLAYBACK, NOT SIMULATION\n"
        f"  state source : {trace['source']} ({trace['task']})\n"
        f"  controller   : {trace['controller']}\n"
        "  PhysX integrates nothing in this window; joints and ball pose are\n"
        "  written every frame. Prims live under /World/Playback.\n"
        + "=" * 78 + "\n"
    )
    print(banner, flush=True)

    simulation = IsaacSportsSimulation("table_tennis", device=args.device)
    # The scene's own ball is the thing we drive; nothing else in the scene moves.
    from multisport_sim.specs import Sport

    ball = simulation.balls[Sport.TABLE_TENNIS]

    mount = trace["mount"]
    root = (mount[0], mount[1], mount[2] + trace["root_offset_z"])
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Playback/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=args.usd,
                # A playback root must not fall over or be pushed: it is a
                # display of a pose, and the MuJoCo run it comes from had a
                # fixed pelvis anyway.
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=root),
            # No actuator groups: this articulation is never driven, its joint
            # state is written every frame.  The field is required, so say
            # "none" explicitly rather than configuring motors that would
            # imply a control law the playback does not have.
            actuators={},
        )
    )
    # The paddle, bolted to the wrist.  Parented under the wrist link with a
    # fixed local transform -- the way a 3D-printed mount holds it -- rather
    # than drawn at a pose written every frame.  It then follows the arm because
    # it is part of the arm, which is also how the MuJoCo model has it.
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom

    stage = get_current_stage()
    wrist_path = _find_link(stage, "/World/Playback/Robot", WRIST_LINK)
    half = trace.get("blade_half_extents", [0.085, 0.008, 0.10])
    blade_cfg = sim_utils.CylinderCfg(
        radius=float(half[0]),
        height=float(half[1]) * 2.0,
        axis="Y",
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.05, 0.035)),
        collision_props=None,
    )
    paddle_path = f"{wrist_path}/benchmark_paddle"
    blade_cfg.func(paddle_path, blade_cfg)
    paddle_xform = UsdGeom.Xformable(stage.GetPrimAtPath(paddle_path))
    paddle_xform.MakeMatrixXform().Set(
        Gf.Matrix4d(1.0).SetTranslateOnly(Gf.Vec3d(*PADDLE_LOCAL_POS))
    )
    print(f"[playback] paddle bolted to {paddle_path}", flush=True)

    # An in-scene camera, so a capture is what Isaac rendered rather than what
    # happened to be on top of the desktop.  Same approach the squash demo uses.
    camera = None
    if args.capture:
        camera = Camera(
            CameraCfg(
                prim_path="/World/Playback/Camera",
                update_period=0.0,
                height=args.capture_height,
                width=args.capture_width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=22.0,
                    focus_distance=6.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.05, 100.0),
                ),
            )
        )

    simulation.sim.reset()
    # After reset: an articulation has no data buffers until the sim is playing.
    wrist_index = robot.data.body_names.index(WRIST_LINK)
    simulation.sim.set_camera_view(eye=args.eye, target=args.target)
    if camera is not None:
        camera.set_world_poses_from_view(
            torch.tensor([list(args.eye)], device=simulation.sim.device),
            torch.tensor([list(args.target)], device=simulation.sim.device),
        )

    # Match by name and drive only the intersection.  The Isaac USD is the
    # with-hands G1 and carries finger joints the 29-DoF MJCF does not have;
    # those stay at their default, which is honest -- MuJoCo never moved them,
    # so the viewer must not either.  Anything the trace *does* carry must land
    # somewhere, or the replayed pose would be silently partial.
    available = {name: index for index, name in enumerate(trace["joint_names"])}
    driven_ids = [
        index for index, name in enumerate(robot.joint_names) if name in available
    ]
    order = [available[robot.joint_names[index]] for index in driven_ids]
    unplaced = set(available) - set(robot.joint_names)
    if unplaced:
        raise SystemExit(
            f"the articulation has no joint for {sorted(unplaced)}; the trace and "
            "the USD describe different robots"
        )
    print(
        f"[playback] driving {len(driven_ids)}/{len(robot.joint_names)} joints "
        f"matched by name; {len(robot.joint_names) - len(driven_ids)} left at "
        "their default (hands, absent from the MJCF)",
        flush=True,
    )

    if camera is not None:
        # Let the RTX renderer converge before the first captured frame; the
        # squash demo warms up for the same reason.  An unconverged frame is
        # visibly noisy and would misrepresent the render, not the motion.
        for _ in range(args.capture_warmup):
            simulation.sim.step(render=True)
            camera.update(simulation.dt)

    device = simulation.sim.device
    zeros = torch.zeros((1, len(driven_ids)), device=device)
    zero_twist = torch.zeros((1, 6), device=device)
    steps_per_frame = max(1, round((1.0 / trace["frame_rate_hz"]) / simulation.dt))

    captured: list = []
    errors: list[float] = []
    checks = 0
    for loop in range(args.loops):
        for episode in trace["episodes"]:
            outcome = (
                f"FAIL:{episode['failure_reason']}"
                if episode["failure_reason"]
                else "VALID RETURN"
                if episode["valid_return"]
                else "hit"
                if episode["hit"]
                else "no contact"
            )
            print(
                f"[playback] loop {loop + 1}/{args.loops}  {episode['shot_id']}"
                f"  ({episode['level']})  ->  {outcome}",
                flush=True,
            )
            for frame in episode["frames"]:
                if not simulation_app.is_running():
                    return 0
                angles = torch.tensor(
                    [[frame["joints"][index] for index in order]],
                    dtype=torch.float32,
                    device=device,
                )
                pose = torch.tensor([frame["ball"]], dtype=torch.float32, device=device)
                # Re-pin before EVERY physics step, not once per rendered frame.
                # This articulation has no actuators, so any step taken between
                # two writes is a step of free fall that the next write snaps
                # back -- which reads on screen as a high-frequency shake. It is
                # an artefact of the viewer, never of the recorded motion.
                for _ in range(steps_per_frame):
                    robot.write_joint_state_to_sim(angles, zeros, joint_ids=driven_ids)
                    ball.write_root_pose_to_sim(pose)
                    ball.write_root_velocity_to_sim(zero_twist)
                    simulation.sim.step(render=True)
                robot.update(simulation.dt)
                ball.update(simulation.dt)
                if checks < args.check_frames:
                    checks += 1
                    _check_paddle(
                        robot,
                        wrist_index,
                        frame["blade"],
                        errors,
                        frame.get("wrist"),
                    )
                if camera is not None:
                    camera.update(simulation.dt)
                    if len(captured) < args.capture_frames:
                        pixels = camera.data.output["rgb"][0].detach().cpu().numpy()
                        captured.append((outcome, pixels[..., :3].copy()))
        if camera is not None and len(captured) >= args.capture_frames:
            break

    if errors:
        import statistics

        distances = [item[0] for item in errors]
        implied = [item[1] for item in errors]
        print(
            f"[playback] paddle placement vs the recorded MuJoCo blade pose over "
            f"{len(errors)} frames: median {statistics.median(distances) * 1000:.1f} mm, "
            f"max {max(distances) * 1000:.1f} mm",
            flush=True,
        )
        wrist_errors = [item[2] for item in errors if item[2] is not None]
        if wrist_errors:
            print(
                "[playback] wrist link placement, same joint angles, Isaac vs "
                f"MuJoCo: median {statistics.median(wrist_errors) * 1000:.1f} mm, "
                f"max {max(wrist_errors) * 1000:.1f} mm",
                flush=True,
            )
        axes = list(zip(*implied, strict=True))
        summary = ", ".join(
            f"{statistics.median(axis):+.4f}+-{statistics.pstdev(axis):.4f}"
            for axis in axes
        )
        print(f"[playback] offset that would match MuJoCo exactly: ({summary})", flush=True)
    if camera is not None and captured:
        _write_capture(captured, Path(args.capture), trace)
        print(f"[playback] wrote {args.capture} ({len(captured)} frames)", flush=True)
    return 0


def _find_link(stage: object, root: str, name: str) -> str:
    """Prim path of the named link inside a spawned articulation."""
    from pxr import Sdf

    prim = stage.GetPrimAtPath(Sdf.Path(root))  # type: ignore[attr-defined]
    if not prim or not prim.IsValid():
        raise SystemExit(f"nothing was spawned at {root}")
    for child in iter(stage.Traverse()):  # type: ignore[attr-defined]
        path = str(child.GetPath())
        if path.startswith(root) and child.GetName() == name:
            return path
    raise SystemExit(f"the USD under {root} has no link named {name!r}")


def _check_paddle(
    robot: object,
    wrist_index: int,
    blade: list,
    errors: list,
    wrist: list | None = None,
) -> None:
    """Compare the bolted paddle's world position with MuJoCo's blade site.

    The USD link frame and the MJCF body frame both descend from Unitree's own
    URDF, so they *should* agree -- but "should" is what this benchmark keeps
    turning into a measurement.  A large error would mean the paddle is bolted
    to the wrong place in Isaac, and the picture would be wrong in a way no
    amount of looking at it would reveal.

    The wrist pose is read from the articulation, not from a USD xform cache.
    An articulation is driven through PhysX tensors and its USD attributes keep
    the spawn pose, so the cache measured a rest pose against a moving blade and
    reported a 382 mm error that was entirely the reader's fault.
    """
    import torch

    position = robot.data.body_pos_w[0, wrist_index]
    quaternion = robot.data.body_quat_w[0, wrist_index]
    local = torch.tensor(PADDLE_LOCAL_POS, dtype=position.dtype, device=position.device)
    w, x, y, z = (float(v) for v in quaternion)
    rotation = torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=position.dtype,
        device=position.device,
    )
    world = position + rotation @ local
    target = torch.tensor(blade[:3], dtype=position.dtype, device=position.device)
    # Also solve for the offset that WOULD reproduce MuJoCo's blade exactly.
    # If that comes out constant across frames the two frames differ by a fixed
    # transform and the constant above is simply wrong; if it wanders, the two
    # models disagree about the kinematics and no constant would fix it.
    implied = rotation.T @ (target - position)
    # The sharper number: same joint angles, does Isaac put the wrist where
    # MuJoCo put it?  If this is large, nothing about the paddle mount can fix
    # the picture, because the arm itself is in a different pose.
    wrist_error = None
    if wrist is not None:
        reference = torch.tensor(
            wrist[:3], dtype=position.dtype, device=position.device
        )
        wrist_error = float(torch.linalg.norm(position - reference))
    errors.append(
        (
            float(torch.linalg.norm(world - target)),
            [float(v) for v in implied],
            wrist_error,
        )
    )


def _write_capture(captured: list, path: Path, trace: dict) -> None:
    """Save the Isaac render, captioned with the same provenance as the banner."""
    from PIL import Image, ImageDraw

    frames = []
    for outcome, pixels in captured:
        image = Image.fromarray(pixels, mode="RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 42), fill=(16, 18, 22))
        draw.text((6, 3), "Isaac Sim  |  PLAYBACK of MuJoCo state, PhysX integrates nothing", fill=(245, 205, 110))
        draw.text((6, 16), f"{trace['robot']}  |  {trace['controller']}", fill=(225, 225, 225))
        draw.text((6, 29), f"judge: {outcome}", fill=(150, 220, 160))
        frames.append(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=True
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    recorder = sub.add_parser("record", help="run the scored MuJoCo path and dump a trace")
    recorder.add_argument("--robot", choices=("g1", "panda"), default="g1")
    recorder.add_argument("--controller", choices=("intercept", "hold"), default="intercept")
    recorder.add_argument("--level", default="L2")
    recorder.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    recorder.add_argument("--shots", type=int, default=3)
    recorder.add_argument(
        "--rate", type=float, default=1.0 / 120.0, help="seconds between recorded frames"
    )
    recorder.add_argument(
        "--hits-only",
        action="store_true",
        help="drop episodes the judge did not score as a hit",
    )
    recorder.add_argument("--out", required=True)

    player = sub.add_parser("play", help="draw a trace in the Isaac Sim GUI")
    player.add_argument("--trace", required=True)
    player.add_argument(
        "--usd",
        default=DEFAULT_G1_USD,
        required=DEFAULT_G1_USD is None,
        help="the 29-DoF G1 USD (or set MULTISPORT_G1_USD); the asset is not vendored",
    )
    player.add_argument("--device", default="cpu")
    player.add_argument("--headless", action="store_true")
    player.add_argument("--loops", type=int, default=20)
    player.add_argument("--eye", type=float, nargs=3, default=(-3.6, -2.4, 2.0))
    player.add_argument("--target", type=float, nargs=3, default=(-1.2, 0.0, 0.9))
    player.add_argument("--capture", help="write the Isaac render to this GIF")
    player.add_argument("--capture-frames", type=int, default=220)
    player.add_argument("--capture-warmup", type=int, default=60)
    player.add_argument(
        "--check-frames",
        type=int,
        default=120,
        help="frames to cross-check the bolted paddle against the recorded pose",
    )
    player.add_argument("--capture-width", type=int, default=640)
    player.add_argument("--capture-height", type=int, default=400)

    args = parser.parse_args(argv)
    return record(args) if args.mode == "record" else play(args)


if __name__ == "__main__":
    raise SystemExit(main())
