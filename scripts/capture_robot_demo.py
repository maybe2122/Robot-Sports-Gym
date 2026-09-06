#!/usr/bin/env python3
"""Record an embodied table-tennis episode as an annotated GIF.

The benchmark's output is numbers, and numbers are the right output -- but a
number cannot tell you that the arm reached the ball by folding through the
table, or that the "hit" was the ball bouncing off a stationary blade.  This
script exists so a claim about an embodiment can be looked at.

It runs the real scored path: the frozen shot bank, the registered task, the
reference controller and the same judge that produces the report.  The caption
on each frame is read from that judge, not composed here, so the GIF cannot
disagree with the score.

    python scripts/capture_robot_demo.py --robot g1 --level L2 --shots 3 \\
        --out docs/images/robot/g1-return.gif

Needs the optional ``submission`` extra (Pillow) for the GIF writer.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

if __package__ is None and __name__ == "__main__":  # pragma: no cover - script use
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multisport_sim.benchmark.assets import asset_available, missing_asset_message
from multisport_sim.benchmark.backends.mujoco_robot import (
    MujocoG1TableTennisBackend,
    MujocoPandaTableTennisBackend,
)
from multisport_sim.benchmark.robot import WorkspaceBox
from multisport_sim.benchmark.robot_controllers import (
    SWING_ROBOTS,
    HoldPoseController,
    RandomJointController,
    ScriptedInterceptController,
)
from multisport_sim.benchmark.robots.g1 import PADDLE_SITE as G1_PADDLE_SITE
from multisport_sim.benchmark.robots.panda import PADDLE_SITE as PANDA_PADDLE_SITE
from multisport_sim.benchmark.rules.table_tennis import TableTennisReturnJudge
from multisport_sim.benchmark.shot_bank import VALID_LEVELS, ShotBank
from multisport_sim.benchmark.task_config import (
    TABLE_TENNIS_RETURN_G1_V1,
    TABLE_TENNIS_RETURN_PANDA_V1,
)

# Each embodiment: its task, its backend, its blade site and the asset it needs.
ROBOTS = {
    "panda": (
        TABLE_TENNIS_RETURN_PANDA_V1,
        MujocoPandaTableTennisBackend,
        PANDA_PADDLE_SITE,
    ),
    "g1": (TABLE_TENNIS_RETURN_G1_V1, MujocoG1TableTennisBackend, G1_PADDLE_SITE),
}

# A view that shows the whole rally: the robot, its half of the table, the net
# and where a legal return has to land.  Behind and to the robot's left, which
# is where a coach stands.
CAMERA = {
    "lookat": (-1.05, 0.0, 1.00),
    "distance": 3.1,
    "azimuth": 141.0,
    "elevation": -11.0,
}


def build_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = CAMERA["lookat"]
    camera.distance = CAMERA["distance"]
    camera.azimuth = CAMERA["azimuth"]
    camera.elevation = CAMERA["elevation"]
    return camera


def controller_for(name: str, robot: str, *, control_dt: float):
    swing = SWING_ROBOTS[robot]
    if name == "hold":
        return HoldPoseController(swing.ready_qpos)
    if name == "random":
        low, high = ROBOTS[robot][0].action_bounds()
        return RandomJointController(
            low,
            high,
            control_dt=control_dt,
            ready_qpos=swing.ready_qpos,
            velocity_limit=swing.velocity_limit,
        )
    return ScriptedInterceptController(control_dt=control_dt, robot=swing)


def capture(args: argparse.Namespace) -> list[np.ndarray]:
    """Run ``args.shots`` episodes and return every rendered frame."""
    task, backend_type, paddle_site = ROBOTS[args.robot]
    task = replace(task, split=args.split)
    bank = ShotBank.from_resource(split=task.split, task=task.bank_resource)
    shots = [shot for shot in bank if shot.level == args.level][: args.shots]
    if not shots:
        raise SystemExit(f"no {args.level} shots in split {task.split!r}")

    backend = backend_type(
        workspace=WorkspaceBox(
            task.workspace.position_low, task.workspace.position_high
        )
    )
    control_dt = task.control_dt(backend.timestep)
    decimation = task.decimation(backend.timestep)
    controller = controller_for(args.controller, args.robot, control_dt=control_dt)
    site = int(
        mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_SITE, paddle_site)
    )

    renderer = mujoco.Renderer(backend.model, height=args.height, width=args.width)
    camera = build_camera()
    # One frame per this many physics steps.  The GIF plays at real time when
    # ``fps`` matches; a slower rate would misrepresent how fast the arm moves.
    stride = max(1, round(1.0 / (args.fps * backend.timestep)))

    frames: list[np.ndarray] = []
    for shot in shots:
        backend.reset()
        controller.reset(shot)
        backend.launch_ball(shot)
        judge = TableTennisReturnJudge(timeout_s=task.timeout_s, table_spec=task.table)
        judge.reset(shot)
        command = None
        steps = round(task.timeout_s / backend.timestep)
        for step in range(steps):
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
                renderer.update_scene(backend.data, camera=camera)
                frames.append(
                    annotate(
                        np.array(renderer.render(), dtype=np.uint8),
                        shot_id=shot.shot_id,
                        level=args.level,
                        robot=task.robot_id,
                        controller=controller.controller_id,
                        blade=backend.data.site_xpos[site],
                        judge=judge,
                    )
                )
            if judge.done:
                break
        # Hold the verdict on screen so a reader can actually read it.
        frames.extend([frames[-1]] * max(1, args.fps // 2))
    return frames


def annotate(
    frame: np.ndarray,
    *,
    shot_id: str,
    level: str,
    robot: str,
    controller: str,
    blade: np.ndarray,
    judge: object,
) -> np.ndarray:
    """Caption a frame with what the judge currently believes.

    Every word here comes from the judge's live result, so a frame can never
    claim an outcome the report would not.
    """
    from PIL import Image, ImageDraw

    result = judge.result  # type: ignore[attr-defined]
    outcome = "in flight"
    if result.failure_reason:
        outcome = f"FAIL: {result.failure_reason}"
    elif result.valid_return:
        outcome = "VALID RETURN"
    elif result.hit:
        outcome = "hit"
    lines = [
        f"{robot}  |  {controller}",
        f"{level}  {shot_id}",
        f"blade  x{blade[0]:+.2f} y{blade[1]:+.2f} z{blade[2]:+.2f}",
        outcome,
    ]
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 12 * len(lines) + 6), fill=(16, 18, 22))
    for index, text in enumerate(lines):
        colour = (235, 235, 235)
        if index == len(lines) - 1:
            colour = (
                (250, 120, 110)
                if outcome.startswith("FAIL")
                else (120, 230, 140)
                if outcome == "VALID RETURN"
                else (245, 205, 110)
                if outcome == "hit"
                else (170, 175, 185)
            )
        draw.text((6, 3 + 12 * index), text, fill=colour)
    return np.array(image)


def write_gif(frames: Sequence[np.ndarray], path: Path, *, fps: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=round(1000.0 / fps),
        loop=0,
        optimize=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", choices=tuple(ROBOTS), default="g1")
    parser.add_argument("--controller", choices=("intercept", "hold", "random"), default="intercept")
    parser.add_argument("--level", choices=VALID_LEVELS, default="L2")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    task = ROBOTS[args.robot][0]
    from multisport_sim.benchmark.assets import ASSETS

    asset = ASSETS["unitree_g1" if args.robot == "g1" else "franka_emika_panda"]
    if not asset_available(asset):
        print(missing_asset_message(asset), file=sys.stderr)
        return 2

    frames = capture(args)
    write_gif(frames, args.out, fps=args.fps)
    print(f"{args.out}  ({len(frames)} frames, {task.task_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
