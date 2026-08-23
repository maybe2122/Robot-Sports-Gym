"""Render the complete PhysX squash scoring rally as an annotated GIF."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from itertools import pairwise
from pathlib import Path

from isaaclab.app import AppLauncher


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--duration", type=float, default=9.0)
    result.add_argument("--fps", type=float, default=12.0)
    result.add_argument("--width", type=int, default=640)
    result.add_argument("--height", type=int, default=360)
    result.add_argument(
        "--output",
        default="docs/images/shot-skill/squash-serve-score-demo.gif",
    )
    result.add_argument(
        "--report",
        default="docs/images/shot-skill/squash-serve-score-demo.json",
    )
    AppLauncher.add_app_launcher_args(result)
    return result


ARGS = parser().parse_args()
ARGS.enable_cameras = True
APP = AppLauncher(ARGS).app


def main() -> int:
    import torch

    try:
        from PIL import Image, ImageChops, ImageDraw, ImageFont
    except ModuleNotFoundError:
        raise RuntimeError(
            "capturing the squash GIF requires Pillow in the Isaac Python environment"
        ) from None

    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg

    from multisport_sim.isaac_backend import IsaacSportsSimulation
    from multisport_sim.squash_demo import validate_gif, validate_score_report

    if ARGS.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if ARGS.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if ARGS.width <= 0 or ARGS.height <= 0:
        raise ValueError("--width and --height must be positive")

    simulation = IsaacSportsSimulation("squash", device=ARGS.device)

    # The regulation ball is only 40 mm across. This visual-only child makes it
    # readable in the documentation render without changing PhysX collision.
    marker_cfg = sim_utils.SphereCfg(
        radius=0.055,
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.80, 0.04),
            emissive_color=(0.25, 0.12, 0.0),
            roughness=0.45,
        ),
    )
    marker_cfg.func(
        "/World/Sports/squash/ball/demo_visual_marker",
        marker_cfg,
        translation=(0.0, 0.0, 0.0),
    )

    camera = Camera(
        CameraCfg(
            prim_path="/World/SquashDemoCamera",
            update_period=0.0,
            height=ARGS.height,
            width=ARGS.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=19.0,
                focus_distance=8.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 100.0),
            ),
        )
    )
    simulation.sim.reset()
    camera.set_world_poses_from_view(
        torch.tensor([[4.55, 0.0, 2.65]], device=simulation.sim.device),
        torch.tensor([[-2.15, 0.0, 1.35]], device=simulation.sim.device),
    )

    for _ in range(8):
        simulation.step(render=True)
        camera.update(simulation.dt)

    captured: list[tuple[float, object]] = []
    next_capture = 0.0

    def capture_frame(elapsed: float) -> None:
        nonlocal next_capture
        camera.update(simulation.dt)
        if elapsed + 1e-9 < next_capture:
            return
        pixels = camera.data.output["rgb"][0].detach().cpu().numpy()
        captured.append((elapsed, Image.fromarray(pixels, mode="RGB")))
        next_capture += 1.0 / ARGS.fps

    report = simulation.run_squash_demo(APP, ARGS.duration, frame_callback=capture_frame)
    validate_score_report(report)

    events = report["events"]
    rendered = [
        _annotate_frame(image, elapsed, events, ImageDraw, ImageFont)
        for elapsed, image in captured
    ]
    if len(rendered) < 12:
        raise RuntimeError(f"expected at least 12 captured frames, received {len(rendered)}")
    changed = sum(
        ImageChops.difference(previous, current).getbbox() is not None
        for previous, current in pairwise(rendered)
    )
    if changed != len(rendered) - 1:
        raise RuntimeError(
            f"GIF contains {len(rendered) - 1 - changed} duplicate frame transitions"
        )

    output = Path(ARGS.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(ARGS.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    durations = [round(1000.0 / ARGS.fps)] * len(rendered)
    durations[-1] = 1300
    gif_temporary = _temporary_path(output)
    report_temporary = _temporary_path(report_path)
    try:
        rendered[0].save(
            gif_temporary,
            save_all=True,
            append_images=rendered[1:],
            duration=durations,
            loop=0,
            optimize=True,
            disposal=2,
        )
        validate_gif(gif_temporary)
        with report_temporary.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        # Both artifacts are complete before either published file changes.
        os.replace(report_temporary, report_path)
        os.replace(gif_temporary, output)
    finally:
        gif_temporary.unlink(missing_ok=True)
        report_temporary.unlink(missing_ok=True)

    print(
        f"squash_demo_gif={output} frames={len(rendered)} "
        f"score={report['score']['A']}-{report['score']['B']}",
        flush=True,
    )
    print(f"squash_demo_report={report_path}", flush=True)
    return 0


def _temporary_path(target: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}-",
        suffix=target.suffix,
        dir=target.parent,
        delete=False,
    ) as stream:
        return Path(stream.name)


def _annotate_frame(image, elapsed: float, events, image_draw, image_font):
    labels = {
        "serve": "A SERVES",
        "front_wall": "FRONT WALL",
        "floor_bounce": "FLOOR BOUNCE",
        "racket_hit": "B RETURNS",
        "point": "POINT TO B",
    }
    active = events[0]
    for event in events:
        if float(event["time_s"]) <= elapsed + 0.04:
            active = event
        else:
            break

    def font(size: int):
        candidates = (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        )
        for candidate in candidates:
            if Path(candidate).is_file():
                return image_font.truetype(candidate, size=size)
        return image_font.load_default()

    annotated = image.copy()
    draw = image_draw.Draw(annotated, "RGBA")
    width, height = annotated.size
    draw.rounded_rectangle((15, 14, width - 15, 74), radius=12, fill=(10, 16, 27, 218))
    draw.text(
        (30, 25),
        "SQUASH  |  ONE COMPLETE POINT",
        font=font(20),
        fill=(245, 247, 250, 255),
    )
    score = "0  -  1" if active["kind"] == "point" else "0  -  0"
    draw.text((width - 137, 22), score, font=font(25), fill=(255, 205, 35, 255))

    label = labels[active["kind"]]
    if active["kind"] == "floor_bounce":
        label += f"  {active.get('detail', '')}"
    footer_top = height - 58
    draw.rounded_rectangle(
        (15, footer_top, width - 15, height - 15),
        radius=10,
        fill=(10, 16, 27, 218),
    )
    draw.text((30, footer_top + 10), label, font=font(18), fill=(255, 205, 35, 255))
    draw.text(
        (width - 180, footer_top + 12),
        f"{elapsed:04.1f} s",
        font=font(16),
        fill=(235, 238, 242, 255),
    )
    draw.text(
        (width - 108, height - 27),
        "ball marker 2.75x",
        font=font(7),
        fill=(190, 197, 207, 255),
    )
    return annotated


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        if ARGS.headless:
            os._exit(1)
        APP.close(wait_for_replicator=False)
        raise

    if ARGS.headless:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    APP.close(wait_for_replicator=False)
    raise SystemExit(exit_code)
