#!/usr/bin/env python3
"""Measure how far the badminton fixture carries a serve at each racket speed.

The scripted serve controller turns a target distance into a racket speed with
a table, and this script produces that table.  Each speed is swung through a
shuttle released at the bank's nominal serving position, aimed down the
diagonal, and the horizontal distance from the strike point to the first
landing is recorded.  Nothing is fitted: the controller interpolates between
measured rows.

    PYTHONPATH=src python scripts/calibrate_serve.py [--speeds 12 14 ... 30]
"""

from __future__ import annotations

import argparse
import json
import sys
from math import hypot

from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
from multisport_sim.benchmark.controllers import ScriptedServeController
from multisport_sim.benchmark.task_config import BADMINTON_SERVE_V0
from multisport_sim.benchmark.types import ShotSpec

RELEASE = (-2.6, -0.8, 1.05)
TARGET = (5.0, 1.3)


def carry(
    speed: float, *, elevation: float = 45.0, release=RELEASE, target=TARGET
) -> dict[str, float] | None:
    task = BADMINTON_SERVE_V0
    backend = MujocoShotBackend(sport="badminton")
    decimation = task.decimation(backend.timestep)
    controller = ScriptedServeController(
        speed_mps=speed,
        elevation_degrees=elevation,
        control_dt=decimation * backend.timestep,
    )
    shot = ShotSpec(
        "calibration", "badminton", "L2", release, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        target={"center_xy": list(target), "radius_m": 0.5},
    )
    backend.reset()
    backend.launch_ball(shot)
    controller.reset(shot)
    strike = None
    for step in range(task.max_physics_steps(backend.timestep)):
        if step % decimation == 0:
            backend.apply_action(controller.act(backend.observe()))
        backend.step()
        contacts = backend.semantic_contacts()
        ball = backend.get_ball_state()
        if strike is None and any("robot_racket" in c.pair for c in contacts):
            strike = ball.position
        if strike is not None and any(("table" in c.pair or "floor" in c.pair) for c in contacts):
            return {
                "speed_mps": speed,
                "carry_m": hypot(ball.position[0] - strike[0], ball.position[1] - strike[1]),
                "landing_x_m": ball.position[0],
                "flight_s": backend.time,
            }
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--speeds", type=float, nargs="+",
        default=[8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0],
    )
    parser.add_argument("--elevation", type=float, default=45.0)
    args = parser.parse_args(argv)
    rows = []
    for speed in args.speeds:
        row = carry(speed, elevation=args.elevation)
        if row is None:
            print(f"{speed:5.1f} m/s: no landing", file=sys.stderr)
            continue
        rows.append(row)
        print(
            f"{speed:5.1f} m/s: carry {row['carry_m']:.2f} m, lands x={row['landing_x_m']:.2f}, "
            f"flight {row['flight_s']:.2f} s",
            file=sys.stderr,
        )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
