"""Command line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .scene import SCENES
from .simulation import Simulation


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="multisport-sim",
        description="Run regulation-size tennis, table-tennis, football, badminton and basketball scenes.",
    )
    result.add_argument("--scene", choices=SCENES, default="campus")
    result.add_argument("--headless", action="store_true", help="simulate without opening a window")
    result.add_argument("--duration", type=float, default=None, help="wall/simulation seconds to run")
    result.add_argument("--no-launch", action="store_true", help="drop balls without initial launch velocity")
    result.add_argument(
        "--wind",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="world-frame wind velocity in m/s",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    simulation = Simulation(args.scene, tuple(args.wind))
    if args.headless:
        duration = 5.0 if args.duration is None else max(0.0, args.duration)
        stats = simulation.run_headless(duration, launch=not args.no_launch)
        print(
            f"scene={args.scene} steps={stats.steps} "
            f"simulated={stats.simulated_seconds:.3f}s max_height={stats.max_ball_height:.3f}m"
        )
    else:
        simulation.run_viewer(args.duration, launch=not args.no_launch)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

