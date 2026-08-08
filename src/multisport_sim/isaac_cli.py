"""Isaac Sim command line entry point.

Only AppLauncher is imported before SimulationApp starts. All Omniverse-dependent
imports live in ``main`` after launch.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Sequence

from .specs import SCENES


def parser() -> argparse.ArgumentParser:
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        raise SystemExit(
            "multisport-isaac requires an Isaac Sim + Isaac Lab Python environment; "
            "see docs/ISAAC_SIM.md"
        ) from None
    result = argparse.ArgumentParser(
        prog="multisport-isaac",
        description="Run the five regulation multi-sport scenes in Isaac Sim/PhysX.",
    )
    result.add_argument("--scene", choices=SCENES, default="campus")
    result.add_argument("--duration", type=float, default=0.0, help="simulation seconds; 0 runs until closed")
    result.add_argument("--no-launch", action="store_true", help="drop balls without launch velocity")
    result.add_argument(
        "--drop-test",
        action="store_true",
        help="run the regulation first-rebound test (requires a single-sport scene)",
    )
    result.add_argument(
        "--evaluate",
        action="store_true",
        help="score a single-sport rebound against the shared fidelity benchmark",
    )
    result.add_argument(
        "--report",
        default=None,
        help="with --evaluate, write a JSON or Markdown report based on the suffix",
    )
    result.add_argument("--export-usd", default=None, help="export the generated stage to this USD path")
    result.add_argument(
        "--wind",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="world-frame wind velocity in m/s",
    )
    AppLauncher.add_app_launcher_args(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from isaaclab.app import AppLauncher

    if args.headless and args.duration <= 0.0:
        args.duration = 5.0
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        from .isaac_backend import IsaacSportsSimulation

        simulation = IsaacSportsSimulation(
            args.scene,
            device=args.device,
            wind=tuple(args.wind),
        )
        if args.export_usd:
            output = simulation.export_usd(args.export_usd)
            print(f"exported_usd={output}", flush=True)
        if args.drop_test or args.evaluate:
            if args.scene == "campus":
                raise ValueError("--drop-test/--evaluate requires a single-sport --scene")
            from .specs import Sport

            sport = Sport(args.scene)
            rebound = simulation.regulation_drop_test(sport, render=not args.headless)
            if args.evaluate:
                from .evaluation import build_report, report_markdown, score_bounce, write_report

                report = build_report(
                    "isaacsim",
                    [score_bounce(sport, rebound)],
                    metadata={"timestep_seconds": simulation.dt, "solver": "PhysX TGS"},
                )
                if args.report:
                    output = write_report(report, args.report)
                    print(f"fidelity_report={output}", flush=True)
                else:
                    print(report_markdown(report), flush=True)
                if report["summary"]["failed"]:
                    raise RuntimeError(f"{sport.value} fidelity benchmark failed")
            else:
                print(
                    f"backend=isaacsim scene={args.scene} drop_test_rebound={rebound:.3f}m",
                    flush=True,
                )
        else:
            steps = simulation.run(
                simulation_app,
                args.duration,
                launch=not args.no_launch,
                render=not args.headless,
            )
            print(
                f"backend=isaacsim scene={args.scene} steps={steps} "
                f"simulated={steps * simulation.dt:.3f}s balls={len(simulation.balls)} "
                f"state={simulation.state_summary()}",
                flush=True,
            )
    # Preserve a useful traceback before the headless Kit fast-exit.
    except BaseException:
        if args.headless:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        simulation_app.close(wait_for_replicator=False)
        raise

    if args.headless:
        # On some mixed AMD/NVIDIA workstations Isaac Kit can hang indefinitely
        # in Framework::unload_all_plugins even though simulation is complete.
        # All output/USD writes are flushed above, so terminate the CLI process
        # without running that unreliable plugin-unload path.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    else:
        # No Replicator pipeline is used. Skipping its completion wait avoids a
        # needless one-second delay when closing the interactive viewer.
        simulation_app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
