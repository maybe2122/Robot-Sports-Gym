"""CLI for the experimental table-tennis single-shot benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys

import mujoco

from . import __version__
from .benchmark.backends.mujoco import MujocoShotBackend
from .benchmark.controllers import NoOpController, ScriptedPaddleController
from .benchmark.metrics import MetricsError, build_benchmark_report
from .benchmark.reporting import report_markdown, write_report
from .benchmark.runner import RunConfig, run_shots
from .benchmark.shot_bank import ShotBank, ShotBankError, VALID_LEVELS
from .benchmark.task_config import TABLE_TENNIS_RETURN_V0


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="multisport-benchmark",
        description=(
            "Run the experimental table-tennis-return-v0 Shot Skill harness. "
            "The built-in mocap controller is a diagnostic fixture, not a robot submission."
        ),
    )
    result.add_argument(
        "--sport",
        choices=("table-tennis", "table_tennis"),
        default="table-tennis",
        help="v0 currently implements table tennis only",
    )
    result.add_argument("--task", choices=("return",), default="return")
    result.add_argument("--backend", choices=("mujoco",), default="mujoco")
    result.add_argument("--level", choices=VALID_LEVELS, default="L1")
    result.add_argument("--split", choices=("dev", "test"), default="dev")
    result.add_argument(
        "--shot-bank",
        type=Path,
        help="custom bank directory, manifest.json, or split JSONL; defaults to package data",
    )
    result.add_argument(
        "--episodes",
        type=_positive_integer,
        help="run the first N fixed shots at the selected level; default runs all once",
    )
    result.add_argument(
        "--seed",
        type=int,
        default=0,
        help="controller seed; shots are not resampled",
    )
    result.add_argument(
        "--controller",
        choices=("scripted", "noop"),
        default="scripted",
        help="built-in diagnostic controller",
    )
    result.add_argument(
        "--control-hz", type=_positive_float, default=TABLE_TENNIS_RETURN_V0.control_hz
    )
    result.add_argument("--report", type=Path, help="write the complete strict-JSON report")
    result.add_argument("--markdown", type=Path, help="write a human-readable summary")
    result.add_argument(
        "--require-pass",
        action="store_true",
        help="return exit status 1 when the selected level threshold is not met",
    )
    return result


def _load_bank(args: argparse.Namespace) -> tuple[ShotBank, ShotBank]:
    source = (
        ShotBank.from_path(args.shot_bank, split=args.split)
        if args.shot_bank is not None
        else ShotBank.from_resource(split=args.split)
    )
    selected = source.filter(level=args.level)
    if not selected:
        raise ShotBankError(f"split {args.split!r} contains no shots for {args.level}")
    if args.episodes is not None and args.episodes > len(selected):
        raise ShotBankError(
            f"requested {args.episodes} episodes, but {args.level}/{args.split} "
            f"contains only {len(selected)}"
        )
    return source, selected


def _controller(name: str) -> NoOpController | ScriptedPaddleController:
    return NoOpController() if name == "noop" else ScriptedPaddleController()


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Run a parsed request and return its complete report."""
    source_bank, level_bank = _load_bank(args)
    shots = tuple(level_bank)
    if args.episodes is not None:
        shots = shots[: args.episodes]

    controller = _controller(args.controller)
    backend = MujocoShotBackend()
    # The frozen shot bank owns the episode timeout; everything else the judge
    # and a future Isaac run must agree on comes from the shared task config.
    timeout_s = float(source_bank.manifest["episode"]["timeout_s"])
    task = replace(
        TABLE_TENNIS_RETURN_V0,
        split=source_bank.split,
        control_hz=args.control_hz,
        timeout_s=timeout_s,
    )
    output = run_shots(
        backend,
        controller,
        shots,
        config=RunConfig.from_task_config(task, seed=args.seed),
    )

    fixture = args.controller == "scripted"
    report = build_benchmark_report(
        output.results,
        bank=level_bank,
        backend={
            "name": "mujoco",
            "version": mujoco.__version__,
            "physics_timestep_s": backend.timestep,
            "wind_mps": list(backend.aerodynamics.atmosphere.wind),
        },
        robot={
            "id": "scripted-mocap-paddle-fixture-v0" if fixture else "no-action-fixture-v0",
            "adapter": "mocap-paddle-fixture" if fixture else "none",
            "benchmark_eligible": False,
        },
        policy={
            "id": controller.controller_id,
            "observation": "privileged-ball-state" if fixture else "none",
            "benchmark_eligible": False,
        },
        seed=args.seed,
    )
    assessments = report.get("level_assessments", [])
    assessment = assessments[0] if len(assessments) == 1 else None
    report.update(
        {
            "schema": "multisport-shot-skill-report-v0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "software": {
                "multisport_sim": __version__,
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "level": args.level,
            "assessment": assessment,
            "task_config": task.to_dict(),
            "shot_bank_digest": source_bank.digest,
            "shot_bank_manifest_digest": source_bank.manifest_digest,
            "shot_bank": {
                "split": source_bank.split,
                "digest": source_bank.digest,
                "manifest_digest": source_bank.manifest_digest,
                "source_digest": source_bank.digest,
                "level_digest": level_bank.digest,
                "source": source_bank.source,
                "selected_count": len(shots),
                "selected_shot_ids": [shot.shot_id for shot in shots],
                "status": source_bank.manifest["status"],
                "leaderboard_eligible": source_bank.manifest["leaderboard_eligible"],
            },
            "execution": {
                "physics_steps": output.physics_steps,
                "control_hz": args.control_hz,
                "effective_control_hz": 1.0 / (output.control_decimation * backend.timestep),
                "control_decimation": output.control_decimation,
                "timeout_s": timeout_s,
                "episode_seeds": list(output.episode_seeds),
                "episode_seed_derivation": "sha256(multisport-shot-seed-v0:seed:index)-63bit",
                "shot_order": "fixed-file-order-without-replacement",
            },
        }
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.report is not None and arguments.report.suffix.lower() != ".json":
            raise ValueError("--report path must end in .json")
        if arguments.markdown is not None and arguments.markdown.suffix.lower() not in {
            ".md",
            ".markdown",
        }:
            raise ValueError("--markdown path must end in .md or .markdown")
        report = run_from_args(arguments)
        if arguments.report is not None:
            print(f"report={write_report(report, arguments.report)}")
        if arguments.markdown is not None:
            print(f"markdown={write_report(report, arguments.markdown)}")
        if arguments.report is None and arguments.markdown is None:
            print(report_markdown(report), end="")
    except (MetricsError, ShotBankError, OSError, RuntimeError, ValueError) as exc:
        print(f"multisport-benchmark: error: {exc}", file=sys.stderr)
        return 2

    assessment = report.get("assessment")
    if arguments.require_pass and isinstance(assessment, dict) and not assessment.get("passed"):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
