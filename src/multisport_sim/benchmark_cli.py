"""CLI for the experimental table-tennis single-shot benchmark."""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import mujoco

from . import __version__
from .benchmark.backends.mujoco import MujocoShotBackend
from .benchmark.controllers import (
    NoOpController,
    ScriptedPaddleController,
    scripted_controller_for,
)
from .benchmark.metrics import MetricsError, build_benchmark_report
from .benchmark.perturbations import PerturbedBackend, for_level
from .benchmark.reporting import report_markdown, write_report
from .benchmark.runner import RunConfig, run_shots
from .benchmark.shot_bank import VALID_LEVELS, ShotBank, ShotBankError
from .benchmark.task_config import (
    TABLE_TENNIS_RETURN_G1_V1,
    TABLE_TENNIS_RETURN_PANDA_V1,
    TABLE_TENNIS_RETURN_V0,
    TENNIS_RETURN_V0,
)
from .benchmark.vision import TABLE_TENNIS_VISION_SENSORS, VisionTrackBackend

MOCAP_TASKS = {
    "table_tennis": TABLE_TENNIS_RETURN_V0,
    "tennis": TENNIS_RETURN_V0,
}
DEFAULT_BANKS = {
    "table_tennis": "table_tennis/return-v0",
    "tennis": "tennis/return-v0",
}
MOCAP_CONTROLLERS = ("scripted", "noop")
ROBOT_CONTROLLERS = ("intercept", "random", "hold")
# Each embodiment names its task config and the swing binding its scripted
# baseline uses.  Adding a robot to the CLI is adding a row here.
ROBOT_TASKS = {
    "panda": TABLE_TENNIS_RETURN_PANDA_V1,
    "g1": TABLE_TENNIS_RETURN_G1_V1,
}
# The vision track is declared and measured for the Panda only; a track that
# has not been run is not one the CLI should let a user think exists.
VISION_ROBOTS = ("panda",)
# A vision controller may only run on the vision track; it has no way to read
# privileged ball state, and the track wrapper makes sure it cannot start.
VISION_CONTROLLERS = ("vision",)
TRACKS = ("state", "vision")


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
        choices=("table-tennis", "table_tennis", "tennis"),
        default="table-tennis",
        help="the sport to score; the robot task exists for table tennis only",
    )
    result.add_argument("--task", choices=("return",), default="return")
    result.add_argument("--backend", choices=("mujoco",), default="mujoco")
    result.add_argument("--level", choices=VALID_LEVELS, default="L1")
    result.add_argument(
        "--split",
        choices=("train", "dev", "test"),
        default="dev",
        help="'train' exists only in banks that publish it, such as return-v1",
    )
    result.add_argument(
        "--bank",
        default=None,
        help=(
            "packaged shot bank to score on; 'table_tennis/return-v1' is the "
            "statistically sufficient one (100 test episodes per level), while "
            "the v0 default stays frozen as the rule-engine fixture"
        ),
    )
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
        "--robot",
        choices=("none", *ROBOT_TASKS),
        default="none",
        help=(
            "'none' runs the v0 mocap fixture; 'panda' runs the embodied "
            "table-tennis-return-panda-v1 task on an actuated Franka Panda; "
            "'g1' runs table-tennis-return-g1-v1 on a fixed-base Unitree G1"
        ),
    )
    result.add_argument(
        "--track",
        choices=TRACKS,
        default="state",
        help=(
            "'state' hands the policy privileged ball state; 'vision' removes "
            "it and attaches the declared stereo camera suite instead "
            "(robot tasks only)"
        ),
    )
    result.add_argument(
        "--controller",
        choices=(*MOCAP_CONTROLLERS, *ROBOT_CONTROLLERS, *VISION_CONTROLLERS),
        default=None,
        help=(
            "built-in reference baseline; defaults to 'scripted' for the mocap "
            "fixture and 'intercept' for the robot. None of these is a "
            "submission: they bracket the score, they do not set it"
        ),
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
        else ShotBank.from_resource(split=args.split, task=_bank_for(args))
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


def _sport(args: argparse.Namespace) -> str:
    return str(getattr(args, "sport", "table-tennis")).replace("-", "_")


def _bank_for(args: argparse.Namespace) -> str:
    """The packaged bank to score on: the explicit one, or the sport's default."""
    explicit = getattr(args, "bank", None)
    if explicit:
        return explicit
    robot = getattr(args, "robot", "none")
    if robot in ROBOT_TASKS:
        return ROBOT_TASKS[robot].bank_resource
    return DEFAULT_BANKS[_sport(args)]


def _controller(name: str, sport: str) -> NoOpController | ScriptedPaddleController:
    return NoOpController() if name == "noop" else scripted_controller_for(sport)


def _default_controller(robot: str, track: str = "state") -> str:
    if robot not in ROBOT_TASKS:
        return "scripted"
    return "vision" if track == "vision" else "intercept"


def _robot_controller(name: str, *, control_dt: float, robot: str = "panda"):
    """Build one embodied baseline, importing the robot stack lazily.

    Every baseline is parameterised by the embodiment rather than duplicated per
    robot: the floor, the random baseline and the scripted swing are the same
    three ideas whichever arm executes them, and forking them per robot would
    make two robots' baselines quietly incomparable.
    """
    from .benchmark.robot_controllers import (
        SWING_ROBOTS,
        HoldPoseController,
        RandomJointController,
        ScriptedInterceptController,
        VisionInterceptController,
    )

    task = ROBOT_TASKS[robot]
    swing = SWING_ROBOTS[robot]
    if name == "hold":
        return HoldPoseController(swing.ready_qpos)
    if name == "random":
        low, high = task.action_bounds()
        return RandomJointController(
            low,
            high,
            control_dt=control_dt,
            ready_qpos=swing.ready_qpos,
            velocity_limit=swing.velocity_limit,
        )
    if name == "vision":
        return VisionInterceptController(control_dt=control_dt, robot=swing)
    return ScriptedInterceptController(control_dt=control_dt, robot=swing)


def robot_metadata(backend, task) -> dict[str, object]:
    """Describe the embodiment that produced a run, for the report envelope.

    Shared with out-of-tree scoring scripts so that a submission's report names
    the same asset, mount and safety envelope the CLI would have named.
    """
    describe = backend.describe()
    return {
        "id": task.robot_id,
        "adapter": f"{describe['backend']}-joint-position",
        "control_mode": task.joint_action.control_mode,
        "asset": describe["robot"]["asset"],
        "mount": describe["robot"]["mount"],
        "safety_limits": describe["robot"]["safety_limits"],
        # The shot bank is still an experimental development fixture
        # (manifest: leaderboard_eligible false), so no run on it is a
        # submission -- a reference baseline least of all, whatever it scores.
        "benchmark_eligible": False,
    }


def _robot_run(args: argparse.Namespace, source_bank: ShotBank, shots):
    """Run the embodied task and return everything the report layer needs."""
    from .benchmark.backends.mujoco_robot import (
        MujocoG1TableTennisBackend,
        MujocoPandaTableTennisBackend,
    )
    from .benchmark.robot import WorkspaceBox

    backends = {
        "panda": MujocoPandaTableTennisBackend,
        "g1": MujocoG1TableTennisBackend,
    }
    timeout_s = float(source_bank.manifest["episode"]["timeout_s"])
    task = replace(
        ROBOT_TASKS[args.robot],
        split=source_bank.split,
        control_hz=args.control_hz,
        timeout_s=timeout_s,
    )
    vision_track = getattr(args, "track", "state") == "vision"
    sensors = TABLE_TENNIS_VISION_SENSORS if vision_track else ()
    backend = backends[args.robot](
        workspace=WorkspaceBox(
            task.workspace.position_low, task.workspace.position_high
        ),
        sensors=sensors,
    )
    controller = _robot_controller(
        args.controller,
        control_dt=task.control_dt(backend.timestep),
        robot=args.robot,
    )
    # The wrapper is what enforces the track: on the vision track the object the
    # controller receives has no ball attribute at all.
    runner_backend = VisionTrackBackend(backend) if vision_track else backend
    perturbations = for_level(source_bank.manifest, args.level)
    if not perturbations.is_identity:
        runner_backend = PerturbedBackend(runner_backend, perturbations)
    output = run_shots(
        runner_backend,
        controller,
        shots,
        config=RunConfig.from_task_config(task, seed=args.seed),
    )
    return (
        backend,
        controller,
        task,
        output,
        robot_metadata(backend, task),
        {
            "id": controller.controller_id,
            "observation": _observation_label(args.controller, vision_track),
            "track": "vision" if vision_track else "state",
            "benchmark_eligible": False,
        },
    )


def _observation_label(controller: str, vision_track: bool) -> str:
    """Name what the controller was actually allowed to read."""
    if vision_track:
        return "stereo-vision+proprioception"
    if controller in {"hold", "random"}:
        return "none"
    return "privileged-ball-state"


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Run a parsed request and return its complete report."""
    if args.controller is None:
        args.controller = _default_controller(args.robot, getattr(args, "track", "state"))
    track = getattr(args, "track", "state")
    if args.robot != "none" and _sport(args) != "table_tennis":
        raise ValueError(
            f"--robot {args.robot!r} is only available for table tennis; the "
            f"tennis task currently has the mocap fixture only"
        )
    if track == "vision" and args.robot not in VISION_ROBOTS:
        raise ValueError(
            "--track vision is declared and measured for "
            f"{', '.join(VISION_ROBOTS)} only; --robot {args.robot!r} has no "
            "vision track yet"
        )
    if args.robot in ROBOT_TASKS:
        allowed = ROBOT_CONTROLLERS
        if args.robot in VISION_ROBOTS:
            allowed = (*allowed, *VISION_CONTROLLERS)
    else:
        allowed = MOCAP_CONTROLLERS
    if args.controller not in allowed:
        raise ValueError(
            f"--controller {args.controller!r} is not available for "
            f"--robot {args.robot!r}; choose one of: {', '.join(allowed)}"
        )
    if (args.controller in VISION_CONTROLLERS) != (track == "vision"):
        raise ValueError(
            f"--controller {args.controller!r} and --track {track!r} disagree; "
            "a vision controller runs only on the vision track, and the vision "
            "track has no privileged ball state for the others to read"
        )

    source_bank, level_bank = _load_bank(args)
    shots = tuple(level_bank)
    if args.episodes is not None:
        shots = shots[: args.episodes]

    if args.robot in ROBOT_TASKS:
        backend, controller, task, output, robot, policy = _robot_run(
            args, source_bank, shots
        )
        timeout_s = task.timeout_s
        return _assemble_report(
            args, source_bank, level_bank, shots, backend, task, output, robot, policy,
            timeout_s,
        )

    sport = _sport(args)
    controller = _controller(args.controller, sport)
    backend = MujocoShotBackend(sport=sport)
    # The frozen shot bank owns the episode timeout; everything else the judge
    # and a future Isaac run must agree on comes from the shared task config.
    timeout_s = float(source_bank.manifest["episode"]["timeout_s"])
    task = replace(
        MOCAP_TASKS[sport],
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
    return _assemble_report(
        args,
        source_bank,
        level_bank,
        shots,
        backend,
        task,
        output,
        {
            "id": "scripted-mocap-paddle-fixture-v0" if fixture else "no-action-fixture-v0",
            "adapter": "mocap-paddle-fixture" if fixture else "none",
            "benchmark_eligible": False,
        },
        {
            "id": controller.controller_id,
            "observation": "privileged-ball-state" if fixture else "none",
            "benchmark_eligible": False,
        },
        timeout_s,
    )


def _contact_error_block(output) -> dict[str, object]:
    """How far off the sweet spot each strike landed, and how fast the blade was.

    ``None`` entries are episodes with no blade contact at all.  They are kept
    rather than dropped so the list lines up with the episodes, and averaged
    over only the strikes that happened: a miss has no contact error, and
    counting it as zero would reward missing.
    """
    offsets = [value for value in output.contact_offset_m if value is not None]
    speeds = [value for value in output.contact_speed_mps if value is not None]
    return {
        "definition": (
            "distance from the blade centre to the first ball-blade contact, "
            "measured across the strike face"
        ),
        "offset_m": [
            None if value is None else round(value, 6) for value in output.contact_offset_m
        ],
        "blade_speed_mps": [
            None if value is None else round(value, 6) for value in output.contact_speed_mps
        ],
        "contacts": len(offsets),
        "mean_offset_m": sum(offsets) / len(offsets) if offsets else None,
        "max_offset_m": max(offsets) if offsets else None,
        "mean_blade_speed_mps": sum(speeds) / len(speeds) if speeds else None,
    }


def _latency_block(output) -> dict[str, object]:
    """Wall-clock cost of the policy's own decision, on this machine.

    It is reported beside the machine description rather than as a task metric:
    the same policy on different hardware is a different number, and comparing
    two submissions on it without comparing their machines is meaningless.
    """
    samples = sorted(output.inference_latency_ms)
    if not samples:
        return {"samples": 0}

    def percentile(fraction: float) -> float:
        index = min(len(samples) - 1, max(0, round(fraction * (len(samples) - 1))))
        return samples[index]

    return {
        "samples": len(samples),
        "mean": sum(samples) / len(samples),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": samples[-1],
        "measured_on": "the machine named in the report's software block",
    }


def _assemble_report(
    args, source_bank, level_bank, shots, backend, task, output, robot, policy, timeout_s
) -> dict[str, object]:
    """Build the report body shared by the mocap fixture and the robot task."""
    report = build_benchmark_report(
        output.results,
        bank=level_bank,
        backend={
            # The sport is already named by the task id; the backend field
            # names the backend, and changing it would move a published value.
            "name": backend.describe()["backend"] if args.robot in ROBOT_TASKS else "mujoco",
            "version": mujoco.__version__,
            "physics_timestep_s": backend.timestep,
            "wind_mps": list(backend.aerodynamics.atmosphere.wind),
        },
        robot=robot,
        policy=policy,
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
            # The shot bank's manifest names the *bank*, which both tasks share.
            # The report must name the task that produced the numbers.
            "task": task.task_id,
            "shot_bank_task": source_bank.manifest["task"],
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
            "robot_metrics": {
                # Empty for the mocap fixture: it has no safety envelope and no
                # actuators, so reporting zeros would claim checks that never ran.
                "safety_violations": list(output.safety_violations),
                "energy_joule": [round(value, 6) for value in output.energy_joule],
                "total_safety_violations": sum(output.safety_violations),
                "mean_energy_joule": (
                    sum(output.energy_joule) / len(output.energy_joule)
                    if output.energy_joule
                    else None
                ),
            },
            "contact_error": _contact_error_block(output),
            "inference_latency_ms": _latency_block(output),
            "track": {
                "id": getattr(args, "track", "state"),
                "privileged_ball_state": getattr(args, "track", "state") != "vision",
                "sensors": [
                    spec.to_dict() for spec in getattr(backend, "sensor_specs", ())
                ],
            },
            # What the level actually applied, not what it is named after.
            "perturbations": for_level(source_bank.manifest, args.level).to_dict(),
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
