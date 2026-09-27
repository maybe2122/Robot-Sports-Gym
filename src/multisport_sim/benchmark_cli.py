"""CLI for the experimental table-tennis single-shot benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from math import isfinite
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
    TABLE_TENNIS_RETURN_STANDING_G1_V2,
    TABLE_TENNIS_RETURN_V0,
    TENNIS_RETURN_V0,
)
from .benchmark.vision import RGBDBallTracker, VisionTrackBackend, vision_sensors_for

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
    "g1-standing": TABLE_TENNIS_RETURN_STANDING_G1_V2,
}
# Sensor suites are bound to each robot's own paddle and joints.
VISION_ROBOTS = ("panda", "g1", "g1-standing")
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
    result.add_argument("--policy", help="custom module:policy or module:factory")
    result.add_argument("--rate-margin", type=_positive_float, help="joint setpoint speed fraction")
    result.add_argument("--blade-tilt-deg", type=float, help="paddle tilt in degrees")
    result.add_argument("--swing-lead-s", type=_positive_float, help="begin forward swing this early")
    result.add_argument("--viewer", action="store_true", help="show the scored robot rollout")
    result.add_argument("--video", type=Path, help="record the scored robot rollout as GIF")
    result.add_argument("--sensor-config", type=Path, help="JSON camera specifications")
    result.add_argument("--config", type=Path, help="JSON defaults; CLI values override")
    result.add_argument("--perception", choices=("stereo", "rgbd", "depth"), default="stereo")
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
            "'g1' uses a fixed pelvis; 'g1-standing' uses a free pelvis and ankle balance"
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


def _robot_controller(name: str, *, control_dt: float, robot: str = "panda", **settings):
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
    swing = SWING_ROBOTS["g1" if robot == "g1-standing" else robot]
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
        return VisionInterceptController(control_dt=control_dt, robot=swing, **settings)
    return ScriptedInterceptController(control_dt=control_dt, robot=swing, **settings)


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
        "base": describe["robot"].get("base", "fixed arm"),
        **({"balance_controller": describe["robot"]["balance_controller"]}
           if "balance_controller" in describe["robot"] else {}),
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
    from .benchmark.backends.mujoco_standing import MujocoStandingG1TableTennisBackend
    from .benchmark.robot import WorkspaceBox

    backends = {
        "panda": MujocoPandaTableTennisBackend,
        "g1": MujocoG1TableTennisBackend,
        "g1-standing": MujocoStandingG1TableTennisBackend,
    }
    timeout_s = float(source_bank.manifest["episode"]["timeout_s"])
    task = replace(
        ROBOT_TASKS[args.robot],
        split=source_bank.split,
        control_hz=args.control_hz,
        timeout_s=timeout_s,
    )
    vision_track = getattr(args, "track", "state") == "vision"
    sensors = vision_sensors_for(
        args.robot, depth=getattr(args, "perception", "stereo") in {"rgbd", "depth"}
    ) if vision_track else ()
    if getattr(args, "sensor_config", None) is not None:
        from .benchmark.sensors import CameraSpec
        camera_config = json.loads(args.sensor_config.read_text())
        if not isinstance(camera_config, list):
            raise ValueError("sensor config must be a list of camera specifications")
        try:
            cameras = tuple(CameraSpec(**item) for item in camera_config)
        except TypeError as exc:
            raise ValueError(f"invalid camera specification: {exc}") from exc
        expected = 1 if getattr(args, "perception", "stereo") in {"rgbd", "depth"} else 2
        if len(cameras) != expected:
            raise ValueError(f"perception requires {expected} cameras")
        if any(camera.mount != "world" for camera in cameras):
            raise ValueError("reference trackers require world-mounted cameras")
        sensors = (*cameras, *(s for s in sensors if not isinstance(s, CameraSpec)))
    backend = backends[args.robot](
        workspace=WorkspaceBox(
            task.workspace.position_low, task.workspace.position_high
        ),
        sensors=sensors,
    )
    controller_settings = {
        key: getattr(args, key) for key in ("rate_margin", "blade_tilt_deg", "swing_lead_s")
        if getattr(args, key, None) is not None
    }
    controller = _robot_controller(
        args.controller,
        control_dt=task.control_dt(backend.timestep),
        robot=args.robot,
        **controller_settings,
    )
    if vision_track and getattr(args, "perception", "stereo") == "rgbd":
        controller.tracker = RGBDBallTracker(camera=sensors[0])
        controller.controller_id = f"rgbd-{args.robot}-intercept-v1"
    elif vision_track and getattr(args, "perception", "stereo") == "depth":
        from .benchmark.depth_vision import DepthBallTracker
        background = backend.read_sensors().camera(sensors[0].name).depth
        controller.tracker = DepthBallTracker(camera=sensors[0], background=background)
        controller.controller_id = f"depth-{args.robot}-intercept-v1"
    elif vision_track and getattr(args, "sensor_config", None) is not None:
        from .benchmark.vision import StereoBallTracker
        controller.tracker = StereoBallTracker(left=sensors[0], right=sensors[1])
    if getattr(args, "policy", None) is not None:
        from .benchmark.policy_eval import PolicyController, load_policy
        controller = PolicyController(load_policy(args.policy), policy_id=args.policy,
                                      track=args.track, task=task)
    if controller_settings:
        controller.controller_id = f"configured-{controller.controller_id}"
    # The wrapper is what enforces the track: on the vision track the object the
    # controller receives has no ball attribute at all.
    runner_backend = VisionTrackBackend(
        backend, depth_only=getattr(args, "perception", "stereo") == "depth"
    ) if vision_track else backend
    perturbations = for_level(source_bank.manifest, args.level)
    if not perturbations.is_identity:
        runner_backend = PerturbedBackend(runner_backend, perturbations)
    from .benchmark.visualization import RolloutDisplay

    try:
        with RolloutDisplay(backend, viewer=getattr(args, "viewer", False),
                            video=getattr(args, "video", None)) as display:
            output = run_shots(
                runner_backend,
                controller,
                shots,
                config=RunConfig.from_task_config(task, seed=args.seed),
                on_step=display if display.viewer_enabled or display.video else None,
            )
    finally:
        if backend.sensors is not None:
            backend.sensors.close()
    return (
        backend,
        controller,
        task,
        output,
        robot_metadata(backend, task),
        {
            "id": controller.controller_id,
            "parameters": controller_settings,
            "observation": ("depth-only+proprioception" if vision_track and
                            getattr(args, "perception", "stereo") == "depth" else
                            "rgbd+proprioception" if vision_track and
                            getattr(args, "perception", "stereo") == "rgbd" else
                            _observation_label(args.controller, vision_track)),
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
    if getattr(args, "policy", None) is not None:
        if args.robot not in ROBOT_TASKS:
            raise ValueError("--policy requires a robot")
        if any(getattr(args, key, None) is not None for key in
               ("rate_margin", "blade_tilt_deg", "swing_lead_s")):
            raise ValueError("custom policies cannot use built-in swing parameters")
    for key in ("rate_margin", "blade_tilt_deg", "swing_lead_s"):
        value = getattr(args, key, None)
        if value is not None:
            if args.controller not in {"intercept", "vision"}:
                raise ValueError(f"{key} requires an intercept or vision controller")
            if not isfinite(value):
                raise ValueError(f"{key} must be finite")
    if getattr(args, "rate_margin", None) is not None and args.rate_margin > 1.:
        raise ValueError("rate_margin must be at most 1")
    if args.robot == "none" and (getattr(args, "viewer", False) or
                                 getattr(args, "video", None) is not None):
        raise ValueError("--viewer and --video require a robot task")
    if track != "vision" and (getattr(args, "sensor_config", None) is not None or
                               getattr(args, "perception", "stereo") != "stereo"):
        raise ValueError("camera configuration requires --track vision")
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
                "safety_details": [
                    [violation.to_dict() for violation in episode]
                    for episode in output.safety_details
                ],
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
                "perception": getattr(args, "perception", "stereo"),
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
    cli = parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    preliminary, _ = cli.parse_known_args(raw)
    try:
        if preliminary.config is not None:
            config = json.loads(preliminary.config.read_text())
            if not isinstance(config, dict):
                raise ValueError("config must be a JSON object")
            actions = {a.dest: a for a in cli._actions if a.dest not in {"help", "config"}}
            configured = []
            for key, value in config.items():
                if key not in actions:
                    raise ValueError(f"unknown config key: {key}")
                action = actions[key]
                if isinstance(action, argparse._StoreTrueAction):
                    if not isinstance(value, bool):
                        raise ValueError(f"{key} must be boolean")
                    if value:
                        configured.append(action.option_strings[0])
                else:
                    if value is None or isinstance(value, (dict, list, bool)):
                        raise ValueError(f"{key} must be a scalar value")
                    configured.extend([action.option_strings[0], str(value)])
            raw = configured + raw
        arguments = cli.parse_args(raw)
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
            metrics = report["metrics"]
            print(f"episodes={report['episodes']} hit={metrics['hit_rate']:.0%} "
                  f"valid_return={metrics['valid_return_rate']:.0%} "
                  f"safety_violation={metrics['safety_violation_rate']:.0%}")
        if arguments.markdown is not None:
            print(f"markdown={write_report(report, arguments.markdown)}")
        if arguments.report is None and arguments.markdown is None:
            print(report_markdown(report), end="")
    except (MetricsError, ShotBankError, OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"multisport-benchmark: error: {exc}", file=sys.stderr)
        return 2

    assessment = report.get("assessment")
    if arguments.require_pass and isinstance(assessment, dict) and not assessment.get("passed"):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
