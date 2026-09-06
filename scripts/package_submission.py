#!/usr/bin/env python3
"""Build the reproducible result package described in BENCHMARK_SPEC section 9.

A score in a terminal is not a result.  A result is something another person can
check: the exact commit, the exact shot bank, the exact policy, the environment
it ran in, the raw per-episode outcomes rather than a summary, and video of the
failures as well as the successes.  This command produces that package::

    submission/
      manifest.json     commit, task and asset versions, backend, hardware, track
      config/           the full task configuration and the run arguments
      metrics.json      every episode's raw result, per level, plus the summaries
      policy/           the weights (content-hashed) or how to obtain them
      videos/           fixed successes *and* failures, chosen deterministically
      environment.txt   Python, platform, simulator and dependency versions

Usage::

    python scripts/package_submission.py --policy my_pkg:load_policy \
        --policy-id my-sac-v3 --split test --out submission/

Videos need Pillow (``uv pip install pillow``).  Without it the package is
still written, and the manifest records that the videos are missing rather than
quietly shipping a package that looks complete.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import mujoco

from multisport_sim import __version__
from multisport_sim.benchmark.policy_eval import (
    TRACKS,
    PolicyController,
    cross_level_robustness_gap,
    default_observation_label,
    evaluate_levels,
    load_policy,
)
from multisport_sim.benchmark.shot_bank import VALID_LEVELS, ShotBank
from multisport_sim.benchmark.task_config import TABLE_TENNIS_RETURN_PANDA_V1
from multisport_sim.benchmark.vision import (
    LEFT_CAMERA,
    TABLE_TENNIS_VISION_SENSORS,
    VisionTrackBackend,
)

VIDEO_CAMERA = LEFT_CAMERA
VIDEO_FPS = 30


def _git(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *arguments), capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _repository_state() -> dict[str, object]:
    """The commit a reviewer would check out, and whether it is the whole story."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the commit alone does not reproduce this result.
        # Saying so is the point; hiding it would make the manifest a lie.
        "dirty": bool(status) if status is not None else None,
        "uncommitted_files": len(status.splitlines()) if status else 0,
    }


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment_text() -> str:
    from importlib.metadata import distributions

    installed = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in distributions()
        if dist.metadata["Name"]
    )
    lines = [
        f"python={platform.python_version()}",
        f"implementation={platform.python_implementation()}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"processor={platform.processor()}",
        f"mujoco={mujoco.__version__}",
        f"multisport_sim={__version__}",
        "",
        "# installed distributions",
        *installed,
    ]
    return "\n".join(lines) + "\n"


def _write_policy(destination: Path, spec: str, files: list[Path]) -> dict[str, object]:
    """Copy the weights, hashing every file; record how to get them if absent."""
    destination.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for path in files:
        if not path.is_file():
            raise SystemExit(f"policy file does not exist: {path}")
        shutil.copy2(path, destination / path.name)
        hashes[path.name] = _hash_file(path)
    (destination / "hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not files:
        (destination / "SOURCE.md").write_text(
            "# Policy source\n\n"
            f"This submission was scored by loading `{spec}` from the evaluating\n"
            "environment.  No weight files were supplied to the packager, so this\n"
            "package does **not** contain the policy.  Add them with\n"
            "`--policy-file <path>` (repeatable) for the result to be reproducible\n"
            "by anyone else.\n",
            encoding="utf-8",
        )
    return {
        "spec": spec,
        "files": sorted(hashes),
        "sha256": hashes,
        "complete": bool(files),
    }


def _select_episodes(reports: dict, count: int) -> tuple[list[dict], list[dict]]:
    """Pick successes and failures deterministically, by shot id.

    "Fixed samples, not only the successes" is a rule about selection, not about
    intent: sorting by shot id and taking the first few of each removes the
    chance to choose flattering ones.
    """
    episodes = [
        {**record, "level": level}
        for level, report in sorted(reports.items())
        for record in report["results"]
    ]
    episodes.sort(key=lambda record: record["shot_id"])
    successes = [record for record in episodes if record["valid_return"]]
    failures = [record for record in episodes if not record["valid_return"]]
    return successes[:count], failures[:count]


def _record_videos(
    policy_spec: str,
    episodes: list[dict],
    *,
    destination: Path,
    split: str,
    track: str,
    policy_id: str,
    seed: int,
    bank_resource: str,
) -> dict[str, object]:
    """Re-run the chosen shots with a camera attached and save one GIF each."""
    try:
        from PIL import Image
    except ImportError:
        return {
            "status": "skipped",
            "reason": "Pillow is not installed; run `uv pip install pillow`",
            "files": [],
        }
    if not episodes:
        return {"status": "empty", "reason": "no episodes selected", "files": []}

    from multisport_sim.benchmark.backends.mujoco_robot import (
        MujocoPandaTableTennisBackend,
    )
    from multisport_sim.benchmark.robot import WorkspaceBox
    from multisport_sim.benchmark.rules.table_tennis import TableTennisReturnJudge

    task = TABLE_TENNIS_RETURN_PANDA_V1
    sensors = TABLE_TENNIS_VISION_SENSORS if track == "vision" else (VIDEO_CAMERA,)
    backend = MujocoPandaTableTennisBackend(
        workspace=WorkspaceBox(task.workspace.position_low, task.workspace.position_high),
        sensors=sensors,
    )
    runner_backend = VisionTrackBackend(backend) if track == "vision" else backend
    bank = ShotBank.from_resource(split=split, task=bank_resource)
    decimation = task.decimation(backend.timestep)
    frame_every = max(1, round(1.0 / (VIDEO_FPS * backend.timestep)))

    destination.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for record in episodes:
        shot = bank.get(record["shot_id"])
        controller = PolicyController(
            load_policy(policy_spec), policy_id=policy_id, track=track
        )
        backend.reset()
        controller.reset(shot, seed=seed)
        backend.launch_ball(shot)
        judge = TableTennisReturnJudge(timeout_s=task.timeout_s, table_spec=task.table)
        judge.reset(shot)

        frames = []
        for step in range(task.max_physics_steps(backend.timestep)):
            if step % decimation == 0:
                backend.apply_action(controller.act(runner_backend.observe()))
            backend.step()
            if step % frame_every == 0:
                frames.append(
                    Image.fromarray(backend.read_sensors().camera(VIDEO_CAMERA.name).rgb)
                )
            judge.update(
                time_s=backend.time,
                ball=backend.get_ball_state(),
                contacts=backend.semantic_contacts(),
            )
            if judge.done:
                break

        outcome = "success" if record["valid_return"] else "failure"
        name = f"{record['level']}-{record['shot_id']}-{outcome}.gif"
        frames[0].save(
            destination / name,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / VIDEO_FPS),
            loop=0,
        )
        written.append(name)

    if backend.sensors is not None:
        backend.sensors.close()
    return {
        "status": "written",
        "camera": VIDEO_CAMERA.to_dict(),
        "fps": VIDEO_FPS,
        "selection": "sorted by shot id; the first N successes and the first N failures",
        "files": sorted(written),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument("--policy", required=True, help="'module:attribute' of the policy")
    result.add_argument("--policy-id", help="name recorded throughout; defaults to --policy")
    result.add_argument(
        "--policy-file",
        type=Path,
        action="append",
        default=[],
        help="weight file to copy into policy/ (repeatable); hashed in the manifest",
    )
    result.add_argument("--split", choices=("train", "dev", "test"), default="test")
    result.add_argument(
        "--bank",
        default=TABLE_TENNIS_RETURN_PANDA_V1.bank_resource,
        help=(
            "packaged shot bank; defaults to the one the task is scored on, so a "
            "packaged result is comparable to the published baselines"
        ),
    )
    result.add_argument("--track", choices=TRACKS, default="state")
    result.add_argument("--levels", default="all")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--out", type=Path, default=Path("submission"))
    result.add_argument(
        "--videos",
        type=int,
        default=2,
        help="how many successes and how many failures to record (0 disables)",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    levels = (
        list(VALID_LEVELS)
        if args.levels == "all"
        else [item.strip() for item in args.levels.split(",") if item.strip()]
    )
    policy_id = args.policy_id or args.policy
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    reports = evaluate_levels(
        load_policy(args.policy),
        levels=levels,
        split=args.split,
        seed=args.seed,
        policy_id=policy_id,
        track=args.track,
        bank=args.bank,
        out=out / "reports",
    )
    first = reports[levels[0]]

    (out / "config").mkdir(parents=True, exist_ok=True)
    (out / "config" / "task_config.json").write_text(
        json.dumps(first["task_config"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "config" / "run.json").write_text(
        json.dumps(
            {
                "policy": args.policy,
                "policy_id": policy_id,
                "split": args.split,
                "track": args.track,
                "levels": levels,
                "seed": args.seed,
                "observation": default_observation_label(args.track),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = {
        "task": first["task"],
        "split": args.split,
        "track": args.track,
        "levels": {
            level: {
                "assessment": report["assessment"],
                "metrics": report["metrics"],
                "confidence_intervals_95": report["confidence_intervals_95"],
                "counts": report["counts"],
                "robot_metrics": report["robot_metrics"],
                "contact_error": report["contact_error"],
                "inference_latency_ms": report["inference_latency_ms"],
                "buckets": report["buckets"],
                "failures": report["failures"],
                "results": report["results"],
            }
            for level, report in reports.items()
        },
        "robustness_gap": cross_level_robustness_gap(reports),
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    policy_block = _write_policy(out / "policy", args.policy, list(args.policy_file))
    (out / "environment.txt").write_text(_environment_text(), encoding="utf-8")

    successes, failures = _select_episodes(reports, args.videos) if args.videos else ([], [])
    videos = (
        _record_videos(
            args.policy,
            successes + failures,
            destination=out / "videos",
            split=args.split,
            track=args.track,
            policy_id=policy_id,
            seed=args.seed,
            bank_resource=args.bank,
        )
        if args.videos
        else {"status": "disabled", "files": []}
    )

    reproduce = (
        f"python scripts/package_submission.py --policy {args.policy} "
        f"--policy-id {policy_id} --split {args.split} --track {args.track} "
        f"--levels {args.levels} --seed {args.seed} --bank {args.bank} --out {out}"
    )
    manifest = {
        "schema": "multisport-submission-v0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": first["task"],
        "benchmark": first["benchmark"],
        "track": first["track"],
        "policy": {**first["policy"], **policy_block},
        "robot": first["robot"],
        "backend": first["backend"],
        "shot_bank": first["shot_bank"],
        "software": first["software"],
        "repository": _repository_state(),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "videos": videos,
        # The manifest states the bank's own verdict rather than the packager's:
        # a package built on an experimental fixture is not a leaderboard entry
        # however complete the package is.
        "leaderboard_eligible": first["leaderboard_eligible"],
        "reproduce_command": reproduce,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"submission written to {out}")
    print(f"  levels        {', '.join(levels)} ({args.split}, {args.track} track)")
    print(f"  policy files  {len(policy_block['files'])}")
    print(f"  videos        {videos['status']} ({len(videos['files'])})")
    if not policy_block["complete"]:
        print("  warning: no weights included; see policy/SOURCE.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
