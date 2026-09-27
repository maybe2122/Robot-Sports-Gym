#!/usr/bin/env python3
"""Same shots, same initial conditions, two physics backends: how far apart?

The benchmark claims one task on two simulators.  This script is the evidence
for (or against) that claim on the table-tennis fixture with the blade parked,
so only ball flight and ball-surface contact are compared:

    # 1. Isaac Lab rollout (Isaac Sim Python; AppLauncher starts first)
    PYTHONPATH=src $ISAAC_PYTHON scripts/backend_parity.py isaac \\
        --bank table_tennis/return-v1 --split dev --out generated/parity-isaac.json

    # 2. MuJoCo rollout of the identical shots (project venv)
    PYTHONPATH=src python scripts/backend_parity.py mujoco \\
        --bank table_tennis/return-v1 --split dev --out generated/parity-mujoco.json

    # 3. Compare
    PYTHONPATH=src python scripts/backend_parity.py compare \\
        generated/parity-mujoco.json generated/parity-isaac.json \\
        --report reports/table-tennis-backend-parity.json \\
        --markdown reports/table-tennis-backend-parity.md

Both rollouts sample the ball once per control step (5 ms at 200 Hz), after the
same number of 1 ms physics steps, and both are scored by the same judge.  The
comparison is split at the first surface bounce: before it, any divergence is
aerodynamics and integration; after it, it is the contact model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

SCHEMA = "multisport-backend-parity-v0"
PARKED_BLADE = (-1.58, 1.25, 1.0, 1.0, 0.0, 0.0, 0.0)
"""Where both fixtures park the blade: outside the table width, never touched."""


def _task(bank: str, split: str):
    from multisport_sim.benchmark.task_config import TABLE_TENNIS_RETURN_V0

    return replace(TABLE_TENNIS_RETURN_V0, bank_resource=bank, split=split)


def _shots(bank: str, split: str, limit: int | None):
    from multisport_sim.benchmark.shot_bank import ShotBank

    shots = list(ShotBank.from_resource(split=split, task=bank))
    return shots if limit is None else shots[:limit]


def _payload(backend: str, bank: str, split: str, episodes: list[dict[str, Any]], **extra):
    return {
        "schema": SCHEMA,
        "backend": backend,
        "bank": bank,
        "split": split,
        "episodes": episodes,
        **extra,
    }


def run_mujoco(args: argparse.Namespace) -> dict[str, Any]:
    import mujoco

    from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
    from multisport_sim.benchmark.controllers import PaddleCommand
    from multisport_sim.benchmark.rules.table_tennis import TableTennisReturnJudge

    task = _task(args.bank, args.split)
    backend = MujocoShotBackend()
    decimation = task.decimation(backend.timestep)
    parked = PaddleCommand(position=PARKED_BLADE[:3], quaternion=PARKED_BLADE[3:])
    episodes = []
    for shot in _shots(args.bank, args.split, args.limit):
        backend.reset()
        backend.apply_action(parked)
        backend.launch_ball(shot)
        judge = TableTennisReturnJudge(timeout_s=task.timeout_s, table_spec=task.table)
        judge.reset(shot)
        samples = []
        for step in range(task.max_physics_steps(backend.timestep)):
            backend.step()
            if not judge.done:
                judge.update(
                    time_s=backend.time,
                    ball=backend.get_ball_state(),
                    contacts=backend.semantic_contacts(),
                )
            if (step + 1) % decimation == 0:
                ball = backend.get_ball_state()
                samples.append([backend.time, *ball.position, *ball.linear_velocity])
            if judge.done and backend.time >= judge.result.episode_time_s + args.tail_s:
                break
        episodes.append(
            {"shot_id": shot.shot_id, "result": judge.result.to_dict(), "samples": samples}
        )
    return _payload(
        "mujoco",
        args.bank,
        args.split,
        episodes,
        physics_dt=backend.timestep,
        control_dt=decimation * backend.timestep,
        engine=f"mujoco {mujoco.__version__}",
    )


def run_isaac(args: argparse.Namespace) -> dict[str, Any]:
    from isaaclab.app import AppLauncher

    launcher_args = argparse.Namespace(headless=True, device=args.device)
    app = AppLauncher(launcher_args).app
    try:
        from importlib.metadata import version

        import torch

        from multisport_sim.benchmark.backends.isaac_lab import (
            PHYSICS_DT,
            IsaacTableTennisReturnEnv,
            make_env_cfg,
        )

        shots = _shots(args.bank, args.split, args.limit)
        task = _task(args.bank, args.split)
        # One environment per shot, all launched by the first reset.  A slot is
        # auto-reset the moment its judge rules, so Isaac samples stop at the
        # verdict; the comparison only uses samples both backends have.
        cfg = make_env_cfg(num_envs=len(shots), device=args.device, task=task)
        cfg.seed = 0
        env = IsaacTableTennisReturnEnv(cfg)
        env.shot_queue.extend(shots)
        env.reset()
        playing = env.current_shots
        order = {shot.shot_id: index for index, shot in enumerate(playing) if shot is not None}
        results: dict[str, dict[str, Any]] = {}
        samples: list[list[list[float]]] = [[] for _ in shots]
        finished_at: list[float | None] = [None] * len(shots)
        action = torch.tensor([PARKED_BLADE] * len(shots), device=env.device)
        control_dt = cfg.decimation * PHYSICS_DT
        steps = round((task.timeout_s + args.tail_s) / control_dt) + 1
        consumed = 0
        for step in range(steps):
            obs, _, terminated, truncated, _ = env.step(action)
            state = obs["policy"].cpu()
            time_s = (step + 1) * control_dt
            records = list(env.episode_records)
            for record in records[consumed:]:
                index = order.get(record["shot_id"])
                if index is not None and record["shot_id"] not in results:
                    results[record["shot_id"]] = record
                    finished_at[index] = time_s
            consumed = len(records)
            done = (terminated | truncated).cpu()
            for index in range(len(shots)):
                end = finished_at[index]
                if end is not None and bool(done[index]):
                    # The environment auto-reset this slot and relaunched a
                    # sampled shot; nothing after this belongs to our shot.
                    finished_at[index] = -1.0
                if finished_at[index] == -1.0:
                    continue
                row = state[index].tolist()
                samples[index].append([time_s, *row[0:3], *row[3:6]])
            if len(results) == len(shots) and all(value == -1.0 for value in finished_at):
                break
        episodes = [
            {
                "shot_id": shot.shot_id,
                "result": results.get(shot.shot_id),
                "samples": samples[index],
            }
            for index, shot in enumerate(shots)
        ]
        payload = _payload(
            "isaac_lab",
            args.bank,
            args.split,
            episodes,
            physics_dt=PHYSICS_DT,
            control_dt=control_dt,
            engine=f"isaacsim {version('isaacsim')} / isaaclab {version('isaaclab')}",
            device=args.device,
        )
        return payload
    finally:
        # Kit can hang while unloading plugins after a headless run; the output
        # has already been produced by the caller-visible return value.
        del app


def _first_bounce(samples: list[list[float]]) -> int | None:
    """Index of the first sample after vz turns from falling to rising."""
    for index in range(1, len(samples)):
        if samples[index - 1][6] < 0.0 and samples[index][6] > 0.0:
            return index
    return None


def _apex_after(samples: list[list[float]], index: int) -> float | None:
    """Height of the first apex after a bounce, if the samples reach it.

    The apex height is set by the vertical restitution alone and does not
    depend on where a 5 ms sample happens to fall inside a millisecond-scale
    contact, which a velocity ratio across the bounce does.
    """
    for k in range(index + 1, len(samples)):
        if samples[k - 1][6] > 0.0 >= samples[k][6]:
            return max(samples[k - 1][3], samples[k][3])
    return None


def _distance(a: list[float], b: list[float]) -> float:
    return sum((a[i] - b[i]) ** 2 for i in (1, 2, 3)) ** 0.5


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": median(ordered),
        "p95": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "max": ordered[-1],
    }


def compare(reference: dict[str, Any], candidate: dict[str, Any], *, post_window_s: float):
    if (reference["bank"], reference["split"]) != (candidate["bank"], candidate["split"]):
        raise SystemExit("the two rollouts were run on different shots")
    by_id = {episode["shot_id"]: episode for episode in candidate["episodes"]}
    tags = {
        shot.shot_id: list(shot.tags)
        for shot in _shots(reference["bank"], reference["split"], None)
    }
    rows = []
    flight_errors: list[float] = []
    post_errors: list[float] = []
    bounce_time: list[float] = []
    bounce_xy: list[float] = []
    apex_height: dict[str, list[float]] = {"reference": [], "candidate": []}
    apex_difference: list[float] = []
    agreement = {"incoming_valid": 0, "failure_reason": 0, "compared": 0}
    for episode in reference["episodes"]:
        other = by_id.get(episode["shot_id"])
        if other is None or other["result"] is None:
            continue
        a, b = episode["samples"], other["samples"]
        ra, rb = episode["result"], other["result"]
        agreement["compared"] += 1
        agreement["incoming_valid"] += ra["incoming_valid"] == rb["incoming_valid"]
        agreement["failure_reason"] += ra["failure_reason"] == rb["failure_reason"]
        bounce_a, bounce_b = _first_bounce(a), _first_bounce(b)
        count = min(len(a), len(b))
        split = min(
            value for value in (bounce_a, bounce_b, count) if value is not None
        )
        flight = max((_distance(a[i], b[i]) for i in range(max(0, split - 1))), default=0.0)
        flight_errors.append(flight)
        row: dict[str, Any] = {
            "shot_id": episode["shot_id"],
            "level": ra["level"],
            "tags": tags.get(episode["shot_id"], []),
            "reference": {k: ra[k] for k in ("incoming_valid", "failure_reason")},
            "candidate": {k: rb[k] for k in ("incoming_valid", "failure_reason")},
            "max_flight_divergence_m": flight,
        }
        if bounce_a is not None and bounce_b is not None:
            dt = b[bounce_b][0] - a[bounce_a][0]
            dxy = ((a[bounce_a][1] - b[bounce_b][1]) ** 2 + (a[bounce_a][2] - b[bounce_b][2]) ** 2) ** 0.5
            bounce_time.append(abs(dt))
            bounce_xy.append(dxy)
            apex = [_apex_after(samples, index) for samples, index in ((a, bounce_a), (b, bounce_b))]
            if None not in apex:
                apex_height["reference"].append(apex[0])
                apex_height["candidate"].append(apex[1])
                apex_difference.append(abs(apex[0] - apex[1]))
                row["post_bounce_apex_z_m"] = {"reference": apex[0], "candidate": apex[1]}
            window = round(post_window_s / reference["control_dt"])
            post = [
                _distance(a[bounce_a + k], b[bounce_b + k])
                for k in range(window)
                if bounce_a + k < len(a) and bounce_b + k < len(b)
            ]
            if post:
                post_errors.append(max(post))
                row["max_post_bounce_divergence_m"] = max(post)
            row["bounce_time_difference_s"] = dt
            row["bounce_xy_difference_m"] = dxy
        rows.append(row)
    compared = agreement["compared"]
    summary = {
        "shots_compared": compared,
        "incoming_valid_agreement": agreement["incoming_valid"] / compared if compared else None,
        "failure_reason_agreement": agreement["failure_reason"] / compared if compared else None,
        "flight_divergence_before_first_bounce_m": _quantiles(flight_errors),
        "first_bounce_time_difference_s": _quantiles(bounce_time),
        "first_bounce_xy_difference_m": _quantiles(bounce_xy),
        f"divergence_within_{post_window_s:g}s_after_bounce_m": _quantiles(post_errors),
        "post_bounce_apex_z_m": {
            name: _quantiles(values) for name, values in apex_height.items()
        },
        "post_bounce_apex_z_difference_m": _quantiles(apex_difference),
    }
    disagreements = [
        row
        for row in rows
        if row["reference"] != row["candidate"]
    ]
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bank": reference["bank"],
        "split": reference["split"],
        "reference": {k: reference.get(k) for k in ("backend", "engine", "physics_dt", "control_dt")},
        "candidate": {
            k: candidate.get(k) for k in ("backend", "engine", "physics_dt", "control_dt", "device")
        },
        "sampling": "ball state once per control step; first bounce = first vz sign change",
        "summary": summary,
        "disagreements": disagreements,
        "episodes": rows,
    }


def _fmt(value: Any, scale: float = 1.0, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * scale:.{digits}f}"


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    ref, cand = report["reference"], report["candidate"]
    post_key = next(key for key in s if key.startswith("divergence_within_"))
    lines = [
        "# Backend parity: table-tennis ball flight and bounce",
        "",
        (f"Bank `{report['bank']}` split `{report['split']}`, {s['shots_compared']} shots, "
        f"blade parked. Reference {ref['engine']} vs candidate {cand['engine']} "
        f"({cand.get('device')}). Generated {report['generated_at']}."),
        "",
        "| Quantity | median | p95 | max |",
        "|---|---:|---:|---:|",
    ]
    for label, key, scale, unit in (
        ("Flight divergence before first bounce", "flight_divergence_before_first_bounce_m", 1000, "mm"),
        ("First-bounce time difference", "first_bounce_time_difference_s", 1000, "ms"),
        ("First-bounce position difference", "first_bounce_xy_difference_m", 1000, "mm"),
        ("Divergence after bounce", post_key, 1000, "mm"),
    ):
        q = s[key]
        lines.append(
            f"| {label} ({unit}) | {_fmt(q['median'], scale)} | {_fmt(q['p95'], scale)} | {_fmt(q['max'], scale)} |"
        )
    for name in ("reference", "candidate"):
        q = s["post_bounce_apex_z_m"][name]
        lines.append(
            f"| Post-bounce apex height, {name} (mm) | {_fmt(q['median'], 1000)} | {_fmt(q['p95'], 1000)} | {_fmt(q['max'], 1000)} |"
        )
    q = s["post_bounce_apex_z_difference_m"]
    lines.append(
        f"| Post-bounce apex height difference (mm) | {_fmt(q['median'], 1000)} | {_fmt(q['p95'], 1000)} | {_fmt(q['max'], 1000)} |"
    )
    lines += [
        "",
        (f"Judge agreement: incoming_valid {_fmt(s['incoming_valid_agreement'], 100)}%, "
        f"failure_reason {_fmt(s['failure_reason_agreement'], 100)}%."),
        "",
    ]
    if report["disagreements"]:
        lines += [
            "## Verdict disagreements",
            "",
            "| shot | tags | MuJoCo | Isaac |",
            "|---|---|---|---|",
        ]
        for row in report["disagreements"]:
            lines.append(
                f"| {row['shot_id']} | {', '.join(row['tags'])} | {row['reference']['failure_reason']} "
                f"(valid={row['reference']['incoming_valid']}) | "
                f"{row['candidate']['failure_reason']} (valid={row['candidate']['incoming_valid']}) |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("mujoco", "isaac"):
        run = sub.add_parser(name)
        run.add_argument("--bank", default="table_tennis/return-v1")
        run.add_argument("--split", default="dev")
        run.add_argument("--limit", type=int)
        run.add_argument("--tail-s", type=float, default=0.3)
        run.add_argument("--out", type=Path, required=True)
        if name == "isaac":
            run.add_argument("--device", default="cpu")
    cmp = sub.add_parser("compare")
    cmp.add_argument("reference", type=Path)
    cmp.add_argument("candidate", type=Path)
    cmp.add_argument("--post-window-s", type=float, default=0.1)
    cmp.add_argument("--report", type=Path)
    cmp.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)

    if args.command == "compare":
        report = compare(
            json.loads(args.reference.read_text()),
            json.loads(args.candidate.read_text()),
            post_window_s=args.post_window_s,
        )
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text)
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(markdown(report))
        print(markdown(report))
        return 0

    payload = run_mujoco(args) if args.command == "mujoco" else run_isaac(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, allow_nan=False) + "\n")
    print(f"wrote {len(payload['episodes'])} episodes to {args.out}", flush=True)
    if args.command == "isaac":
        sys.stdout.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
