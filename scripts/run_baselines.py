#!/usr/bin/env python3
"""Regenerate the embodied table-tennis baseline table.

Runs the three reference baselines (`hold`, `random`, `intercept`) over every
level of both frozen splits and writes one JSON plus one Markdown artifact to
``reports/``.  None of these controllers is a submission: `hold` and `random`
bracket the floor, `intercept` uses privileged ball state.  They exist so a
policy's numbers can be read against something.

    python scripts/run_baselines.py [--out reports] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from multisport_sim.benchmark.metrics import robustness_gap
from multisport_sim.benchmark.shot_bank import VALID_LEVELS
from multisport_sim.benchmark.task_config import (
    TABLE_TENNIS_RETURN_G1_V1,
    TABLE_TENNIS_RETURN_PANDA_V1,
)
from multisport_sim.benchmark.types import EpisodeResult
from multisport_sim.benchmark_cli import run_from_args

# (controller, track).  The vision baseline runs the same swing from two
# cameras, so the gap between it and `intercept` is the cost of perception.
BASELINES = (
    ("hold", "state"),
    ("random", "state"),
    ("intercept", "state"),
    ("vision", "vision"),
)
SPLITS = ("dev", "test")

# Which embodiment to sweep, and which baselines exist for it.  The G1 has no
# vision track: the stereo pair is declared relative to the table so it would
# transfer, but a track that has not been run is not one this table may print.
ROBOTS = {
    "panda": (TABLE_TENNIS_RETURN_PANDA_V1, BASELINES),
    "g1": (
        TABLE_TENNIS_RETURN_G1_V1,
        tuple(row for row in BASELINES if row[1] == "state"),
    ),
}


def _row(controller: str, split: str, report: dict) -> dict:
    assessment = report.get("assessment") or {}
    metrics = report.get("metrics", {})
    robot = report.get("robot_metrics", {})
    return {
        "controller": controller,
        "controller_id": report["policy"]["id"],
        "track": report["track"]["id"],
        "split": split,
        "level": report["level"],
        "episodes": report["episodes"],
        "primary_metric": assessment.get("primary_metric"),
        "primary_value": assessment.get("primary_value"),
        "pass_threshold": assessment.get("pass_threshold"),
        "passed": assessment.get("passed"),
        "worst_bucket": assessment.get("worst_bucket"),
        "primary_confidence_interval_95": assessment.get(
            "primary_confidence_interval_95"
        ),
        "hit_rate": metrics.get("hit_rate"),
        "valid_return_rate": metrics.get("valid_return_rate"),
        "target_rate": metrics.get("target_rate"),
        "total_safety_violations": robot.get("total_safety_violations"),
        "mean_energy_joule": robot.get("mean_energy_joule"),
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _interval(value: list[float] | None) -> str:
    return "n/a" if value is None else f"{value[0]:.0%}–{value[1]:.0%}"


def _gap_rows(gaps: dict[tuple[str, str], dict | None]) -> list[str]:
    """One gap row with L4/L5 exposed separately, per baseline and split."""
    lines = [
        "| Split | Controller | L1-L3 | L4 | L5 | L4-L5 | Gap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (split, controller), gap in gaps.items():
        if gap is None:
            continue
        breakdown = gap.get("level_breakdown", {})
        l4 = breakdown.get("L4", {})
        l5 = breakdown.get("L5", {})
        lines.append(
            f"| `{split}` | `{controller}` | {gap['in_distribution']:.0%} "
            f"({gap['in_distribution_episodes']}) | {_percent(l4.get('value'))} "
            f"({l4.get('episodes', 0)}) | {_percent(l5.get('value'))} "
            f"({l5.get('episodes', 0)}) | {gap['perturbed']:.0%} "
            f"({gap['perturbed_episodes']}) | {gap['gap']:+.0%} |"
        )
    return lines


ROBOT_BLURB = {
    "franka-panda-tabletennis-v1": "a Franka Panda",
    "unitree-g1-tabletennis-v1": "a fixed-base Unitree G1 (waist + right arm, 10 joints)",
}


def markdown(
    rows: list[dict],
    *,
    digests: dict[str, str],
    gaps: dict[tuple[str, str], dict | None],
    task: object,
) -> str:
    robot = getattr(task, "robot_id", "an unnamed robot")
    has_vision = any(row["track"] == "vision" for row in rows)
    lines = [
        "# Embodied table-tennis baselines",
        "",
        (
            f"`{getattr(task, 'task_id', '?')}` on "
            f"{ROBOT_BLURB.get(robot, robot)}, MuJoCo backend."
        ),
        (
            f"Shot bank `{getattr(task, 'bank_resource', '?')}`: "
            "dev has 50 and test 100 episodes per level."
        ),
        (
            "Regenerate with `python scripts/run_baselines.py --robot "
            f"{'g1' if 'g1' in robot else 'panda'}`."
        ),
        "",
        "**None of these rows is a submission.** `hold` and `random` are floors;",
        "`intercept` is a scripted controller reading privileged ball state, so its",
        "score bounds what this embodiment can do, it does not represent a policy.",
    ]
    if has_vision:
        lines += [
            "`vision` runs the identical swing from the declared stereo pair with no",
            "privileged state, so the gap between the two is the cost of perception.",
        ]
    lines.append("")
    for split in SPLITS:
        lines += [
            f"## split `{split}` (digest `{digests[split][:12]}`)",
            "",
            "| Level | Controller | Track | n | Primary metric | Value | 95% CI | Threshold | Pass | Hit | Return | Target | Safety | Energy (J) |",
            "|---|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            if row["split"] != split:
                continue
            lines.append(
                "| {level} | `{controller}` | {track} | {n} | `{metric}` | {value} | "
                "{interval} | {threshold} | {passed} | {hit} | {ret} | {tgt} | {safety} | "
                "{energy:.1f} |".format(
                    level=row["level"],
                    controller=row["controller"],
                    track=row["track"],
                    n=row["episodes"],
                    metric=row["primary_metric"],
                    value=_percent(row["primary_value"]),
                    interval=_interval(row["primary_confidence_interval_95"]),
                    threshold=_percent(row["pass_threshold"]),
                    passed="PASS" if row["passed"] else "FAIL",
                    hit=_percent(row["hit_rate"]),
                    ret=_percent(row["valid_return_rate"]),
                    tgt=_percent(row["target_rate"]),
                    safety=row["total_safety_violations"],
                    energy=row["mean_energy_joule"] or 0.0,
                )
            )
        lines.append("")
    lines += [
        "## Robustness gap",
        "",
        "`valid_return_rate` on L1-L3 minus the same rate on L4-L5. L4 and L5",
        "are also shown separately: they are independently stratified challenge",
        "distributions (and only L5 applies declared perturbations), so their",
        "empirical rates are not expected to be monotonic.",
        "",
        *_gap_rows(gaps),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--robot",
        choices=tuple(ROBOTS),
        default="panda",
        help="which embodiment to sweep; each writes its own report pair",
    )
    args = parser.parse_args(argv)
    task, baselines = ROBOTS[args.robot]

    rows: list[dict] = []
    digests: dict[str, str] = {}
    episodes: dict[tuple[str, str], list[EpisodeResult]] = {}
    for split in SPLITS:
        for level in VALID_LEVELS:
            for controller, track in baselines:
                request = argparse.Namespace(
                    sport="table-tennis",
                    task="return",
                    backend="mujoco",
                    level=level,
                    split=split,
                    bank=task.bank_resource,
                    shot_bank=None,
                    episodes=None,
                    seed=args.seed,
                    robot=args.robot,
                    track=track,
                    controller=controller,
                    control_hz=task.control_hz,
                    report=None,
                    markdown=None,
                    require_pass=False,
                )
                report = run_from_args(request)
                digests[split] = report["shot_bank"]["digest"]
                rows.append(_row(controller, split, report))
                episodes.setdefault((split, controller), []).extend(
                    EpisodeResult.from_dict(record) for record in report["results"]
                )
                print(
                    f"{split} {level} {controller}: "
                    f"{report['assessment']['primary_metric']}="
                    f"{report['assessment']['primary_value']}",
                    file=sys.stderr,
                )

    gaps = {key: robustness_gap(values) for key, values in episodes.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "multisport-baseline-table-v0",
        "task": task.task_id,
        "robot": task.robot_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "shot_bank_digests": digests,
        "rows": rows,
        "robustness_gap": [
            {"split": split, "controller": controller, **(gap or {})}
            for (split, controller), gap in gaps.items()
        ],
    }
    stem = f"table-tennis-{args.robot}-baselines"
    (args.out / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out / f"{stem}.md").write_text(
        markdown(rows, digests=digests, gaps=gaps, task=task), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
