#!/usr/bin/env python3
"""Reference baselines for a mocap-fixture task, level by level.

The fixture tasks -- tennis return, badminton serve, and the other launch tasks
as they land -- have no arm holding the implement, so there is no embodied
baseline.  Both rows here are diagnostics: `noop` is the floor, `scripted`
bounds what a perfectly informed fixture can do, and neither is a submission.

    python scripts/run_fixture_baselines.py --sport badminton [--out reports] [--split test]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from multisport_sim.benchmark.metrics import robustness_gap
from multisport_sim.benchmark.shot_bank import VALID_LEVELS
from multisport_sim.benchmark.types import EpisodeResult
from multisport_sim.benchmark_cli import MOCAP_TASKS, run_from_args

SPORTS = {
    "tennis": {
        "title": "Tennis return baselines",
        "stem": "tennis-baselines",
        "setting": "on the regulation singles court",
        "scripted": (
            "`scripted` reads privileged ball state and moves a kinematic racket, so it "
            "bounds what the fixture can do rather than what a robot could.  Tennis has no "
            "embodied task yet: there is no arm holding the racket."
        ),
    },
    "badminton": {
        "title": "Badminton serve baselines",
        "stem": "badminton-baselines",
        "setting": "into the BWF singles service court",
        "scripted": (
            "`scripted` reads the privileged shuttle state and the shot's target and swings "
            "a kinematic racket along a straight line at a speed read from a measured carry "
            "table (`scripts/calibrate_serve.py`); it bounds what the fixture can do, not "
            "what a robot could.  There is no embodied badminton task yet."
        ),
    },
}

CONTROLLERS = ("noop", "scripted")


def _request(
    sport: str, level: str, split: str, seed: int, controller: str
) -> argparse.Namespace:
    return argparse.Namespace(
        sport=sport,
        task=None,
        backend="mujoco",
        level=level,
        split=split,
        bank=None,
        shot_bank=None,
        episodes=None,
        seed=seed,
        robot="none",
        track="state",
        controller=controller,
        control_hz=200.0,
        report=None,
        markdown=None,
        require_pass=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sport", choices=sorted(SPORTS), required=True)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    sport = SPORTS[args.sport]
    task_id = MOCAP_TASKS[args.sport].task_id

    rows: list[dict] = []
    episodes: dict[str, list[EpisodeResult]] = {}
    digest = ""
    for level in VALID_LEVELS:
        for controller in CONTROLLERS:
            report = run_from_args(
                _request(args.sport, level, args.split, args.seed, controller)
            )
            digest = report["shot_bank"]["digest"]
            assessment = report["assessment"] or {}
            metrics = report["metrics"]
            rows.append(
                {
                    "level": level,
                    "controller": controller,
                    "episodes": report["episodes"],
                    "primary_metric": assessment.get("primary_metric"),
                    "primary_value": assessment.get("primary_value"),
                    "pass_threshold": assessment.get("pass_threshold"),
                    "passed": assessment.get("passed"),
                    "worst_bucket": assessment.get("worst_bucket"),
                    "primary_confidence_interval_95": assessment.get(
                        "primary_confidence_interval_95"
                    ),
                    "hit_rate": metrics["hit_rate"],
                    "valid_return_rate": metrics["valid_return_rate"],
                    "target_rate": metrics["target_rate"],
                    "mean_episode_time_s": metrics["mean_episode_time_s"],
                }
            )
            episodes.setdefault(controller, []).extend(
                EpisodeResult.from_dict(record) for record in report["results"]
            )
            print(
                f"{level} {controller}: {assessment.get('primary_metric')}="
                f"{assessment.get('primary_value')}",
                file=sys.stderr,
            )

    gaps = {controller: robustness_gap(values) for controller, values in episodes.items()}
    lines = [
        f"# {sport['title']}",
        "",
        (
            f"`{task_id}` {sport['setting']}, split "
            f"`{args.split}` (digest `{digest[:12]}`), MuJoCo mocap fixture."
        ),
        f"Regenerate with `python scripts/run_fixture_baselines.py --sport {args.sport}`.",
        "",
        f"**Neither row is a submission.** `noop` is the floor; {sport['scripted']}",
        "",
        "| Level | Controller | n | Primary metric | Value | 95% CI | Threshold | Pass | Hit | Return | Target |",
        "|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {level} | `{controller}` | {n} | `{metric}` | {value:.0%} | {interval} | "
            "{threshold:.0%} | {passed} | {hit:.0%} | {ret:.0%} | {tgt:.0%} |".format(
                level=row["level"],
                controller=row["controller"],
                n=row["episodes"],
                metric=row["primary_metric"],
                value=row["primary_value"] or 0.0,
                threshold=row["pass_threshold"] or 0.0,
                interval=(
                    "n/a"
                    if row["primary_confidence_interval_95"] is None
                    else "{:.0%}–{:.0%}".format(
                        *row["primary_confidence_interval_95"]
                    )
                ),
                passed="PASS" if row["passed"] else "FAIL",
                hit=row["hit_rate"],
                ret=row["valid_return_rate"],
                tgt=row["target_rate"],
            )
        )
    lines += ["", "## Robustness gap", ""]
    for controller, gap in gaps.items():
        if gap is None:
            continue
        breakdown = gap.get("level_breakdown", {})
        l4 = breakdown.get("L4", {})
        l5 = breakdown.get("L5", {})
        lines.append(
            f"- `{controller}`: {gap['in_distribution']:.0%} on L2-L3 "
            f"({gap['in_distribution_episodes']} episodes) minus {gap['perturbed']:.0%} "
            f"on L4-L5 ({gap['perturbed_episodes']}) = **{gap['gap']:+.0%}**; "
            f"L4 {l4.get('value', 0.0):.0%} ({l4.get('episodes', 0)}), "
            f"L5 {l5.get('value', 0.0):.0%} ({l5.get('episodes', 0)})"
        )
    lines += [
        "",
        "L4 and L5 are independently stratified challenge distributions; only L5",
        "applies the declared observation/action/domain perturbations. Their point",
        "estimates therefore need not be monotonic, which is why both are retained.",
    ]
    lines.append("")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{sport['stem']}.md").write_text("\n".join(lines), encoding="utf-8")
    (args.out / f"{sport['stem']}.json").write_text(
        json.dumps(
            {
                "schema": "multisport-baseline-table-v0",
                "task": task_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "split": args.split,
                "seed": args.seed,
                "shot_bank_digest": digest,
                "rows": rows,
                "robustness_gap": [
                    {"controller": controller, **(gap or {})} for controller, gap in gaps.items()
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
