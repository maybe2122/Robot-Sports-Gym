#!/usr/bin/env python3
"""Score a trained policy on the table-tennis return benchmark, level by level.

The evaluation itself lives in
:mod:`multisport_sim.benchmark.policy_eval`, so a submission can call it from
Python and get exactly what this command produces.  See that module for the
policy contract; in short, a policy is any callable ``obs -> action`` returning
seven joint-position setpoints in radians.

Usage::

    python scripts/eval_policy.py --policy my_pkg.eval:load_policy \
        --policy-id my-sac-v3 --split test --levels all --out reports/my-sac-v3

    python scripts/eval_policy.py --policy my_pkg.eval:load_policy \
        --policy-id my-vision --track vision --split test --levels all

``--policy`` names ``module:attribute``.  The attribute may be the policy
itself, or a zero-argument factory returning one; the factory is tried first,
so loading weights happens here rather than at import time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from multisport_sim.benchmark.policy_eval import (
    TRACKS,
    cross_level_robustness_gap,
    difficulty_table,
    evaluate_levels,
    load_policy,
    robustness_line,
)
from multisport_sim.benchmark.shot_bank import VALID_LEVELS
from multisport_sim.benchmark.task_config import TABLE_TENNIS_RETURN_PANDA_V1 as TASK


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument(
        "--policy", required=True, help="'module:attribute' of the policy or its factory"
    )
    result.add_argument("--policy-id", help="name recorded in the report; defaults to --policy")
    result.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    result.add_argument(
        "--bank",
        default=TASK.bank_resource,
        help=(
            "packaged shot bank; defaults to the one the task is scored on, so a "
            "policy's numbers land beside the published baselines rather than "
            "beside a frozen fixture with a handful of episodes per level"
        ),
    )
    result.add_argument(
        "--track",
        choices=TRACKS,
        default="state",
        help=(
            "'state' feeds the policy the 33-number privileged observation; "
            "'vision' feeds it a VisionObservation with cameras and no ball state"
        ),
    )
    result.add_argument(
        "--levels",
        default="all",
        help=f"comma-separated subset of {', '.join(VALID_LEVELS)}, or 'all'",
    )
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--episodes", type=int, help="run only the first N shots of each level")
    result.add_argument("--out", type=Path, help="directory for per-level JSON and Markdown")
    result.add_argument(
        "--observation",
        default=None,
        help="what the policy was given; recorded in the report for auditing",
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

    try:
        reports = evaluate_levels(
            load_policy(args.policy),
            levels=levels,
            split=args.split,
            seed=args.seed,
            policy_id=policy_id,
            track=args.track,
            observation_label=args.observation,
            episodes=args.episodes,
            bank=args.bank,
            out=args.out,
        )
    except (ValueError, TypeError, KeyError, ImportError) as exc:
        print(f"eval_policy: error: {exc}", file=sys.stderr)
        return 2

    gap = cross_level_robustness_gap(reports)
    summary = (
        f"# {policy_id} on table-tennis-return-panda-v1 "
        f"({args.split}, {args.track} track)\n\n"
        f"Shot bank digest `{reports[levels[0]]['shot_bank']['digest'][:12]}`, "
        f"seed {args.seed}.\n\n"
        f"{difficulty_table(reports)}\n"
        f"\n{robustness_line(gap)}\n"
    )
    print(summary)
    if args.out is not None:
        (args.out / f"{args.split}-{args.track}-summary.md").write_text(summary, encoding="utf-8")
        (args.out / f"{args.split}-{args.track}-summary.json").write_text(
            json.dumps(
                {
                    "levels": {
                        level: report.get("assessment") for level, report in reports.items()
                    },
                    "robustness_gap": gap,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    passed = all((report.get("assessment") or {}).get("passed") for report in reports.values())
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
