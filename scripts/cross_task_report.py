#!/usr/bin/env python3
"""One table for every task: level by level, controller by controller.

Each task writes its own baseline table (``scripts/run_baselines.py`` for the
embodied table-tennis tasks, ``scripts/run_fixture_baselines.py`` for the mocap
fixtures).  This script reads them all and writes a single summary, so the
state of the whole benchmark can be read in one place:

    PYTHONPATH=src python scripts/cross_task_report.py [--reports reports]

It re-runs nothing; ``make baselines-all`` regenerates every input first.  Only
the ``test`` split is summarised -- it is the one the thresholds are stated for.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")
# (report stem, human label, embodiment) in the order the table lists them.
TABLES = (
    ("table-tennis-panda-baselines", "Table tennis return", "Franka Panda (joint space)"),
    ("table-tennis-g1-baselines", "Table tennis return", "Unitree G1, fixed base"),
    ("tennis-baselines", "Tennis return", "mocap racket fixture"),
    ("badminton-baselines", "Badminton serve", "mocap racket fixture"),
    ("football-baselines", "Football kick", "mocap boot fixture"),
    ("basketball-baselines", "Basketball shoot", "mocap launcher fixture"),
)


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarise(reports: Path) -> dict[str, Any]:
    tasks = []
    for stem, label, embodiment in TABLES:
        payload = _load(reports / f"{stem}.json")
        if payload is None:
            tasks.append({"report": f"{stem}.json", "label": label, "missing": True})
            continue
        rows = [row for row in payload["rows"] if row.get("split", "test") == "test"]
        controllers: dict[str, dict[str, Any]] = {}
        for row in rows:
            entry = controllers.setdefault(row["controller"], {"levels": {}})
            entry["levels"][row["level"]] = {
                "metric": row["primary_metric"],
                "value": row["primary_value"],
                "threshold": row["pass_threshold"],
                "passed": row["passed"],
                "episodes": row["episodes"],
                "interval_95": row.get("primary_confidence_interval_95"),
            }
        for gap in payload.get("robustness_gap", []):
            if gap.get("split", "test") != "test" or "gap" not in gap:
                continue
            controllers.setdefault(gap["controller"], {"levels": {}})["robustness_gap"] = gap["gap"]
        for entry in controllers.values():
            entry["levels_passed"] = sum(
                1 for level in entry["levels"].values() if level["passed"]
            )
        tasks.append(
            {
                "report": f"{stem}.json",
                "label": label,
                "embodiment": embodiment,
                "task": payload["task"],
                "controllers": controllers,
            }
        )
    return {
        "schema": "multisport-cross-task-summary-v0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": "test",
        "tasks": tasks,
    }


def _cell(level: dict[str, Any] | None) -> str:
    if level is None or level["value"] is None:
        return "—"
    mark = "**" if level["passed"] else ""
    return f"{mark}{level['value']:.0%}{mark}"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Cross-task baseline summary",
        "",
        ("Test split, primary metric per level (bold = meets the frozen threshold). "
        "Regenerate every input with `make baselines-all`; this page with "
        "`python scripts/cross_task_report.py`."),
        "",
        ("**No row is a submission.** Every controller here is a reference: `hold`/`noop` "
        "and `random` are floors, the scripted ones read privileged state."),
        "",
        "| Task | Embodiment | Controller | " + " | ".join(LEVELS) + " | Passed | Gap |",
        "|---|---|---|" + "---:|" * len(LEVELS) + "---:|---:|",
    ]
    for task in summary["tasks"]:
        if task.get("missing"):
            lines.append(f"| {task['label']} | — | *missing `{task['report']}`* |" + " |" * 8)
            continue
        for controller, entry in task["controllers"].items():
            cells = " | ".join(_cell(entry["levels"].get(level)) for level in LEVELS)
            gap = entry.get("robustness_gap")
            gap_text = "—" if gap is None else f"{gap:+.0%}"
            lines.append(
                f"| {task['label']} (`{task['task']}`) | {task['embodiment']} | `{controller}` | "
                f"{cells} | {entry['levels_passed']}/{len(entry['levels'])} | {gap_text} |"
            )
    lines += [
        "",
        ("Primary metrics: L0 `incoming_valid_rate`, L1 `hit_rate`, L2 `valid_return_rate`, "
        "L3 `target_rate`, L4/L5 the worst pass bucket's `valid_return_rate`. For the launch "
        "tasks `valid_return` means a legal serve, a goal or a made basket. Gap is "
        "`valid_return_rate` on L2-L3 minus L4-L5."),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    summary = summarise(args.reports)
    (args.reports / "cross-task-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    text = markdown(summary)
    (args.reports / "cross-task-summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
