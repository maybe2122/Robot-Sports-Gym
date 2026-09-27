"""Every file the benchmark ships or writes validates against its versioned schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")

from multisport_sim.benchmark.schemas import SCHEMAS, load, validate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "multisport_sim" / "benchmark" / "data"
REPORTS = ROOT / "reports"


def test_every_schema_has_a_stable_versioned_id() -> None:
    for name, filename in SCHEMAS.items():
        schema = load(name)
        assert schema["$id"].endswith("/" + filename)
        assert ".v" in filename


@pytest.mark.parametrize(
    "manifest", sorted(DATA.glob("*/*/manifest.json")), ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}"
)
def test_every_shipped_bank_validates(manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    validate("shot-bank-manifest", payload)
    for split in payload["splits"].values():
        lines = (manifest.parent / split["file"]).read_text(encoding="utf-8").splitlines()
        for line in lines[:: max(1, len(lines) // 25)]:
            validate("shot-record", json.loads(line))


@pytest.mark.parametrize("path", sorted(REPORTS.glob("*-baselines.json")), ids=lambda p: p.name)
def test_committed_baseline_tables_validate(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = "learned-baseline" if path.name.startswith("learned-") else "baseline-table"
    validate(name, payload)


def test_the_committed_summaries_validate() -> None:
    validate("cross-task-summary", json.loads((REPORTS / "cross-task-summary.json").read_text()))
    validate(
        "backend-parity", json.loads((REPORTS / "table-tennis-backend-parity.json").read_text())
    )


def test_a_live_report_validates() -> None:
    from multisport_sim.benchmark_cli import run_from_args

    report = run_from_args(
        argparse.Namespace(
            sport="table-tennis", task=None, backend="mujoco", level="L1", split="dev",
            bank=None, shot_bank=None, episodes=None, seed=0, robot="none", track="state",
            controller="scripted", control_hz=200.0, report=None, markdown=None,
            require_pass=False,
        )
    )
    validate("shot-skill-report", json.loads(json.dumps(report)))
