"""The submission audit catches what a package cannot hide."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
from multisport_sim.benchmark.controllers import NoOpController
from multisport_sim.benchmark.metrics import aggregate_results, assess_level
from multisport_sim.benchmark.runner import RunConfig, run_shots
from multisport_sim.benchmark.shot_bank import ShotBank
from multisport_sim.benchmark.task_config import TABLE_TENNIS_RETURN_V0

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audit_submission


@pytest.fixture(scope="module")
def genuine(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("package")
    bank = ShotBank.from_resource(split="dev")
    task = replace(TABLE_TENNIS_RETURN_V0, split="dev")
    levels = {}
    for level in ("L0", "L1"):
        output = run_shots(
            MujocoShotBackend(), NoOpController(), bank.filter(level=level),
            config=RunConfig.from_task_config(task),
        )
        aggregate = aggregate_results(output.results, bucket_groups=bank.manifest["bucket_groups"])
        levels[level] = {
            "assessment": assess_level(output.results, bank.manifest, level=level),
            "metrics": aggregate["metrics"],
            "results": [result.to_dict() for result in output.results],
        }
    (root / "config").mkdir()
    (root / "policy").mkdir()
    (root / "config" / "run.json").write_text("{}")
    (root / "config" / "task_config.json").write_text("{}")
    (root / "environment.txt").write_text("test\n")
    weights = b"weights"
    (root / "policy" / "actor.bin").write_bytes(weights)
    (root / "policy" / "hashes.json").write_text(
        json.dumps({"actor.bin": sha256(weights).hexdigest()})
    )
    (root / "metrics.json").write_text(json.dumps({"split": "dev", "levels": levels}))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "shot_bank": {"digest": bank.digest},
                "repository": {"commit": "abc", "dirty": False},
                "reproduce_command": "python scripts/package_submission.py --out x",
            }
        )
    )
    return root


def _copy(genuine: Path, tmp_path: Path) -> Path:
    import shutil

    target = tmp_path / "copy"
    shutil.copytree(genuine, target)
    return target


def _failed(audit) -> set[str]:
    return {item["check"] for item in audit.checks if item["hard"] and not item["passed"]}


def test_a_genuine_package_passes(genuine: Path) -> None:
    audit = audit_submission.audit_package(genuine)
    assert audit.passed, audit.checks
    warnings = {item["check"] for item in audit.checks if not item["passed"]}
    assert warnings == {"eligibility.leaderboard"}  # v0 is an experimental bank


def test_dropping_an_episode_is_caught(genuine: Path, tmp_path: Path) -> None:
    package = _copy(genuine, tmp_path)
    metrics = json.loads((package / "metrics.json").read_text())
    metrics["levels"]["L1"]["results"].pop()
    (package / "metrics.json").write_text(json.dumps(metrics))
    assert "L1.completeness" in _failed(audit_submission.audit_package(package))


def test_editing_a_result_is_caught(genuine: Path, tmp_path: Path) -> None:
    package = _copy(genuine, tmp_path)
    metrics = json.loads((package / "metrics.json").read_text())
    metrics["levels"]["L0"]["results"][0]["incoming_valid"] = False
    metrics["levels"]["L0"]["results"][0]["failure_reason"] = "out"
    (package / "metrics.json").write_text(json.dumps(metrics))
    failed = _failed(audit_submission.audit_package(package))
    assert {"L0.verdict", "L0.metrics"} <= failed


def test_swapped_weights_are_caught(genuine: Path, tmp_path: Path) -> None:
    package = _copy(genuine, tmp_path)
    (package / "policy" / "actor.bin").write_bytes(b"other weights")
    assert "integrity.policy_files" in _failed(audit_submission.audit_package(package))


def test_an_unknown_bank_is_caught(genuine: Path, tmp_path: Path) -> None:
    package = _copy(genuine, tmp_path)
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["shot_bank"]["digest"] = "0" * 64
    (package / "manifest.json").write_text(json.dumps(manifest))
    assert "integrity.shot_bank" in _failed(audit_submission.audit_package(package))
