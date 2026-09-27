#!/usr/bin/env python3
"""Audit a result package: does every number in it follow from its own evidence?

``scripts/package_submission.py`` writes a package; this script is what a
leaderboard maintainer runs on one before it is listed.  It trusts nothing the
package *says* about itself that it can recompute:

* **completeness** -- every level reports exactly the bank's shots for that
  split and level, each once: a package cannot drop the episodes it lost;
* **integrity** -- the shot bank digest matches a bank this repository ships,
  and every weight file matches its recorded sha256;
* **recomputation** -- every pass/fail verdict, primary value and aggregate
  metric is recomputed from the raw per-episode results with the bank's frozen
  criteria, and must equal what the package reports;
* **provenance** -- the package was built from a clean checkout of a commit;
* **replay** (``--rerun``) -- the reproduce command is executed and every
  episode must come out identical.

    python scripts/audit_submission.py submission/ [--rerun] [--report audit.json]

Exit status 0 means every hard check passed.  Warnings (a dirty tree, an
experimental bank, a package without weights) do not fail the audit but are
stated in the verdict, because each one limits what the result can claim.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

from multisport_sim.benchmark.metrics import aggregate_results, assess_level
from multisport_sim.benchmark.shot_bank import ShotBank, ShotBankError
from multisport_sim.benchmark.types import EpisodeResult

REQUIRED_FILES = (
    "manifest.json",
    "metrics.json",
    "config/run.json",
    "config/task_config.json",
    "environment.txt",
    "policy/hashes.json",
)
DATA_ROOT = Path(__file__).resolve().parents[1] / "src" / "multisport_sim" / "benchmark" / "data"
TOLERANCE = 1e-12


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def record(self, name: str, passed: bool, detail: str = "", *, hard: bool = True) -> bool:
        self.checks.append(
            {"check": name, "passed": bool(passed), "hard": hard, "detail": detail}
        )
        return passed

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.checks if item["hard"])


def _find_bank(digest: str, split: str) -> ShotBank | None:
    """The shipped bank whose ``split`` has this exact digest, if any."""
    for manifest in DATA_ROOT.glob("*/*/manifest.json"):
        resource = f"{manifest.parent.parent.name}/{manifest.parent.name}"
        try:
            bank = ShotBank.from_resource(split=split, task=resource)
        except (ShotBankError, KeyError, OSError):
            continue
        if bank.digest == digest:
            return bank
    return None


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(float(a) - float(b)) <= TOLERANCE
    return a == b


def audit_package(root: Path) -> Audit:
    audit = Audit()
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if not audit.record("structure", not missing, f"missing: {missing}" if missing else ""):
        return audit
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

    repository = manifest.get("repository", {})
    audit.record(
        "provenance.clean_tree",
        repository.get("dirty") is False and bool(repository.get("commit")),
        f"commit {repository.get('commit')}, dirty={repository.get('dirty')}",
        hard=False,
    )

    split = metrics.get("split")
    digest = manifest.get("shot_bank", {}).get("digest")
    bank = _find_bank(digest, split) if digest and split else None
    if not audit.record(
        "integrity.shot_bank",
        bank is not None,
        f"split {split!r} digest {digest}" + ("" if bank else " matches no shipped bank"),
    ):
        return audit
    audit.record(
        "eligibility.leaderboard",
        bool(bank.manifest.get("leaderboard_eligible")),
        f"bank status {bank.manifest.get('status')!r}",
        hard=False,
    )

    for level, block in sorted(metrics.get("levels", {}).items()):
        try:
            results = [EpisodeResult.from_dict(record) for record in block["results"]]
        except (KeyError, ValueError) as exc:
            audit.record(f"{level}.results_valid", False, str(exc))
            continue
        expected = sorted(shot.shot_id for shot in bank.filter(level=level))
        reported = sorted(result.shot_id for result in results)
        audit.record(
            f"{level}.completeness",
            reported == expected,
            f"{len(reported)} episodes reported, {len(expected)} shots in the bank",
        )
        verdict = assess_level(results, bank.manifest, level=level)
        claimed = block.get("assessment") or {}
        audit.record(
            f"{level}.verdict",
            _close(verdict["primary_value"], claimed.get("primary_value"))
            and verdict["passed"] == claimed.get("passed"),
            f"recomputed {verdict['primary_value']} passed={verdict['passed']}; "
            f"reported {claimed.get('primary_value')} passed={claimed.get('passed')}",
        )
        aggregate = aggregate_results(results, bucket_groups=bank.manifest.get("bucket_groups", {}))
        mismatched = [
            name
            for name, value in aggregate["metrics"].items()
            if name in block.get("metrics", {}) and not _close(value, block["metrics"][name])
        ]
        audit.record(
            f"{level}.metrics",
            not mismatched,
            f"differ: {mismatched}" if mismatched else "",
        )

    hashes = json.loads((root / "policy" / "hashes.json").read_text(encoding="utf-8"))
    bad = [
        name
        for name, value in hashes.items()
        if not (root / "policy" / name).is_file()
        or sha256((root / "policy" / name).read_bytes()).hexdigest() != value
    ]
    audit.record("integrity.policy_files", not bad, f"mismatched: {bad}" if bad else "")
    audit.record(
        "provenance.weights_included",
        bool(hashes),
        "" if hashes else "no weight files; the result cannot be reproduced by others",
        hard=False,
    )
    return audit


def rerun(root: Path, audit: Audit) -> None:
    """Execute the reproduce command into a scratch directory and compare episodes."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    command = shlex.split(manifest["reproduce_command"])
    with tempfile.TemporaryDirectory() as scratch:
        out_index = command.index("--out") + 1
        command[out_index] = scratch
        completed = subprocess.run(
            [sys.executable, *command[1:]], capture_output=True, text=True, check=False
        )
        if not audit.record(
            "replay.runs", completed.returncode == 0, completed.stderr[-400:]
        ):
            return
        original = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        replayed = json.loads((Path(scratch) / "metrics.json").read_text(encoding="utf-8"))
    for level, block in original["levels"].items():
        before = {r["shot_id"]: r for r in block["results"]}
        after = {r["shot_id"]: r for r in replayed["levels"].get(level, {}).get("results", [])}
        differing = sorted(
            shot for shot in before if before[shot] != after.get(shot)
        )
        audit.record(
            f"{level}.replay",
            not differing,
            f"{len(differing)} of {len(before)} episodes differ" if differing else "",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package", type=Path)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    audit = audit_package(args.package)
    if args.rerun and audit.passed:
        rerun(args.package, audit)
    verdict = {
        "schema": "multisport-audit-v0",
        "package": str(args.package),
        "passed": audit.passed,
        "checks": audit.checks,
    }
    if args.report:
        args.report.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    for item in audit.checks:
        status = "PASS" if item["passed"] else ("FAIL" if item["hard"] else "WARN")
        detail = f"  {item['detail']}" if item["detail"] else ""
        print(f"{status:4}  {item['check']}{detail}")
    print("audit passed" if audit.passed else "audit FAILED")
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
