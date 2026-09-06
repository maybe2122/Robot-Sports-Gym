from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_committed_baseline_reports_match_their_packaged_banks() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [sys.executable, "scripts/check_baseline_reports.py"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "48 rows, dev=50, test=100" in result.stdout
    assert "12 rows, test=100" in result.stdout
