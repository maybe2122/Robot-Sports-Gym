from __future__ import annotations

import json
import re
from pathlib import Path

from multisport_sim.squash_demo import validate_gif, validate_score_report

ROOT = Path(__file__).parents[1]
REQUIRED_PUBLIC_FILES = (
    "README.md",
    "README.en.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "SUPPORT.md",
    "GOVERNANCE.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "docs/BENCHMARK_SPEC.md",
    "docs/TABLE_TENNIS_SHOT_SKILL.md",
    "docs/REPRODUCIBILITY.md",
    "docs/ROADMAP.md",
    "scripts/capture_squash_demo.py",
    "docs/images/shot-skill/squash-serve-score-demo.json",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/workflows/ci.yml",
)


def test_public_repository_files_are_present() -> None:
    missing = [path for path in REQUIRED_PUBLIC_FILES if not (ROOT / path).is_file()]
    assert not missing


def test_local_markdown_links_resolve() -> None:
    broken: list[str] = []
    for document in ROOT.rglob("*.md"):
        if any(part.startswith(".") for part in document.relative_to(ROOT).parts):
            continue
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]*]\(([^)]+)\)", text):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not broken


def test_squash_score_demo_is_a_real_animation() -> None:
    asset = ROOT / "docs/images/shot-skill/squash-serve-score-demo.gif"
    info = validate_gif(asset)
    assert info.frames == 47


def test_squash_score_demo_report_matches_the_expected_rally() -> None:
    report = ROOT / "docs/images/shot-skill/squash-serve-score-demo.json"
    with report.open(encoding="utf-8") as stream:
        summary = validate_score_report(json.load(stream))
    assert summary["score"] == {"A": 0, "B": 1}
    assert summary["point_time_s"] == 2.833
