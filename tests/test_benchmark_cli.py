from __future__ import annotations

import json

from multisport_sim.benchmark_cli import _bank_for, main, parser, run_from_args


def test_cli_writes_strict_json_and_markdown_reports(tmp_path) -> None:
    json_path = tmp_path / "reports" / "l1.json"
    markdown_path = tmp_path / "reports" / "l1.md"

    exit_status = main(
        [
            "--level",
            "L1",
            "--episodes",
            "1",
            "--controller",
            "scripted",
            "--seed",
            "7",
            "--report",
            str(json_path),
            "--markdown",
            str(markdown_path),
        ]
    )

    assert exit_status == 0
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["schema"] == "multisport-shot-skill-report-v0"
    assert report["task"] == "table-tennis-return-v0"
    assert report["backend"]["name"] == "mujoco"
    assert report["software"]["multisport_sim"]
    assert report["software"]["python"]
    assert report["shot_bank"]["selected_shot_ids"] == ["tt-return-v0-dev-l1-0001"]
    assert report["shot_bank"]["selected_count"] == 1
    assert len(report["shot_bank"]["manifest_digest"]) == 64
    assert report["shot_bank_digest"] == report["shot_bank"]["source_digest"]
    assert report["results"][0]["hit"] is True
    assert len(report["execution"]["episode_seeds"]) == 1
    assert "Result: **PASS**" in markdown_path.read_text(encoding="utf-8")


def test_low_score_is_data_unless_require_pass_is_explicit() -> None:
    base = ["--level", "L2", "--episodes", "1", "--controller", "noop"]

    assert main(base) == 0
    assert main([*base, "--require-pass"]) == 1


def test_cli_rejects_episode_count_larger_than_fixed_level(capsys) -> None:
    status = main(["--level", "L1", "--episodes", "999"])

    assert status == 2
    assert "contains only" in capsys.readouterr().err


def test_cli_rejects_misleading_output_suffix(capsys, tmp_path) -> None:
    status = main(["--episodes", "1", "--report", str(tmp_path / "report.md")])

    assert status == 2
    assert "must end in .json" in capsys.readouterr().err


def test_same_seed_preserves_selection_and_raw_results() -> None:
    arguments = parser().parse_args(
        ["--level", "L2", "--episodes", "1", "--controller", "scripted", "--seed", "19"]
    )

    first = run_from_args(arguments)
    second = run_from_args(arguments)

    assert second["shot_bank"]["selected_shot_ids"] == first["shot_bank"]["selected_shot_ids"]
    assert second["results"] == first["results"]


def test_robot_defaults_to_the_statistically_sufficient_bank() -> None:
    arguments = parser().parse_args(["--robot", "panda", "--controller", "hold"])

    assert _bank_for(arguments) == "table_tennis/return-v1"
