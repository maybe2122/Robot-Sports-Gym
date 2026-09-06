"""The submission package: what another person needs to check a result.

The packager is checked on the properties that make a package worth trusting --
the manifest admits a dirty tree, the weights are content-hashed, the videos are
chosen by a rule rather than by taste, and a package built on an experimental
shot bank still says it is not leaderboard-eligible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from multisport_sim.benchmark.assets import FRANKA_PANDA, asset_available

SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

package_submission = pytest.importorskip("package_submission")


def test_video_selection_takes_failures_as_well_as_successes() -> None:
    """"Not only the successes" is a rule about selection, so it is tested."""
    reports = {
        "L2": {
            "results": [
                {"shot_id": "b-win", "valid_return": True},
                {"shot_id": "a-loss", "valid_return": False},
                {"shot_id": "c-win", "valid_return": True},
            ]
        },
        "L4": {
            "results": [
                {"shot_id": "d-loss", "valid_return": False},
            ]
        },
    }

    successes, failures = package_submission._select_episodes(reports, 1)

    # Sorted by shot id, so the choice cannot be shopped for.
    assert [record["shot_id"] for record in successes] == ["b-win"]
    assert [record["shot_id"] for record in failures] == ["a-loss"]
    assert successes[0]["level"] == "L2"
    assert failures[0]["level"] == "L2"


def test_video_selection_never_exceeds_what_was_asked_for() -> None:
    reports = {
        "L2": {
            "results": [
                {"shot_id": f"s{index}", "valid_return": index % 2 == 0} for index in range(10)
            ]
        }
    }

    successes, failures = package_submission._select_episodes(reports, 2)

    assert len(successes) == 2
    assert len(failures) == 2


def test_a_missing_policy_file_stops_the_package(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="policy file does not exist"):
        package_submission._write_policy(
            tmp_path / "policy", "pkg:factory", [tmp_path / "absent.bin"]
        )


def test_weights_are_content_hashed_and_copied(tmp_path: Path) -> None:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"not really a network")

    block = package_submission._write_policy(tmp_path / "policy", "pkg:factory", [weights])

    assert block["complete"] is True
    assert block["files"] == ["weights.bin"]
    assert (tmp_path / "policy" / "weights.bin").read_bytes() == b"not really a network"
    stored = json.loads((tmp_path / "policy" / "hashes.json").read_text())
    assert stored == block["sha256"]
    assert len(stored["weights.bin"]) == 64


def test_a_package_without_weights_says_so_instead_of_looking_complete(tmp_path: Path) -> None:
    block = package_submission._write_policy(tmp_path / "policy", "pkg:factory", [])

    assert block["complete"] is False
    source = (tmp_path / "policy" / "SOURCE.md").read_text()
    assert "does **not** contain the policy" in source
    assert "pkg:factory" in source


def test_the_environment_file_names_the_simulator_and_the_interpreter() -> None:
    text = package_submission._environment_text()

    assert "python=" in text
    assert "mujoco=" in text
    assert "multisport_sim=" in text
    assert "# installed distributions" in text


def test_the_repository_block_admits_an_uncommitted_tree() -> None:
    state = package_submission._repository_state()

    assert set(state) == {"commit", "branch", "dirty", "uncommitted_files"}
    # Either git answered, or it did not; a fabricated clean commit is the one
    # thing that must not happen.
    assert state["commit"] is None or len(state["commit"]) == 40
    assert state["dirty"] in (True, False, None)


@pytest.mark.skipif(
    not asset_available(FRANKA_PANDA),
    reason="the Franka Panda asset is not installed; see docs/ROBOT_LAYER.md",
)
def test_a_whole_package_is_written_and_is_internally_consistent(tmp_path: Path) -> None:
    from multisport_sim.benchmark.robots.panda import PANDA_READY_QPOS

    module = tmp_path / "held_policy.py"
    module.write_text(
        "import numpy as np\n"
        "from multisport_sim.benchmark.robots.panda import PANDA_READY_QPOS\n"
        "READY = np.asarray(PANDA_READY_QPOS, dtype=np.float32)\n"
        "def load_policy():\n"
        "    return lambda obs: READY\n",
        encoding="utf-8",
    )
    weights = tmp_path / "weights.bin"
    weights.write_bytes(bytes(len(PANDA_READY_QPOS)))
    sys.path.insert(0, str(tmp_path))
    out = tmp_path / "submission"
    try:
        exit_code = package_submission.main(
            [
                "--policy",
                "held_policy:load_policy",
                "--policy-id",
                "held-pose",
                "--split",
                "dev",
                "--levels",
                "L1",
                "--policy-file",
                str(weights),
                "--videos",
                "1",
                "--out",
                str(out),
            ]
        )
    finally:
        sys.path.remove(str(tmp_path))

    assert exit_code == 0
    for relative in (
        "manifest.json",
        "metrics.json",
        "environment.txt",
        "config/task_config.json",
        "config/run.json",
        "policy/hashes.json",
        "policy/weights.bin",
    ):
        assert (out / relative).is_file(), relative

    manifest = json.loads((out / "manifest.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())

    assert manifest["schema"] == "multisport-submission-v0"
    assert manifest["task"] == "table-tennis-return-panda-v1"
    # The bank is an experimental fixture; a complete package does not change that.
    assert manifest["leaderboard_eligible"] is False
    assert manifest["track"]["id"] == "state"
    assert manifest["policy"]["complete"] is True
    assert manifest["reproduce_command"].startswith("python scripts/package_submission.py")
    assert metrics["levels"]["L1"]["assessment"]["primary_metric"] == "hit_rate"
    assert metrics["levels"]["L1"]["results"]
    # One level cannot show a robustness gap.
    assert metrics["robustness_gap"] is None

    if manifest["videos"]["status"] == "written":
        for name in manifest["videos"]["files"]:
            assert (out / "videos" / name).stat().st_size > 0
    else:
        assert manifest["videos"]["status"] in {"skipped", "empty"}
        assert manifest["videos"]["reason"]
