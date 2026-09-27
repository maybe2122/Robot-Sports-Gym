"""The user-facing configuration must control the actual run, with no silent defaults."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from multisport_sim import benchmark_cli as cli
from multisport_sim.benchmark.policy_eval import PolicyController
from multisport_sim.benchmark.task_config import TABLE_TENNIS_RETURN_G1_V1


def test_config_is_validated_and_cli_overrides_it(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "robot": "g1",
                "track": "vision",
                "perception": "rgbd",
                "episodes": 5,
                "rate_margin": 0.2,
            }
        )
    )
    seen = []
    monkeypatch.setattr(cli, "run_from_args", lambda args: seen.append(args) or {})
    monkeypatch.setattr(cli, "report_markdown", lambda _: "")
    assert cli.main(["--config", str(path), "--episodes", "2"]) == 0
    assert seen[0].robot == "g1"
    assert seen[0].perception == "rgbd"
    assert seen[0].episodes == 2
    assert seen[0].rate_margin == 0.2
    path.write_text('{"misspelled_camera": true}')
    assert cli.main(["--config", str(path)]) == 2
    assert len(seen) == 1


@pytest.mark.parametrize("content", ["[]", '{"episodes": true}', '{"viewer": "false"}'])
def test_malformed_config_fails_before_running(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    assert cli.main(["--config", str(path)]) == 2


def test_g1_custom_vision_policy_sees_images_and_returns_ten_joints():
    from multisport_sim.benchmark.robots.g1 import G1_READY_QPOS

    obs = SimpleNamespace(sensors="camera payload")

    def policy(value):
        assert value is obs
        assert not hasattr(value, "ball")
        return G1_READY_QPOS

    adapter = PolicyController(
        policy, policy_id="test", track="vision", task=TABLE_TENNIS_RETURN_G1_V1
    )
    adapter.reset(None, seed=1)
    assert len(adapter.act(obs).targets) == 10
    bad = PolicyController(
        lambda _: np.zeros(7),
        policy_id="wrong-robot",
        track="vision",
        task=TABLE_TENNIS_RETURN_G1_V1,
    )
    with pytest.raises(ValueError, match="10 finite"):
        bad.act(obs)
