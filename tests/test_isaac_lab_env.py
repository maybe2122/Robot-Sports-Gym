from __future__ import annotations

import subprocess
import sys

import pytest


def test_importing_benchmark_does_not_require_isaac_lab() -> None:
    """The MuJoCo path must stay importable on a machine without Isaac."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, multisport_sim.benchmark; "
                "print(any(name.startswith('isaaclab') for name in sys.modules))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "False"


def test_isaac_lab_env_cfg_derives_everything_from_the_shared_task_config() -> None:
    pytest.importorskip("isaaclab", reason="Isaac Lab runtime is not installed")
    from multisport_sim.benchmark.backends.isaac_lab import (
        PHYSICS_DT,
        SEMANTIC_FILTERS,
        make_env_cfg,
    )
    from multisport_sim.benchmark.events import BALL
    from multisport_sim.benchmark.task_config import TABLE_TENNIS_RETURN_V0

    cfg = make_env_cfg(num_envs=4, device="cpu", split="test")

    assert cfg.scene.num_envs == 4
    assert cfg.task.split == "test"
    assert cfg.sim.dt == PHYSICS_DT
    assert cfg.decimation == cfg.task.decimation(PHYSICS_DT)
    assert cfg.episode_length_s == TABLE_TENNIS_RETURN_V0.timeout_s
    # Contacts are filtered into the same vocabulary the judge already speaks.
    names = [name for name, _ in SEMANTIC_FILTERS]
    assert BALL not in names
    assert set(names) == {"table", "net", "floor", "robot_racket"}
    assert [path for _, path in SEMANTIC_FILTERS] == list(
        cfg.scene.ball_contacts.filter_prim_paths_expr
    )
