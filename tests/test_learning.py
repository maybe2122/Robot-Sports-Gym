"""The learned-baseline plumbing: primitive, one-decision env, and committed weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from multisport_sim.benchmark.learning import (
    FEATURES,
    SPECS,
    PrimitiveLaunchEnv,
    SwingPrimitive,
    _scale,
    launch_reward,
)
from multisport_sim.benchmark.types import EpisodeResult

WEIGHTS = Path(__file__).resolve().parents[1] / "baselines" / "learned"


def test_actions_map_linearly_onto_the_primitive_ranges() -> None:
    assert _scale(-1.0, (8.0, 24.0)) == 8.0
    assert _scale(1.0, (8.0, 24.0)) == 24.0
    assert _scale(0.0, (8.0, 24.0)) == 16.0
    assert _scale(7.0, (8.0, 24.0)) == 24.0  # clipped, never extrapolated


@pytest.mark.parametrize("sport", sorted(SPECS))
def test_the_one_decision_env_matches_the_gymnasium_contract(sport: str) -> None:
    from gymnasium.utils.env_checker import check_env

    env = PrimitiveLaunchEnv(sport)
    check_env(env, skip_render_check=True)
    features, info = env.reset(seed=3)
    assert features.shape == (len(FEATURES),)
    # Normalised: every feature of a real shot is within a small multiple of one.
    assert np.all(np.abs(features) < 5.0)
    _, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated and not truncated
    assert 0.0 <= reward <= 2.6
    assert info["episode_result"]["shot_id"]


def test_the_reward_ranks_success_then_placement() -> None:
    miss = EpisodeResult("m", level="L3")
    hit = EpisodeResult("h", level="L3", hit=True)
    near = EpisodeResult("n", level="L3", hit=True, valid_return=True, target_error_m=0.8)
    inside = EpisodeResult(
        "i", level="L3", hit=True, valid_return=True, target_hit=True, target_error_m=0.1
    )
    rewards = [launch_reward(r, target_radius_m=0.5) for r in (miss, hit, near, inside)]
    assert rewards == sorted(rewards)
    assert rewards[0] == 0.0


def test_the_primitive_idles_at_l0() -> None:
    from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
    from multisport_sim.benchmark.shot_bank import ShotBank

    shot = ShotBank.from_resource(split="dev", task="football/kick-v0").filter(level="L0")[0]
    primitive = SwingPrimitive(SPECS["football"])
    backend = MujocoShotBackend(sport="football")
    backend.reset()
    backend.launch_ball(shot)
    primitive.reset(shot)
    assert primitive.act(backend.observe()) is None


@pytest.mark.parametrize("sport", sorted(SPECS))
def test_committed_weights_load_and_play(sport: str) -> None:
    pytest.importorskip("stable_baselines3")
    weights = WEIGHTS / sport / "seed0.zip"
    if not weights.exists():
        pytest.skip(f"no committed weights for {sport}")
    from dataclasses import replace

    from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
    from multisport_sim.benchmark.learning import load_learned_controller
    from multisport_sim.benchmark.registry import get_task
    from multisport_sim.benchmark.runner import RunConfig, run_shots
    from multisport_sim.benchmark.shot_bank import ShotBank
    from multisport_sim.benchmark.tasks import register_builtin_tasks

    register_builtin_tasks()
    spec = SPECS[sport]
    task = replace(get_task(spec.task_id).config, split="test")
    shots = ShotBank.from_resource(split="test", task=spec.bank).filter(level="L2")[:3]
    output = run_shots(
        MujocoShotBackend(sport=sport),
        load_learned_controller(sport, str(weights)),
        shots,
        config=RunConfig.from_task_config(task),
    )
    assert all(result.hit for result in output.results)
