"""The football kick and basketball shoot tasks, end to end on MuJoCo.

Both put the task frame's origin at their goal -- the goal line, the floor
under the rim -- so that the robot is at x < 0 and launches toward +x like
every other task.  These tests pin that frame, the banks, the environments,
the fixtures' measured contact law and the solver choice it depends on.
"""

from __future__ import annotations

from dataclasses import replace
from math import cos, radians, sin

import gymnasium as gym
import numpy as np
import pytest

from multisport_sim.benchmark import registry
from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
from multisport_sim.benchmark.controllers import PaddleCommand, _quaternion_taking_y_to
from multisport_sim.benchmark.launch_controllers import (
    IMPULSE_GAIN,
    ScriptedKickController,
    ScriptedShootController,
)
from multisport_sim.benchmark.rules.basketball import BasketballShootJudge
from multisport_sim.benchmark.rules.football import FootballKickJudge
from multisport_sim.benchmark.runner import RunConfig, run_shots
from multisport_sim.benchmark.shot_bank import ShotBank
from multisport_sim.benchmark.task_config import BASKETBALL_SHOOT_V0, FOOTBALL_KICK_V0
from multisport_sim.benchmark.tasks import register_builtin_tasks
from multisport_sim.benchmark.types import ShotSpec
from multisport_sim.scene import build_model

register_builtin_tasks()


@pytest.mark.parametrize(
    ("sport", "origin"), [("football", (52.5, 0.0, 0.0)), ("basketball", (12.425, 0.0, 0.0))]
)
def test_the_backend_speaks_only_task_coordinates(sport, origin) -> None:
    backend = MujocoShotBackend(sport=sport)
    assert backend.task_frame.origin_xyz == origin
    shot = ShotSpec("frame", sport, "L2", (-4.0, 1.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    backend.reset()
    backend.launch_ball(shot)
    assert backend.get_ball_state().position == pytest.approx((-4.0, 1.0, 2.0))
    body = backend._ball_body_id
    assert backend.data.xpos[body] == pytest.approx(np.add((-4.0, 1.0, 2.0), origin), abs=1e-3)
    command = PaddleCommand(position=(-3.0, 0.5, 1.0), quaternion=(1.0, 0.0, 0.0, 0.0))
    backend.apply_action(command)
    for _ in range(10):
        backend.step()
    assert backend.observe().paddle_position == pytest.approx((-3.0, 0.5, 1.0))


def test_the_net_sports_keep_the_identity_frame() -> None:
    assert MujocoShotBackend(sport="tennis").task_frame.is_identity


def test_only_the_football_benchmark_scene_uses_pgs() -> None:
    import mujoco

    assert build_model("football", benchmark_paddle=True).opt.solver == mujoco.mjtSolver.mjSOL_PGS
    assert build_model("football").opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert build_model("tennis", benchmark_paddle=True).opt.solver == mujoco.mjtSolver.mjSOL_NEWTON


@pytest.mark.parametrize("yaw_degrees", [0.0, 60.0, 90.0])
def test_a_kick_leaves_along_the_boot_normal_in_every_direction(yaw_degrees) -> None:
    """Under Newton the pitch-and-boot contact pair jammed off-axis (4-7 deg, 20-30 rad/s)."""
    yaw, elevation = radians(yaw_degrees), radians(17.0)
    normal = np.array([cos(elevation) * cos(yaw), cos(elevation) * sin(yaw), sin(elevation)])
    quaternion = _quaternion_taking_y_to(tuple(normal))
    backend = MujocoShotBackend(sport="football")
    start = np.array((-16.0, 0.0, 0.11))
    backend.reset()
    backend.launch_ball(ShotSpec("k", "football", "L2", tuple(start), (0, 0, 0), (0, 0, 0)))
    face, speed, touched = start - normal * 0.14, 16.0, False
    for step in range(600):
        if step % 5 == 0:
            travel = min(max(speed * (backend.time + 0.005 - 0.3), -0.4), 0.1)
            backend.apply_action(PaddleCommand(tuple(face + normal * travel), quaternion))
        backend.step()
        touching = any("robot_racket" in c.pair for c in backend.semantic_contacts())
        touched |= touching
        if touched and not touching:
            break
    ball = backend.get_ball_state()
    velocity = np.asarray(ball.linear_velocity)
    angle = np.degrees(np.arccos(velocity @ normal / np.linalg.norm(velocity)))
    assert angle < 0.5
    assert np.linalg.norm(ball.angular_velocity) < 1.0
    assert np.linalg.norm(velocity) / speed == pytest.approx(IMPULSE_GAIN, abs=0.02)


@pytest.mark.parametrize(
    ("task_id", "judge_type", "bank_task"),
    [
        ("football-kick-v0", FootballKickJudge, "football/kick-v0"),
        ("basketball-shoot-v0", BasketballShootJudge, "basketball/shoot-v0"),
    ],
)
def test_the_tasks_are_registered_with_their_banks(task_id, judge_type, bank_task) -> None:
    entry = registry.get_task(task_id)
    assert isinstance(entry.make_judge(), judge_type)
    assert entry.config.bank_resource == bank_task
    bank = ShotBank.from_resource(split="test", task=bank_task)
    assert len(bank) == 600
    assert all(shot.position[0] < 0.0 for shot in bank)
    assert bank.manifest["targets"]["placement_plane"]["axes"] in (["y", "z"], ["x", "y"])


def test_moving_football_starts_rolling_not_sliding() -> None:
    bank = ShotBank.from_resource(split="dev", task="football/kick-v0")
    moving = [shot for shot in bank if any(shot.linear_velocity)]
    assert moving
    for shot in moving:
        vx, vy, _ = shot.linear_velocity
        wx, wy, _ = shot.angular_velocity
        assert wx == pytest.approx(-vy / 0.11, abs=1e-3)
        assert wy == pytest.approx(vx / 0.11, abs=1e-3)


@pytest.mark.parametrize(
    "env_id", ["MultiSportRobot/FootballKick-v0", "MultiSportRobot/BasketballShoot-v0"]
)
def test_the_environments_match_the_shared_contract(env_id) -> None:
    from gymnasium.utils.env_checker import check_env

    from multisport_sim.benchmark import register_envs

    register_envs()
    env = gym.make(env_id).unwrapped
    try:
        check_env(env, skip_render_check=True)
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        assert observation[0] < 0.0  # the ball starts on the robot's side of the goal
    finally:
        env.close()


def test_the_kick_fixture_scores_and_places() -> None:
    bank = ShotBank.from_resource(split="dev", task="football/kick-v0")
    task = replace(FOOTBALL_KICK_V0, split="dev")
    output = run_shots(
        MujocoShotBackend(sport="football"),
        ScriptedKickController(),
        bank.filter(level="L3")[:6],
        config=RunConfig.from_task_config(task),
    )
    assert all(result.valid_return for result in output.results)
    assert sum(result.target_hit for result in output.results) >= 5


def test_the_shoot_fixture_makes_baskets() -> None:
    bank = ShotBank.from_resource(split="dev", task="basketball/shoot-v0")
    task = replace(BASKETBALL_SHOOT_V0, split="dev")
    output = run_shots(
        MujocoShotBackend(sport="basketball"),
        ScriptedShootController(),
        bank.filter(level="L2")[:6],
        config=RunConfig.from_task_config(task),
    )
    assert sum(result.valid_return for result in output.results) >= 5
    idle = run_shots(
        MujocoShotBackend(sport="basketball"),
        ScriptedShootController(),
        bank.filter(level="L0")[:3],
        config=RunConfig.from_task_config(task),
    )
    assert all(r.incoming_valid and not r.hit for r in idle.results)
