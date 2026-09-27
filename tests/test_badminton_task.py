"""The badminton serve task: the first launch task on the shared framework.

A serve reverses the return tasks' episode -- the shuttle waits on the robot's
side to be struck -- while keeping their action, observation, report schema
and level names.  These tests pin what had to change to make that physical:
a 0.5 ms step, a face whose pose is interpolated across the control period,
and a shuttle whose skirt drag is evaluated at its centre of pressure.
"""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from multisport_sim.benchmark import registry
from multisport_sim.benchmark.backends.mujoco import PROFILES, MujocoShotBackend
from multisport_sim.benchmark.controllers import PaddleCommand, ScriptedServeController
from multisport_sim.benchmark.rules.badminton import BadmintonServeJudge
from multisport_sim.benchmark.runner import RunConfig, run_shots
from multisport_sim.benchmark.shot_bank import ShotBank
from multisport_sim.benchmark.task_config import BADMINTON_SERVE_V0
from multisport_sim.benchmark.tasks import register_builtin_tasks
from multisport_sim.physics import Aerodynamics, Atmosphere
from multisport_sim.scene import BENCHMARK_EFFECTORS, build_model
from multisport_sim.specs import Sport

register_builtin_tasks()


def test_only_the_benchmark_scene_uses_the_short_step() -> None:
    """The regulation drop tests keep the 1 ms step they were calibrated at."""
    assert build_model("badminton", benchmark_paddle=True).opt.timestep == 0.0005
    assert build_model("badminton").opt.timestep == 0.001
    assert build_model("tennis", benchmark_paddle=True).opt.timestep == 0.001


def test_the_racket_is_parked_inside_its_own_action_space() -> None:
    parked = BENCHMARK_EFFECTORS[Sport.BADMINTON].parked_at
    workspace = BADMINTON_SERVE_V0.workspace
    assert all(
        low <= value <= high
        for value, low, high in zip(parked, workspace.position_low, workspace.position_high)
    )


def test_the_profile_maps_the_court_and_net() -> None:
    semantic = PROFILES[Sport.BADMINTON].semantic_geom_names()
    assert semantic["badminton_benchmark_paddle_blade"] == "robot_racket"
    assert semantic["badminton_surface"] == "table"
    assert semantic["badminton_net"] == "net"


def test_the_face_is_interpolated_across_the_control_period() -> None:
    backend = MujocoShotBackend(sport="badminton")
    start = backend.observe().paddle_position
    goal = (start[0] + 0.10, start[1], start[2])
    backend.apply_action(PaddleCommand(position=goal, quaternion=(1.0, 0.0, 0.0, 0.0)))
    backend.step()
    first = backend.observe().paddle_position
    # Ten 0.5 ms steps per 5 ms control period: one tenth of the way.
    assert first[0] - start[0] == pytest.approx(0.01)
    for _ in range(12):
        backend.step()
    assert backend.observe().paddle_position[0] == pytest.approx(goal[0])


def test_other_sports_still_teleport_their_face() -> None:
    backend = MujocoShotBackend(sport="tennis")
    start = backend.observe().paddle_position
    goal = (start[0] + 0.10, start[1], start[2])
    backend.apply_action(PaddleCommand(position=goal, quaternion=(1.0, 0.0, 0.0, 0.0)))
    assert backend.observe().paddle_position[0] == pytest.approx(goal[0])


def test_a_tumbling_shuttle_rights_itself_instead_of_blowing_up() -> None:
    """Skirt drag at the centre of pressure damps pitch; at the CoM it diverged."""
    model = build_model("badminton")
    data = mujoco.MjData(model)
    aerodynamics = Aerodynamics(model, Atmosphere())
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "badminton_ball")
    joint = model.body_jntadr[body]
    qpos, dof = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    data.qpos[qpos : qpos + 3] = (0.0, 0.0, 6.0)
    # Skirt-first, i.e. backwards, at a smash speed and spinning hard.
    data.qpos[qpos + 3 : qpos + 7] = (0.0, 1.0, 0.0, 0.0)
    data.qvel[dof : dof + 3] = (30.0, 0.0, 0.0)
    data.qvel[dof + 3 : dof + 6] = (0.0, 150.0, 40.0)
    mujoco.mj_forward(model, data)
    for _ in range(300):
        data.xfrc_applied[:] = 0.0
        aerodynamics.apply(data)
        mujoco.mj_step(model, data)
        assert np.all(np.isfinite(data.qvel))
    axis = data.xmat[body].reshape(3, 3)[:, 2]
    velocity = data.cvel[body, 3:6]
    # Cork first: the cork-to-skirt axis points against the flow.
    assert float(axis @ velocity) / np.linalg.norm(velocity) < -0.9


def test_the_task_is_registered_and_addressable() -> None:
    entry = registry.get_task("badminton-serve-v0")
    assert entry.config is BADMINTON_SERVE_V0
    assert entry.env_entry_point.endswith(":BadmintonServeEnv")
    assert isinstance(entry.make_judge(), BadmintonServeJudge)
    assert isinstance(entry.make_backend(), MujocoShotBackend)


def test_the_bank_starts_every_shot_on_the_servers_side() -> None:
    bank = ShotBank.from_resource(split="test", task="badminton/serve-v0")
    assert len(bank) == 600
    assert all(shot.sport == "badminton" for shot in bank)
    assert all(shot.position[0] < -1.98 for shot in bank)
    assert all(abs(shot.position[1]) <= 2.59 for shot in bank)
    for shot in bank.filter(level="L3"):
        # The target is in the court diagonally opposite the server.
        assert shot.target is not None
        assert shot.target.center_xy[0] > 1.98
        assert shot.target.center_xy[1] * shot.position[1] < 0.0


def test_the_environment_matches_the_shared_contract() -> None:
    from gymnasium.utils.env_checker import check_env

    from multisport_sim.benchmark import register_envs

    register_envs()
    env = gym.make("MultiSportRobot/BadmintonServe-v0").unwrapped
    try:
        check_env(env, skip_render_check=True)
        observation, info = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        assert info["task_id"] == "badminton-serve-v0"
        assert info["control_dt"] == pytest.approx(0.005)
        assert observation.shape == (BADMINTON_SERVE_V0.OBSERVATION_DIM,)
    finally:
        env.close()


def test_the_scripted_fixture_serves_legally_and_idles_at_l0() -> None:
    bank = ShotBank.from_resource(split="dev", task="badminton/serve-v0")
    backend = MujocoShotBackend(sport="badminton")
    task = replace(BADMINTON_SERVE_V0, split="dev")
    controller = ScriptedServeController()

    serves = run_shots(
        backend, controller, bank.filter(level="L2")[:8], config=RunConfig.from_task_config(task)
    )
    assert sum(result.valid_return for result in serves.results) >= 7
    assert all(
        result.outgoing_speed_mps is not None and result.outgoing_speed_mps > 10.0
        for result in serves.results
    )
    physics = run_shots(
        backend, controller, bank.filter(level="L0")[:4], config=RunConfig.from_task_config(task)
    )
    assert all(result.incoming_valid and not result.hit for result in physics.results)


def test_the_serve_carry_tables_prefer_the_high_serve() -> None:
    controller = ScriptedServeController()
    assert controller.elevation_for_carry(6.0) == 45.0
    assert controller.elevation_for_carry(8.3) == 35.0
    assert controller.speed_for_carry(3.0) == 8.0
    assert controller.speed_for_carry(20.0) == 24.0
