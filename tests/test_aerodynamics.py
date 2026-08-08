from __future__ import annotations

import mujoco
import numpy as np
import pytest

from multisport_sim.physics import Aerodynamics, quadratic_drag
from multisport_sim.scene import build_model
from multisport_sim.simulation import Simulation
from multisport_sim.specs import BALLS, Sport


def test_quadratic_drag_opposes_motion_and_scales_with_speed_squared() -> None:
    slow = quadratic_drag(np.array([2.0, 0.0, 0.0]), 0.01, 0.5)
    fast = quadratic_drag(np.array([4.0, 0.0, 0.0]), 0.01, 0.5)
    assert slow[0] < 0
    assert np.linalg.norm(fast) == pytest.approx(4 * np.linalg.norm(slow))
    assert quadratic_drag(np.zeros(3), 0.01, 0.5).tolist() == [0.0, 0.0, 0.0]


def test_shuttle_drag_produces_much_larger_deceleration_than_football() -> None:
    accelerations = {}
    for sport in (Sport.BADMINTON, Sport.FOOTBALL):
        model = build_model(sport.value)
        data = mujoco.MjData(model)
        aero = Aerodynamics(model)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{sport.value}_ball")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{sport.value}_ball_free")
        dof = model.jnt_dofadr[joint_id]
        data.qvel[dof : dof + 3] = (20.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        aero.apply(data)
        accelerations[sport] = np.linalg.norm(data.xfrc_applied[body_id, :3]) / BALLS[sport].mass

    assert accelerations[Sport.BADMINTON] > 5 * accelerations[Sport.FOOTBALL]


def test_aerodynamics_only_applies_to_balls_present_in_scene() -> None:
    model = build_model("tennis")
    aero = Aerodynamics(model)
    assert list(aero._body_ids) == [Sport.TENNIS]


def test_shuttle_approaches_a_plausible_terminal_speed() -> None:
    simulation = Simulation("badminton")
    joint_id = mujoco.mj_name2id(
        simulation.model, mujoco.mjtObj.mjOBJ_JOINT, "badminton_ball_free"
    )
    qpos = simulation.model.jnt_qposadr[joint_id]
    dof = simulation.model.jnt_dofadr[joint_id]
    simulation.data.qpos[qpos : qpos + 3] = (0.0, 0.0, 60.0)
    simulation.data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(simulation.model, simulation.data)
    simulation.step(4_500)
    terminal_speed = abs(float(simulation.data.qvel[dof + 2]))
    assert 5.5 < terminal_speed < 7.2
