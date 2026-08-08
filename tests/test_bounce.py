from __future__ import annotations

import mujoco
import pytest

from multisport_sim.simulation import Simulation
from multisport_sim.specs import BALLS, Sport


def first_rebound_height(sport: Sport, drop_height: float, surface_height: float = 0.0) -> float:
    """Return clearance under the ball at the first rebound apex."""
    simulation = Simulation(sport.value)
    joint_id = mujoco.mj_name2id(
        simulation.model, mujoco.mjtObj.mjOBJ_JOINT, f"{sport.value}_ball_free"
    )
    qpos = simulation.model.jnt_qposadr[joint_id]
    dof = simulation.model.jnt_dofadr[joint_id]
    simulation.data.qpos[qpos + 2] = surface_height + drop_height + BALLS[sport].radius
    simulation.data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(simulation.model, simulation.data)

    rising = False
    previous_vz = -1.0
    apex = 0.0
    for _ in range(15_000):
        simulation.step()
        vz = float(simulation.data.qvel[dof + 2])
        if not rising and previous_vz < 0.0 < vz:
            rising = True
        if rising:
            apex = max(
                apex,
                float(simulation.data.qpos[qpos + 2]) - surface_height - BALLS[sport].radius,
            )
        if rising and previous_vz > 0.0 >= vz:
            return apex
        previous_vz = vz
    raise AssertionError("ball did not complete its first rebound")


def test_tennis_regulation_rebound() -> None:
    # ITF Type 2 ball: approximately 1.35-1.47 m from a 2.54 m drop.
    assert first_rebound_height(Sport.TENNIS, 2.54) == pytest.approx(1.41, abs=0.08)


def test_table_tennis_regulation_table_rebound() -> None:
    # ITTF: about 0.23 m rebound when dropped 0.30 m onto the table.
    assert first_rebound_height(Sport.TABLE_TENNIS, 0.30, surface_height=0.76) == pytest.approx(
        0.23, abs=0.025
    )


def test_basketball_regulation_floor_rebound() -> None:
    # FIBA: underside rebounds 1.035-1.085 m from an underside drop height of 1.8 m.
    assert first_rebound_height(Sport.BASKETBALL, 1.8) == pytest.approx(1.06, abs=0.04)


def test_football_and_shuttle_have_distinct_surface_response() -> None:
    football = first_rebound_height(Sport.FOOTBALL, 2.0)
    shuttle = first_rebound_height(Sport.BADMINTON, 1.8)
    assert 0.85 < football < 1.15
    assert shuttle < 0.06

