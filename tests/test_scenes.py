from __future__ import annotations

import mujoco
import pytest

from multisport_sim.scene import SCENES, build_model, build_xml
from multisport_sim.specs import BALLS, COURTS, Sport


@pytest.mark.parametrize("scene", SCENES)
def test_every_scene_compiles(scene: str) -> None:
    model = build_model(scene)
    assert model.nbody > 1
    assert model.ngeom > 10
    assert model.opt.timestep == pytest.approx(0.001)


def test_campus_contains_every_ball_surface_and_equipment() -> None:
    model = build_model("campus")
    for sport in Sport:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{sport.value}_ball") > 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{sport.value}_surface") >= 0

    required_geometries = (
        "tennis_a_racket_handle",
        "table_tennis_table",
        "table_tennis_a_paddle_blade",
        "table_tennis_net_top_tape",
        "football_goal_w_crossbar",
        "badminton_a_racket_handle",
        "basketball_w_backboard",
        "basketball_w_board_top",
        "basketball_w_rim_0",
        "basketball_w_net_strand_0_a",
    )
    for name in required_geometries:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0, name


def test_ball_mass_and_radius_reach_compiled_model() -> None:
    for sport, spec in BALLS.items():
        model = build_model(sport.value)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{sport.value}_ball")
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{sport.value}_ball_geom")
        assert model.body_mass[body_id] == pytest.approx(spec.mass, rel=1e-5)
        assert model.geom_size[geom_id, 0] == pytest.approx(spec.radius)


def test_regulation_dimensions_are_embedded_in_mjcf() -> None:
    assert COURTS[Sport.TENNIS].length == 23.77
    assert COURTS[Sport.TABLE_TENNIS].width == 5.0  # surrounding play zone
    assert COURTS[Sport.FOOTBALL].length == 105.0
    assert COURTS[Sport.BADMINTON].width == 6.10
    assert COURTS[Sport.BASKETBALL].length == 28.0
    xml = build_xml("table_tennis")
    assert 'size="1.37 0.7625 0.02"' in xml
    assert "0.025 0.03" in xml


def test_unknown_scene_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        build_xml("volleyball")
