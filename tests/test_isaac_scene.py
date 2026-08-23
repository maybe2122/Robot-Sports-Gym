from __future__ import annotations

from math import sqrt

import pytest

from multisport_sim.isaac_scene import build_isaac_scene_spec
from multisport_sim.scene import SCENES
from multisport_sim.specs import Sport


@pytest.mark.parametrize("scene", SCENES)
def test_every_isaac_scene_has_geometry_and_the_expected_balls(scene: str) -> None:
    spec = build_isaac_scene_spec(scene)
    expected_balls = len(Sport) if scene == "campus" else 1
    assert len(spec.balls) == expected_balls
    assert len(spec.primitives) > 10
    assert spec.camera_eye != spec.camera_target


def test_isaac_campus_contains_all_sports_and_equipment() -> None:
    spec = build_isaac_scene_spec("campus")
    assert {ball.sport for ball in spec.balls} == set(Sport)
    names = {primitive.name for primitive in spec.primitives}
    required = {
        "tennis/a_racket_handle",
        "table_tennis/table",
        "table_tennis/a_paddle_blade",
        "table_tennis/table_edge_north",
        "table_tennis/table_tennis_net_top_tape",
        "football/goal_w_crossbar",
        "badminton/a_racket_handle",
        "basketball/west_backboard",
        "basketball/west_board_top",
        "basketball/west_rim_0",
        "basketball/west_net_strand_0_a",
    }
    assert required <= names


def test_regulation_table_and_basket_details_are_visible() -> None:
    table = {item.name: item for item in build_isaac_scene_spec("table_tennis").primitives}
    assert table["table_tennis/table"].size == pytest.approx((2.74, 1.525, 0.04))
    assert table["table_tennis/table_tennis_net"].size == pytest.approx((0.012, 1.83, 0.1525))
    assert table["table_tennis/table_tennis_net_top_tape"].opacity == 1.0

    basket = {item.name: item for item in build_isaac_scene_spec("basketball").primitives}
    assert basket["basketball/west_backboard"].size == pytest.approx((0.07, 1.80, 1.05))
    assert basket["basketball/west_backboard"].opacity >= 0.8
    assert not basket["basketball/west_net_strand_0_a"].collision

    squash = {item.name: item for item in build_isaac_scene_spec("squash").primitives}
    assert squash["squash/front_wall"].size == pytest.approx((0.10, 6.40, 4.57))
    assert squash["squash/back_wall"].size == pytest.approx((0.10, 6.40, 2.13))
    assert not squash["squash/tin"].collision
    assert not squash["squash/player_a_handle"].collision
    assert not squash["squash/player_b_hoop_0"].collision
    assert squash["squash/score_a_top"].position[2] == pytest.approx(2.65)
    assert squash["squash/score_b_bottom"].position[2] == pytest.approx(2.05)


def test_isaac_collision_and_visual_layers_are_distinct() -> None:
    spec = build_isaac_scene_spec("tennis")
    primitives = {primitive.name: primitive for primitive in spec.primitives}
    assert primitives["tennis/surface"].collision
    assert not primitives["tennis/doubles_north"].collision
    assert primitives["tennis/tennis_net"].opacity < 0.5


def test_capsule_orientations_are_unit_quaternions() -> None:
    spec = build_isaac_scene_spec("basketball")
    for primitive in spec.primitives:
        if primitive.kind == "capsule":
            norm = sqrt(sum(value * value for value in primitive.orientation))
            assert norm == pytest.approx(1.0, abs=1e-6)


def test_unknown_isaac_scene_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        build_isaac_scene_spec("baseball")
