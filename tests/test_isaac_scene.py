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
        "football/goal_w_crossbar",
        "badminton/a_racket_handle",
        "basketball/west_backboard",
        "basketball/west_rim_0",
    }
    assert required <= names


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

