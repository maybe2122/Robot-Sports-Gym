"""Backend-neutral scene descriptions consumed by Isaac Sim/PhysX.

This module deliberately does not import Isaac Sim. It can be unit-tested with a
normal Python interpreter; :mod:`multisport_sim.isaac_backend` turns these specs
into USD prims after SimulationApp has started.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt

from .specs import COURTS, SCENES, Sport

Color = tuple[float, float, float]
Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]

WHITE: Color = (0.96, 0.96, 0.94)
BLACK: Color = (0.06, 0.07, 0.08)


@dataclass(frozen=True)
class IsaacPrimitive:
    name: str
    kind: str
    position: Vector3
    size: tuple[float, ...]
    color: Color
    orientation: Quaternion = (1.0, 0.0, 0.0, 0.0)
    collision: bool = True
    opacity: float = 1.0
    static_friction: float = 0.7
    dynamic_friction: float = 0.6
    restitution: float = 0.0


@dataclass(frozen=True)
class IsaacBall:
    sport: Sport
    position: Vector3


@dataclass(frozen=True)
class IsaacSceneSpec:
    name: str
    primitives: tuple[IsaacPrimitive, ...]
    balls: tuple[IsaacBall, ...]
    camera_eye: Vector3
    camera_target: Vector3


def _box(
    name: str,
    position: Vector3,
    size: Vector3,
    color: Color,
    *,
    collision: bool = True,
    opacity: float = 1.0,
    friction: tuple[float, float] = (0.7, 0.6),
    restitution: float = 0.0,
) -> IsaacPrimitive:
    return IsaacPrimitive(
        name,
        "cuboid",
        position,
        size,
        color,
        collision=collision,
        opacity=opacity,
        static_friction=friction[0],
        dynamic_friction=friction[1],
        restitution=restitution,
    )


def _sphere(
    name: str, position: Vector3, radius: float, color: Color, *, collision: bool = False
) -> IsaacPrimitive:
    return IsaacPrimitive(name, "sphere", position, (radius,), color, collision=collision)


def _quat_from_z(direction: Vector3) -> Quaternion:
    length = sqrt(sum(value * value for value in direction))
    dx, dy, dz = (value / length for value in direction)
    if dz < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)
    scale = sqrt(2.0 * (1.0 + dz))
    return (scale / 2.0, -dy / scale, dx / scale, 0.0)


def _capsule_between(
    name: str,
    start: Vector3,
    end: Vector3,
    radius: float,
    color: Color,
    *,
    collision: bool = True,
    opacity: float = 1.0,
) -> IsaacPrimitive:
    direction = tuple(end[index] - start[index] for index in range(3))
    length = sqrt(sum(value * value for value in direction))
    center = tuple((start[index] + end[index]) / 2.0 for index in range(3))
    return IsaacPrimitive(
        name,
        "capsule",
        center,
        (radius, length),
        color,
        orientation=_quat_from_z(direction),
        collision=collision,
        opacity=opacity,
    )


def _cylinder(
    name: str,
    position: Vector3,
    radius: float,
    height: float,
    color: Color,
    *,
    collision: bool = True,
) -> IsaacPrimitive:
    return IsaacPrimitive(name, "cylinder", position, (radius, height), color, collision=collision)


def _line_box(
    name: str, center: tuple[float, float], half_size: tuple[float, float], z: float = 0.008
) -> IsaacPrimitive:
    return _box(
        name,
        (center[0], center[1], z),
        (2 * half_size[0], 2 * half_size[1], 0.012),
        WHITE,
        collision=False,
    )


def _rectangle_lines(
    prefix: str, x: float, y: float, length: float, width: float, line_width: float = 0.05
) -> list[IsaacPrimitive]:
    return [
        _line_box(f"{prefix}_north", (x, y + width / 2), (length / 2, line_width / 2)),
        _line_box(f"{prefix}_south", (x, y - width / 2), (length / 2, line_width / 2)),
        _line_box(f"{prefix}_east", (x + length / 2, y), (line_width / 2, width / 2)),
        _line_box(f"{prefix}_west", (x - length / 2, y), (line_width / 2, width / 2)),
    ]


def _arc_lines(
    prefix: str,
    x: float,
    y: float,
    radius: float,
    start: float = 0.0,
    stop: float = 2 * pi,
    segments: int = 36,
    width: float = 0.035,
    z: float = 0.018,
) -> list[IsaacPrimitive]:
    result = []
    for index in range(segments):
        a0 = start + (stop - start) * index / segments
        a1 = start + (stop - start) * (index + 1) / segments
        result.append(
            _capsule_between(
                f"{prefix}_{index}",
                (x + radius * cos(a0), y + radius * sin(a0), z),
                (x + radius * cos(a1), y + radius * sin(a1), z),
                width,
                WHITE,
                collision=False,
            )
        )
    return result


def _surface(sport: Sport, x: float, y: float) -> IsaacPrimitive:
    court = COURTS[sport]
    friction = {
        Sport.TENNIS: (0.78, 0.68),
        Sport.TABLE_TENNIS: (0.65, 0.55),
        Sport.FOOTBALL: (0.90, 0.78),
        Sport.BADMINTON: (0.72, 0.62),
        Sport.BASKETBALL: (0.82, 0.70),
    }[sport]
    return _box(
        "surface",
        (x, y, -0.04),
        (court.length, court.width, 0.08),
        court.color[:3],
        friction=friction,
    )


def _net(
    prefix: str, x: float, y: float, span: float, height: float, bottom: float = 0.0
) -> list[IsaacPrimitive]:
    return [
        _box(
            f"{prefix}_net",
            (x, y, (height + bottom) / 2),
            (0.024, span, height - bottom),
            (0.10, 0.13, 0.16),
            opacity=0.38,
            friction=(0.3, 0.25),
        ),
        _cylinder(f"{prefix}_post_a", (x, y - span / 2, height / 2), 0.035, height, BLACK),
        _cylinder(f"{prefix}_post_b", (x, y + span / 2, height / 2), 0.035, height, BLACK),
    ]


def _racket(
    prefix: str, x: float, y: float, z: float, scale: float, color: Color
) -> list[IsaacPrimitive]:
    result = [
        _capsule_between(
            f"{prefix}_racket_handle",
            (x, y, z - 0.34 * scale),
            (x, y, z - 0.04 * scale),
            0.025 * scale,
            (0.18, 0.12, 0.07),
        )
    ]
    rx, rz, center_z = 0.16 * scale, 0.21 * scale, z + 0.17 * scale
    for index in range(24):
        a0, a1 = 2 * pi * index / 24, 2 * pi * (index + 1) / 24
        result.append(
            _capsule_between(
                f"{prefix}_racket_hoop_{index}",
                (x + rx * cos(a0), y, center_z + rz * sin(a0)),
                (x + rx * cos(a1), y, center_z + rz * sin(a1)),
                0.011 * scale,
                color,
            )
        )
    for index, fraction in enumerate((-0.65, -0.32, 0.0, 0.32, 0.65)):
        chord = rz * sqrt(1.0 - fraction * fraction)
        sx = x + rx * fraction
        result.append(
            _capsule_between(
                f"{prefix}_racket_string_v_{index}",
                (sx, y, center_z - chord),
                (sx, y, center_z + chord),
                0.0015 * scale,
                (0.9, 0.9, 0.85),
                collision=False,
            )
        )
    for index, fraction in enumerate((-0.55, -0.27, 0.0, 0.27, 0.55)):
        chord = rx * sqrt(1.0 - fraction * fraction)
        sz = center_z + rz * fraction
        result.append(
            _capsule_between(
                f"{prefix}_racket_string_h_{index}",
                (x - chord, y, sz),
                (x + chord, y, sz),
                0.0015 * scale,
                (0.9, 0.9, 0.85),
                collision=False,
            )
        )
    return result


def _paddle(prefix: str, x: float, y: float, z: float, color: Color) -> list[IsaacPrimitive]:
    return [
        # A thin sphere/ellipsoid-like blade is represented as a rounded cylinder.
        IsaacPrimitive(
            f"{prefix}_paddle_blade",
            "cylinder",
            (x, y, z + 0.08),
            (0.078, 0.016),
            color,
            orientation=(0.7071068, 0.7071068, 0.0, 0.0),
            static_friction=0.85,
            dynamic_friction=0.75,
        ),
        _capsule_between(
            f"{prefix}_paddle_handle",
            (x, y, z - 0.08),
            (x, y, z + 0.015),
            0.017,
            (0.48, 0.27, 0.10),
        ),
    ]


def _table_tennis_net(x: float, y: float) -> list[IsaacPrimitive]:
    """Create a regulation 1.83 m by 15.25 cm net with visible mesh and tape."""
    bottom, top, span = 0.76, 0.9125, 1.83
    parts = [
        # The thin panel supplies reliable ball collision; cords are visual only.
        _box(
            "table_tennis_net",
            (x, y, (bottom + top) / 2),
            (0.012, span, top - bottom),
            (0.04, 0.07, 0.11),
            opacity=0.20,
            friction=(0.30, 0.25),
        ),
        _cylinder("table_tennis_post_a", (x, y - span / 2, 0.82), 0.018, 0.31, BLACK),
        _cylinder("table_tennis_post_b", (x, y + span / 2, 0.82), 0.018, 0.31, BLACK),
        _capsule_between(
            "table_tennis_net_top_tape",
            (x, y - span / 2, top),
            (x, y + span / 2, top),
            0.010,
            WHITE,
            collision=False,
        ),
    ]
    for index in range(1, 12):
        cord_y = y - span / 2 + span * index / 12
        parts.append(
            _capsule_between(
                f"table_tennis_net_vertical_{index}",
                (x, cord_y, bottom),
                (x, cord_y, top),
                0.0022,
                (0.84, 0.87, 0.88),
                collision=False,
            )
        )
    for index in range(1, 4):
        cord_z = bottom + (top - bottom) * index / 4
        parts.append(
            _capsule_between(
                f"table_tennis_net_horizontal_{index}",
                (x, y - span / 2, cord_z),
                (x, y + span / 2, cord_z),
                0.0022,
                (0.84, 0.87, 0.88),
                collision=False,
            )
        )
    return parts


def _tennis(x: float, y: float) -> tuple[list[IsaacPrimitive], IsaacBall]:
    parts = [_surface(Sport.TENNIS, x, y)]
    parts += _rectangle_lines("doubles", x, y, 23.77, 10.97)
    parts += _rectangle_lines("singles", x, y, 23.77, 8.23)
    parts += [
        _line_box("service_e", (x + 6.40, y), (0.025, 4.115)),
        _line_box("service_w", (x - 6.40, y), (0.025, 4.115)),
        _line_box("service_center", (x, y), (6.40, 0.025)),
    ]
    parts += _net("tennis", x, y, 12.8, 0.914)
    parts += _racket("a", x - 2.0, y - 4.6, 0.38, 1.0, (0.06, 0.12, 0.19))
    parts += _racket("b", x + 2.0, y + 4.6, 0.38, 1.0, (0.70, 0.12, 0.04))
    return parts, IsaacBall(Sport.TENNIS, (x - 5.0, y, 1.4))


def _table_tennis(x: float, y: float) -> tuple[list[IsaacPrimitive], IsaacBall]:
    parts = [_surface(Sport.TABLE_TENNIS, x, y)]
    parts += _rectangle_lines("play_zone", x, y, 8.0, 5.0)
    parts += [
        _box(
            "table",
            (x, y, 0.74),
            (2.74, 1.525, 0.04),
            (0.025, 0.27, 0.52),
            friction=(0.35, 0.25),
        ),
        _line_box("table_center", (x, y), (1.37, 0.0015), z=0.766),
        _line_box("table_edge_north", (x, y + 0.7525), (1.37, 0.010), z=0.766),
        _line_box("table_edge_south", (x, y - 0.7525), (1.37, 0.010), z=0.766),
        _line_box("table_edge_east", (x + 1.36, y), (0.010, 0.7625), z=0.766),
        _line_box("table_edge_west", (x - 1.36, y), (0.010, 0.7625), z=0.766),
        _box("table_apron_north", (x, y + 0.745, 0.69), (2.66, 0.035, 0.10), BLACK),
        _box("table_apron_south", (x, y - 0.745, 0.69), (2.66, 0.035, 0.10), BLACK),
    ]
    for index, (dx, dy) in enumerate(((-1.05, -0.55), (-1.05, 0.55), (1.05, -0.55), (1.05, 0.55))):
        parts.append(_box(f"table_leg_{index}", (x + dx, y + dy, 0.35), (0.06, 0.06, 0.70), BLACK))
    parts += [
        _capsule_between(
            "table_brace_w", (x - 1.05, y - 0.55, 0.48), (x - 1.05, y + 0.55, 0.48), 0.025, BLACK
        ),
        _capsule_between(
            "table_brace_e", (x + 1.05, y - 0.55, 0.48), (x + 1.05, y + 0.55, 0.48), 0.025, BLACK
        ),
    ]
    parts += _table_tennis_net(x, y)
    parts += _paddle("a", x - 1.7, y - 0.95, 0.23, (0.75, 0.04, 0.03))
    parts += _paddle("b", x + 1.7, y + 0.95, 0.23, BLACK)
    return parts, IsaacBall(Sport.TABLE_TENNIS, (x - 0.7, y, 1.25))


def _goal(prefix: str, x: float, y: float, direction: float) -> list[IsaacPrimitive]:
    depth = 2.0 * direction
    return [
        _cylinder(f"{prefix}_post_a", (x, y - 3.66, 1.22), 0.06, 2.44, WHITE),
        _cylinder(f"{prefix}_post_b", (x, y + 3.66, 1.22), 0.06, 2.44, WHITE),
        _capsule_between(
            f"{prefix}_crossbar", (x, y - 3.66, 2.44), (x, y + 3.66, 2.44), 0.06, WHITE
        ),
        _box(
            f"{prefix}_net",
            (x + depth / 2, y, 1.22),
            (abs(depth), 7.32, 2.44),
            (0.88, 0.88, 0.88),
            opacity=0.12,
            friction=(0.2, 0.15),
        ),
    ]


def _football(x: float, y: float) -> tuple[list[IsaacPrimitive], IsaacBall]:
    parts = [_surface(Sport.FOOTBALL, x, y)]
    parts += _rectangle_lines("boundary", x, y, 105.0, 68.0, 0.10)
    parts += [_line_box("halfway", (x, y), (0.05, 34.0))]
    parts += _arc_lines("center_circle", x, y, 9.15, width=0.05)
    parts += _rectangle_lines("penalty_w", x - 44.25, y, 16.5, 40.32, 0.08)
    parts += _rectangle_lines("penalty_e", x + 44.25, y, 16.5, 40.32, 0.08)
    parts += _goal("goal_w", x - 52.5, y, -1.0)
    parts += _goal("goal_e", x + 52.5, y, 1.0)
    return parts, IsaacBall(Sport.FOOTBALL, (x - 18.0, y, 1.0))


def _badminton(x: float, y: float) -> tuple[list[IsaacPrimitive], IsaacBall]:
    parts = [_surface(Sport.BADMINTON, x, y)]
    parts += _rectangle_lines("doubles", x, y, 13.40, 6.10, 0.04)
    parts += _rectangle_lines("singles", x, y, 13.40, 5.18, 0.04)
    for dx, name in ((1.98, "short_e"), (-1.98, "short_w"), (5.94, "long_e"), (-5.94, "long_w")):
        parts.append(_line_box(name, (x + dx, y), (0.02, 3.05)))
    parts += [
        _line_box("center_e", (x + 4.34, y), (2.36, 0.02)),
        _line_box("center_w", (x - 4.34, y), (2.36, 0.02)),
    ]
    parts += _net("badminton", x, y, 6.10, 1.55)
    parts += _racket("a", x - 2.4, y - 2.7, 0.28, 0.68, BLACK)
    parts += _racket("b", x + 2.4, y + 2.7, 0.28, 0.68, (0.18, 0.36, 0.64))
    return parts, IsaacBall(Sport.BADMINTON, (x - 3.0, y, 2.2))


def _basket_net(prefix: str, rim_x: float, y: float) -> list[IsaacPrimitive]:
    """Create the hanging tapered cord net below a basketball rim."""
    parts: list[IsaacPrimitive] = []
    strands, rim_z, bottom_z = 12, 3.045, 2.62
    for index in range(strands):
        angle = 2 * pi * index / strands
        next_angle = 2 * pi * (index + 1) / strands
        parts += [
            _capsule_between(
                f"{prefix}_net_strand_{index}_a",
                (rim_x + 0.225 * cos(angle), y + 0.225 * sin(angle), rim_z),
                (rim_x + 0.13 * cos(next_angle), y + 0.13 * sin(next_angle), bottom_z),
                0.004,
                WHITE,
                collision=False,
            ),
            _capsule_between(
                f"{prefix}_net_strand_{index}_b",
                (rim_x + 0.225 * cos(next_angle), y + 0.225 * sin(next_angle), rim_z),
                (rim_x + 0.13 * cos(angle), y + 0.13 * sin(angle), bottom_z),
                0.004,
                WHITE,
                collision=False,
            ),
        ]
    for tier, (radius, z) in enumerate(((0.19, 2.90), (0.16, 2.76), (0.13, bottom_z))):
        for index in range(strands):
            a0, a1 = 2 * pi * index / strands, 2 * pi * (index + 1) / strands
            parts.append(
                _capsule_between(
                    f"{prefix}_net_ring_{tier}_{index}",
                    (rim_x + radius * cos(a0), y + radius * sin(a0), z),
                    (rim_x + radius * cos(a1), y + radius * sin(a1), z),
                    0.0035,
                    WHITE,
                    collision=False,
                )
            )
    return parts


def _basket(prefix: str, x: float, y: float, facing: float) -> list[IsaacPrimitive]:
    # FIBA geometry: board face is 1.20 m in court from the end line and the
    # rim center is another 0.375 m in front of the board.
    board_x = x + facing * 1.20
    rim_x = x + facing * 1.575
    support_x = x - facing * 0.45
    front_x = board_x + facing * 0.041
    parts = [
        _cylinder(f"{prefix}_support", (support_x, y, 1.6), 0.10, 3.2, BLACK),
        _box(f"{prefix}_support_base", (support_x, y, 0.12), (0.75, 1.05, 0.24), BLACK),
        _capsule_between(f"{prefix}_arm", (support_x, y, 3.15), (board_x, y, 3.15), 0.07, BLACK),
        _box(
            f"{prefix}_backboard",
            (board_x, y, 3.40),
            (0.07, 1.80, 1.05),
            (0.82, 0.91, 0.96),
            opacity=0.82,
        ),
        _box(
            f"{prefix}_board_top", (board_x, y, 3.91), (0.085, 1.80, 0.045), WHITE, collision=False
        ),
        _box(
            f"{prefix}_board_bottom",
            (board_x, y, 2.89),
            (0.085, 1.80, 0.045),
            WHITE,
            collision=False,
        ),
        _box(
            f"{prefix}_board_left",
            (board_x, y - 0.88, 3.40),
            (0.085, 0.045, 1.05),
            WHITE,
            collision=False,
        ),
        _box(
            f"{prefix}_board_right",
            (board_x, y + 0.88, 3.40),
            (0.085, 0.045, 1.05),
            WHITE,
            collision=False,
        ),
        _box(
            f"{prefix}_target_top", (front_x, y, 3.65), (0.012, 0.59, 0.035), WHITE, collision=False
        ),
        _box(
            f"{prefix}_target_bottom",
            (front_x, y, 3.20),
            (0.012, 0.59, 0.035),
            WHITE,
            collision=False,
        ),
        _box(
            f"{prefix}_target_left",
            (front_x, y - 0.2775, 3.425),
            (0.012, 0.035, 0.45),
            WHITE,
            collision=False,
        ),
        _box(
            f"{prefix}_target_right",
            (front_x, y + 0.2775, 3.425),
            (0.012, 0.035, 0.45),
            WHITE,
            collision=False,
        ),
    ]
    for index in range(32):
        a0, a1 = 2 * pi * index / 32, 2 * pi * (index + 1) / 32
        parts.append(
            _capsule_between(
                f"{prefix}_rim_{index}",
                (rim_x + 0.225 * cos(a0), y + 0.225 * sin(a0), 3.05),
                (rim_x + 0.225 * cos(a1), y + 0.225 * sin(a1), 3.05),
                0.01,
                (0.95, 0.23, 0.03),
            )
        )
    parts += _basket_net(prefix, rim_x, y)
    return parts


def _basketball(x: float, y: float) -> tuple[list[IsaacPrimitive], IsaacBall]:
    parts = [_surface(Sport.BASKETBALL, x, y)]
    parts += _rectangle_lines("boundary", x, y, 28.0, 15.0, 0.05)
    parts += [_line_box("halfway", (x, y), (0.025, 7.5))]
    parts += _arc_lines("center_circle", x, y, 1.80, width=0.025)
    for side, dx in (("w", -11.1), ("e", 11.1)):
        parts += _rectangle_lines(f"key_{side}", x + dx, y, 5.8, 4.9, 0.04)
        parts += _arc_lines(f"free_throw_{side}", x + dx, y, 1.80, width=0.025)
    parts += _arc_lines("three_w", x - 12.425, y, 6.75, -1.20, 1.20, 24, 0.025)
    parts += _arc_lines("three_e", x + 12.425, y, 6.75, pi - 1.20, pi + 1.20, 24, 0.025)
    parts += _basket("west", x - 14.0, y, 1.0)
    parts += _basket("east", x + 14.0, y, -1.0)
    return parts, IsaacBall(Sport.BASKETBALL, (x - 4.0, y, 1.5))


BUILDERS = {
    Sport.TENNIS: _tennis,
    Sport.TABLE_TENNIS: _table_tennis,
    Sport.FOOTBALL: _football,
    Sport.BADMINTON: _badminton,
    Sport.BASKETBALL: _basketball,
}


def build_isaac_scene_spec(scene: str = "campus") -> IsaacSceneSpec:
    if scene not in SCENES:
        raise ValueError(f"unknown scene {scene!r}; expected one of {', '.join(SCENES)}")

    if scene == "campus":
        sports = list(Sport)
        offsets = {
            Sport.FOOTBALL: (0.0, 0.0),
            Sport.TENNIS: (-41.0, 48.0),
            Sport.TABLE_TENNIS: (-15.0, 48.0),
            Sport.BADMINTON: (2.0, 48.0),
            Sport.BASKETBALL: (27.0, 48.0),
        }
        camera_eye, camera_target = (-92.0, -100.0, 112.0), (0.0, 14.0, 0.0)
    else:
        sport = Sport(scene)
        sports = [sport]
        offsets = {sport: (0.0, 0.0)}
        camera = {
            Sport.TENNIS: ((-15.0, -15.0, 10.0), (0.0, 0.0, 0.0)),
            Sport.TABLE_TENNIS: ((-4.5, -4.5, 3.3), (0.0, 0.0, 0.65)),
            Sport.FOOTBALL: ((-60.0, -50.0, 55.0), (0.0, 0.0, 0.0)),
            Sport.BADMINTON: ((-9.0, -9.0, 6.5), (0.0, 0.0, 0.5)),
            Sport.BASKETBALL: ((-19.0, -18.0, 12.0), (0.0, 0.0, 0.5)),
        }[sport]
        camera_eye, camera_target = camera

    primitives: list[IsaacPrimitive] = [
        _box(
            "campus_ground",
            (0.0, 10.0, -0.10),
            (150.0, 150.0, 0.10),
            (0.09, 0.11, 0.12),
            friction=(0.9, 0.8),
        )
    ]
    balls = []
    for sport in sports:
        sport_primitives, ball = BUILDERS[sport](*offsets[sport])
        primitives.extend(
            IsaacPrimitive(
                name=f"{sport.value}/{primitive.name}",
                kind=primitive.kind,
                position=primitive.position,
                size=primitive.size,
                color=primitive.color,
                orientation=primitive.orientation,
                collision=primitive.collision,
                opacity=primitive.opacity,
                static_friction=primitive.static_friction,
                dynamic_friction=primitive.dynamic_friction,
                restitution=primitive.restitution,
            )
            for primitive in sport_primitives
        )
        balls.append(ball)
    return IsaacSceneSpec(scene, tuple(primitives), tuple(balls), camera_eye, camera_target)
