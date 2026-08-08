"""Programmatic regulation-size MuJoCo scene construction."""

from __future__ import annotations

from html import escape
from math import cos, pi, sin

import mujoco

from .specs import BALLS, COURTS, Sport

SCENES = ("campus", *(sport.value for sport in Sport))
WHITE = "0.96 0.96 0.94 1"


def _rgba(values: tuple[float, float, float, float]) -> str:
    return " ".join(f"{value:.4g}" for value in values)


def _geom(name: str, geom_type: str = "box", **attributes: object) -> str:
    attrs = {"name": name, "type": geom_type, **attributes}
    encoded = " ".join(
        f'{key.rstrip("_")}="{escape(str(value))}"' for key, value in attrs.items()
    )
    return f"<geom {encoded}/>"


def _line_box(
    name: str, center: tuple[float, float], half_size: tuple[float, float], z: float = 0.006
) -> str:
    return _geom(
        name,
        pos=f"{center[0]} {center[1]} {z}",
        size=f"{half_size[0]} {half_size[1]} 0.006",
        rgba=WHITE,
        contype="0",
        conaffinity="0",
        mass="0",
        group="2",
    )


def _rectangle_lines(
    prefix: str, x: float, y: float, length: float, width: float, line_width: float = 0.05
) -> list[str]:
    return [
        _line_box(f"{prefix}_line_n", (x, y + width / 2), (length / 2, line_width / 2)),
        _line_box(f"{prefix}_line_s", (x, y - width / 2), (length / 2, line_width / 2)),
        _line_box(f"{prefix}_line_e", (x + length / 2, y), (line_width / 2, width / 2)),
        _line_box(f"{prefix}_line_w", (x - length / 2, y), (line_width / 2, width / 2)),
    ]


def _circle_lines(
    prefix: str,
    x: float,
    y: float,
    radius: float,
    segments: int = 40,
    width: float = 0.035,
    z: float = 0.013,
) -> list[str]:
    lines: list[str] = []
    for index in range(segments):
        a0 = 2 * pi * index / segments
        a1 = 2 * pi * (index + 1) / segments
        lines.append(
            _geom(
                f"{prefix}_{index}",
                "capsule",
                fromto=(
                    f"{x + radius * cos(a0)} {y + radius * sin(a0)} {z} "
                    f"{x + radius * cos(a1)} {y + radius * sin(a1)} {z}"
                ),
                size=width,
                rgba=WHITE,
                contype="0",
                conaffinity="0",
                mass="0",
                group="2",
            )
        )
    return lines


def _arc_lines(
    prefix: str,
    x: float,
    y: float,
    radius: float,
    start: float,
    stop: float,
    segments: int = 24,
    width: float = 0.035,
) -> list[str]:
    lines: list[str] = []
    for index in range(segments):
        a0 = start + (stop - start) * index / segments
        a1 = start + (stop - start) * (index + 1) / segments
        lines.append(
            _geom(
                f"{prefix}_{index}",
                "capsule",
                fromto=(
                    f"{x + radius*cos(a0)} {y + radius*sin(a0)} 0.013 "
                    f"{x + radius*cos(a1)} {y + radius*sin(a1)} 0.013"
                ),
                size=width,
                rgba=WHITE,
                contype="0",
                conaffinity="0",
                mass="0",
                group="2",
            )
        )
    return lines


def _court_surface(sport: Sport, x: float, y: float) -> str:
    spec = COURTS[sport]
    return _geom(
        f"{sport.value}_surface",
        pos=f"{x} {y} -0.04",
        size=f"{spec.length / 2} {spec.width / 2} 0.04",
        rgba=_rgba(spec.color),
        friction="0.8 0.01 0.001",
        condim="6",
    )


def _net(
    prefix: str,
    x: float,
    y: float,
    span: float,
    height: float,
    along_y: bool = True,
    bottom: float = 0.0,
) -> list[str]:
    post_a = (x, y - span / 2) if along_y else (x - span / 2, y)
    post_b = (x, y + span / 2) if along_y else (x + span / 2, y)
    net_size = f"0.012 {span / 2} {(height - bottom) / 2}" if along_y else f"{span / 2} 0.012 {(height - bottom) / 2}"
    return [
        _geom(
            f"{prefix}_net",
            pos=f"{x} {y} {(height + bottom) / 2}",
            size=net_size,
            rgba="0.12 0.15 0.18 0.38",
            friction="0.3 0.005 0.0001",
        ),
        _geom(
            f"{prefix}_post_a",
            "cylinder",
            pos=f"{post_a[0]} {post_a[1]} {height / 2}",
            size=f"0.035 {height / 2}",
            rgba="0.16 0.16 0.17 1",
        ),
        _geom(
            f"{prefix}_post_b",
            "cylinder",
            pos=f"{post_b[0]} {post_b[1]} {height / 2}",
            size=f"0.035 {height / 2}",
            rgba="0.16 0.16 0.17 1",
        ),
    ]


def _racket_body(prefix: str, pos: tuple[float, float, float], scale: float, color: str) -> str:
    """Create a fixed racket with a collision hoop, handle and visible strings."""
    parts = [
        f'<body name="{prefix}_racket" pos="{pos[0]} {pos[1]} {pos[2]}">',
        _geom(
            f"{prefix}_racket_handle",
            "capsule",
            fromto=f"0 0 {-0.34 * scale} 0 0 {-0.04 * scale}",
            size=0.025 * scale,
            rgba="0.18 0.12 0.07 1",
        ),
    ]
    rx, rz, center_z = 0.16 * scale, 0.21 * scale, 0.17 * scale
    for index in range(24):
        a0, a1 = 2 * pi * index / 24, 2 * pi * (index + 1) / 24
        parts.append(
            _geom(
                f"{prefix}_racket_hoop_{index}",
                "capsule",
                fromto=(
                    f"{rx * cos(a0)} 0 {center_z + rz * sin(a0)} "
                    f"{rx * cos(a1)} 0 {center_z + rz * sin(a1)}"
                ),
                size=0.011 * scale,
                rgba=color,
            )
        )
    for index, value in enumerate((-0.10, -0.05, 0.0, 0.05, 0.10)):
        chord = rz * (1 - (value / rx) ** 2) ** 0.5
        parts.append(
            _geom(
                f"{prefix}_racket_vstring_{index}",
                "capsule",
                fromto=f"{value * scale} 0 {center_z - chord} {value * scale} 0 {center_z + chord}",
                size=0.0015 * scale,
                rgba="0.9 0.9 0.85 0.8",
                contype="0",
                conaffinity="0",
                mass="0",
            )
        )
    for index, value in enumerate((-0.12, -0.06, 0.0, 0.06, 0.12)):
        half_chord = rx * (1 - (value / rz) ** 2) ** 0.5
        parts.append(
            _geom(
                f"{prefix}_racket_hstring_{index}",
                "capsule",
                fromto=(
                    f"{-half_chord} 0 {center_z + value * scale} "
                    f"{half_chord} 0 {center_z + value * scale}"
                ),
                size=0.0015 * scale,
                rgba="0.9 0.9 0.85 0.8",
                contype="0",
                conaffinity="0",
                mass="0",
            )
        )
    parts.append("</body>")
    return "".join(parts)


def _paddle_body(prefix: str, pos: tuple[float, float, float], red: bool) -> str:
    color = "0.75 0.04 0.03 1" if red else "0.03 0.04 0.05 1"
    return "".join(
        [
            f'<body name="{prefix}_paddle" pos="{pos[0]} {pos[1]} {pos[2]}">',
            _geom(
                f"{prefix}_paddle_blade",
                "ellipsoid",
                pos="0 0 0.08",
                size="0.075 0.008 0.09",
                rgba=color,
                friction="0.85 0.01 0.001",
            ),
            _geom(
                f"{prefix}_paddle_handle",
                "capsule",
                fromto="0 0 -0.08 0 0 0.015",
                size="0.017",
                rgba="0.48 0.27 0.10 1",
            ),
            "</body>",
        ]
    )


def _ball(sport: Sport, x: float, y: float, z: float) -> str:
    spec = BALLS[sport]
    if sport is Sport.BADMINTON:
        return _shuttlecock(x, y, z)
    decorations: list[str] = []
    if sport is Sport.FOOTBALL:
        patch_distance = spec.radius * 0.87
        for index, position in enumerate(
            (
                (patch_distance, 0.0, 0.0),
                (-patch_distance, 0.0, 0.0),
                (0.0, patch_distance, 0.0),
                (0.0, -patch_distance, 0.0),
                (0.0, 0.0, patch_distance),
                (0.0, 0.0, -patch_distance),
            )
        ):
            decorations.append(
                _geom(
                    f"football_patch_{index}",
                    "sphere",
                    pos=" ".join(str(value) for value in position),
                    size=spec.radius * 0.29,
                    rgba="0.035 0.04 0.045 1",
                    contype="0",
                    conaffinity="0",
                    mass="0",
                )
            )
    elif sport is Sport.BASKETBALL:
        seam_radius = spec.radius * 1.006
        for plane in range(3):
            for index in range(24):
                a0, a1 = 2 * pi * index / 24, 2 * pi * (index + 1) / 24
                p0 = (seam_radius * cos(a0), seam_radius * sin(a0))
                p1 = (seam_radius * cos(a1), seam_radius * sin(a1))
                if plane == 0:
                    start, end = (*p0, 0.0), (*p1, 0.0)
                elif plane == 1:
                    start, end = (p0[0], 0.0, p0[1]), (p1[0], 0.0, p1[1])
                else:
                    start, end = (0.0, *p0), (0.0, *p1)
                decorations.append(
                    _geom(
                        f"basketball_seam_{plane}_{index}",
                        "capsule",
                        fromto=" ".join(str(value) for value in (*start, *end)),
                        size="0.0018",
                        rgba="0.08 0.035 0.015 1",
                        contype="0",
                        conaffinity="0",
                        mass="0",
                    )
                )
    return "".join(
        [
            f'<body name="{sport.value}_ball" pos="{x} {y} {z}">',
            f'<freejoint name="{sport.value}_ball_free"/>',
            _geom(
                f"{sport.value}_ball_geom",
                "sphere",
                size=spec.radius,
                mass=spec.mass,
                rgba=_rgba(spec.color),
                friction=f"0.55 0.006 {spec.rolling_friction}",
                condim="6",
                solref="0.003 0.35",
            ),
            "".join(decorations),
            "</body>",
        ]
    )


def _shuttlecock(x: float, y: float, z: float) -> str:
    spec = BALLS[Sport.BADMINTON]
    parts = [
        f'<body name="badminton_ball" pos="{x} {y} {z}">',
        '<freejoint name="badminton_ball_free"/>',
        _geom(
            "badminton_ball_geom",
            "sphere",
            pos="0 0 -0.012",
            size=spec.radius,
            mass=spec.mass,
            rgba="0.93 0.88 0.72 1",
            friction="0.45 0.004 0.004",
            condim="6",
        ),
    ]
    for index in range(16):
        angle = 2 * pi * index / 16
        bx, by = 0.010 * cos(angle), 0.010 * sin(angle)
        tx, ty = 0.031 * cos(angle), 0.031 * sin(angle)
        parts.append(
            _geom(
                f"shuttle_feather_{index}",
                "capsule",
                fromto=f"{bx} {by} 0 {tx} {ty} 0.066",
                size="0.0013",
                rgba="0.97 0.97 0.93 0.92",
                contype="0",
                conaffinity="0",
                mass="0",
            )
        )
    # Two rings visually bind the feather skirt.
    for ring, z_ring, radius in ((0, 0.025, 0.018), (1, 0.046, 0.025)):
        for index in range(16):
            a0, a1 = 2 * pi * index / 16, 2 * pi * (index + 1) / 16
            parts.append(
                _geom(
                    f"shuttle_ring_{ring}_{index}",
                    "capsule",
                    fromto=(
                        f"{radius*cos(a0)} {radius*sin(a0)} {z_ring} "
                        f"{radius*cos(a1)} {radius*sin(a1)} {z_ring}"
                    ),
                    size="0.0008",
                    rgba="0.85 0.85 0.80 1",
                    contype="0",
                    conaffinity="0",
                    mass="0",
                )
            )
    parts.append("</body>")
    return "".join(parts)


def _tennis(x: float, y: float) -> list[str]:
    items = [_court_surface(Sport.TENNIS, x, y)]
    items += _rectangle_lines("tennis_doubles", x, y, 23.77, 10.97)
    items += _rectangle_lines("tennis_singles", x, y, 23.77, 8.23)
    items += [
        _line_box("tennis_service_e", (x + 6.40, y), (0.025, 4.115)),
        _line_box("tennis_service_w", (x - 6.40, y), (0.025, 4.115)),
        _line_box("tennis_service_center", (x, y), (6.40, 0.025)),
    ]
    items += _net("tennis", x, y, 12.8, 0.914)
    items += [
        _racket_body("tennis_a", (x - 2.0, y - 4.6, 0.38), 1.0, "0.06 0.12 0.19 1"),
        _racket_body("tennis_b", (x + 2.0, y + 4.6, 0.38), 1.0, "0.7 0.12 0.04 1"),
        _ball(Sport.TENNIS, x - 5.0, y, 1.4),
        f'<camera name="tennis_camera" pos="{x - 15} {y - 15} 10" xyaxes="0.70 -0.70 0 0.35 0.35 0.87"/>',
    ]
    return items


def _table_tennis(x: float, y: float) -> list[str]:
    items = [_court_surface(Sport.TABLE_TENNIS, x, y)]
    items += _rectangle_lines("table_tennis_zone", x, y, 8.0, 5.0)
    tabletop_z, tabletop_t = 0.76, 0.04
    items += [
        _geom(
            "table_tennis_table",
            pos=f"{x} {y} {tabletop_z - tabletop_t / 2}",
            size=f"{2.74/2} {1.525/2} {tabletop_t/2}",
            rgba="0.04 0.24 0.43 1",
            friction="0.35 0.003 0.0005",
            condim="6",
        ),
        _line_box("table_tennis_center_line", (x, y), (1.37, 0.0015), tabletop_z + 0.008),
    ]
    for ix in (-1.15, 1.15):
        for iy in (-0.62, 0.62):
            items.append(
                _geom(
                    f"table_leg_{ix}_{iy}",
                    pos=f"{x+ix} {y+iy} 0.36",
                    size="0.025 0.025 0.36",
                    rgba="0.15 0.16 0.17 1",
                )
            )
    items += _net("table_tennis", x, y, 1.83, tabletop_z + 0.1525, bottom=tabletop_z)
    items += [
        _paddle_body("table_tennis_a", (x - 1.7, y - 0.95, 0.23), True),
        _paddle_body("table_tennis_b", (x + 1.7, y + 0.95, 0.23), False),
        _ball(Sport.TABLE_TENNIS, x - 0.7, y, 1.25),
        f'<camera name="table_tennis_camera" pos="{x-4.5} {y-4.5} 3.3" xyaxes="0.707 -0.707 0 0.35 0.35 0.87"/>',
    ]
    return items


def _goal(prefix: str, x: float, y: float, facing: float) -> list[str]:
    depth = 2.0 * facing
    return [
        _geom(f"{prefix}_left_post", "cylinder", pos=f"{x} {y-3.66} 1.22", size="0.06 1.22", rgba=WHITE),
        _geom(f"{prefix}_right_post", "cylinder", pos=f"{x} {y+3.66} 1.22", size="0.06 1.22", rgba=WHITE),
        _geom(f"{prefix}_crossbar", "capsule", fromto=f"{x} {y-3.66} 2.44 {x} {y+3.66} 2.44", size="0.06", rgba=WHITE),
        _geom(
            f"{prefix}_goal_net",
            pos=f"{x + depth/2} {y} 1.22",
            size=f"{abs(depth)/2} 3.66 1.22",
            rgba="0.9 0.9 0.9 0.12",
            friction="0.2 0.001 0.0001",
        ),
    ]


def _football(x: float, y: float) -> list[str]:
    items = [_court_surface(Sport.FOOTBALL, x, y)]
    items += _rectangle_lines("football_boundary", x, y, 105.0, 68.0, 0.10)
    items += [_line_box("football_halfway", (x, y), (0.05, 34.0))]
    items += _circle_lines("football_center_circle", x, y, 9.15, width=0.05)
    items += _rectangle_lines("football_penalty_w", x - 52.5 + 8.25, y, 16.5, 40.32, 0.08)
    items += _rectangle_lines("football_penalty_e", x + 52.5 - 8.25, y, 16.5, 40.32, 0.08)
    items += _goal("football_goal_w", x - 52.5, y, -1.0)
    items += _goal("football_goal_e", x + 52.5, y, 1.0)
    items += [
        _ball(Sport.FOOTBALL, x - 18.0, y, 1.0),
        f'<camera name="football_camera" pos="{x-60} {y-50} 55" xyaxes="0.64 -0.77 0 0.44 0.37 0.82"/>',
    ]
    return items


def _badminton(x: float, y: float) -> list[str]:
    items = [_court_surface(Sport.BADMINTON, x, y)]
    items += _rectangle_lines("badminton_doubles", x, y, 13.40, 6.10, 0.04)
    items += _rectangle_lines("badminton_singles", x, y, 13.40, 5.18, 0.04)
    for dx, name in ((1.98, "short_e"), (-1.98, "short_w"), (5.94, "long_e"), (-5.94, "long_w")):
        items.append(_line_box(f"badminton_{name}", (x + dx, y), (0.02, 3.05)))
    items += [
        _line_box("badminton_center_e", (x + 4.34, y), (2.36, 0.02)),
        _line_box("badminton_center_w", (x - 4.34, y), (2.36, 0.02)),
    ]
    items += _net("badminton", x, y, 6.10, 1.55)
    items += [
        _racket_body("badminton_a", (x - 2.4, y - 2.7, 0.28), 0.68, "0.08 0.08 0.09 1"),
        _racket_body("badminton_b", (x + 2.4, y + 2.7, 0.28), 0.68, "0.18 0.36 0.64 1"),
        _ball(Sport.BADMINTON, x - 3.0, y, 2.2),
        f'<camera name="badminton_camera" pos="{x-9} {y-9} 6.5" xyaxes="0.707 -0.707 0 0.36 0.36 0.86"/>',
    ]
    return items


def _basket_structure(prefix: str, x: float, y: float, facing: float) -> list[str]:
    rim_x = x + facing * 1.2
    support_x = x - facing * 0.35
    items = [
        _geom(
            f"{prefix}_support",
            "cylinder",
            pos=f"{support_x} {y} 1.6",
            size="0.10 1.6",
            rgba="0.12 0.13 0.15 1",
        ),
        _geom(
            f"{prefix}_arm",
            "capsule",
            fromto=f"{support_x} {y} 3.0 {x} {y} 3.0",
            size="0.07",
            rgba="0.12 0.13 0.15 1",
        ),
        _geom(
            f"{prefix}_backboard",
            pos=f"{x} {y} 3.40",
            size="0.035 0.90 0.525",
            rgba="0.88 0.91 0.92 0.55",
            friction="0.55 0.005 0.0005",
        ),
    ]
    for index in range(32):
        a0, a1 = 2 * pi * index / 32, 2 * pi * (index + 1) / 32
        items.append(
            _geom(
                f"{prefix}_rim_{index}",
                "capsule",
                fromto=(
                    f"{rim_x + 0.225*cos(a0)} {y + 0.225*sin(a0)} 3.05 "
                    f"{rim_x + 0.225*cos(a1)} {y + 0.225*sin(a1)} 3.05"
                ),
                size="0.01",
                rgba="0.95 0.23 0.03 1",
                friction="0.5 0.003 0.0003",
            )
        )
    return items


def _basketball(x: float, y: float) -> list[str]:
    items = [_court_surface(Sport.BASKETBALL, x, y)]
    items += _rectangle_lines("basketball_boundary", x, y, 28.0, 15.0, 0.05)
    items += [_line_box("basketball_halfway", (x, y), (0.025, 7.5))]
    items += _circle_lines("basketball_center_circle", x, y, 1.80, width=0.025)
    for side, dx in (("w", -11.1), ("e", 11.1)):
        items += _rectangle_lines(f"basketball_key_{side}", x + dx, y, 5.8, 4.9, 0.04)
        items += _circle_lines(f"basketball_free_throw_{side}", x + dx, y, 1.80, width=0.025)
    # FIBA three-point radius is 6.75 m; baseline straight segments complete each arc.
    items += _arc_lines("basketball_three_w", x - 12.425, y, 6.75, -1.20, 1.20, width=0.025)
    items += _arc_lines("basketball_three_e", x + 12.425, y, 6.75, pi - 1.20, pi + 1.20, width=0.025)
    items += [
        _line_box("basketball_three_w_n", (x - 11.0, y + 6.60), (3.0, 0.025)),
        _line_box("basketball_three_w_s", (x - 11.0, y - 6.60), (3.0, 0.025)),
        _line_box("basketball_three_e_n", (x + 11.0, y + 6.60), (3.0, 0.025)),
        _line_box("basketball_three_e_s", (x + 11.0, y - 6.60), (3.0, 0.025)),
    ]
    items += _basket_structure("basketball_w", x - 14.0, y, 1.0)
    items += _basket_structure("basketball_e", x + 14.0, y, -1.0)
    items += [
        _ball(Sport.BASKETBALL, x - 4.0, y, 1.5),
        f'<camera name="basketball_camera" pos="{x-19} {y-18} 12" xyaxes="0.69 -0.72 0 0.36 0.35 0.86"/>',
    ]
    return items


BUILDERS = {
    Sport.TENNIS: _tennis,
    Sport.TABLE_TENNIS: _table_tennis,
    Sport.FOOTBALL: _football,
    Sport.BADMINTON: _badminton,
    Sport.BASKETBALL: _basketball,
}


CONTACTS = {
    # Values are (time constant, damping ratio). They were calibrated with the
    # regulation drop tests in tests/test_bounce.py at the 1 ms model timestep.
    Sport.TENNIS: ("0.012 0.10", "0.75 0.015 0.003"),
    Sport.TABLE_TENNIS: ("0.025 0.03", "0.35 0.004 0.001"),
    Sport.FOOTBALL: ("0.020 0.10", "0.80 0.018 0.015"),
    Sport.BADMINTON: ("0.008 1.00", "0.45 0.006 0.004"),
    Sport.BASKETBALL: ("0.020 0.075", "0.80 0.018 0.012"),
}


def _contact_pairs(sports: list[Sport]) -> str:
    pairs: list[str] = []
    for sport in sports:
        solref, friction = CONTACTS[sport]
        pairs.append(
            f'<pair name="{sport.value}_surface_contact" '
            f'geom1="{sport.value}_surface" geom2="{sport.value}_ball_geom" '
            f'condim="6" friction="{friction}" solref="{solref}" solimp="0.96 0.99 0.001"/>'
        )
        if sport is Sport.TABLE_TENNIS:
            pairs.append(
                '<pair name="table_tennis_table_contact" geom1="table_tennis_table" '
                'geom2="table_tennis_ball_geom" condim="6" friction="0.25 0.003 0.0005" '
                'solref="0.025 0.03" solimp="0.97 0.995 0.0005"/>'
            )
    return "".join(pairs)


def build_xml(scene: str = "campus") -> str:
    """Build a complete MJCF string for one sport or the complete campus."""
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
    else:
        sports = [Sport(scene)]
        offsets = {sports[0]: (0.0, 0.0)}

    world: list[str] = [
        '<light name="sun" pos="0 -20 80" dir="0 0 -1" directional="true" diffuse="0.9 0.9 0.86"/>',
        '<light name="fill" pos="20 30 30" diffuse="0.35 0.40 0.48"/>',
        _geom(
            "campus_ground",
            pos="0 10 -0.09",
            size="75 75 0.05",
            rgba="0.09 0.11 0.12 1",
            friction="0.9 0.015 0.002",
        ),
    ]
    for sport in sports:
        world.extend(BUILDERS[sport](*offsets[sport]))
    if scene == "campus":
        world.append(
            '<camera name="campus_camera" pos="-92 -100 112" xyaxes="0.735 -0.678 0 0.43 0.466 0.773"/>'
        )

    return f"""<mujoco model="multisport_{scene}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" iterations="80"/>
  <visual>
    <headlight ambient="0.22 0.22 0.25" diffuse="0.70 0.70 0.68" specular="0.25 0.25 0.25"/>
    <rgba haze="0.12 0.15 0.19 1"/>
    <global offwidth="1920" offheight="1080" azimuth="135" elevation="-32"/>
  </visual>
  <statistic meansize="2" extent="70" center="0 15 0"/>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.10 0.16 0.26" rgb2="0.55 0.69 0.82" width="512" height="3072"/>
  </asset>
  <default>
    <geom solref="0.006 0.7" solimp="0.94 0.99 0.001" friction="0.7 0.01 0.001" condim="4"/>
  </default>
  <worldbody>{''.join(world)}</worldbody>
  <contact>{_contact_pairs(sports)}</contact>
</mujoco>"""


def build_model(scene: str = "campus") -> mujoco.MjModel:
    """Compile and return a MuJoCo model, raising with MJCF diagnostics on errors."""
    return mujoco.MjModel.from_xml_string(build_xml(scene))
