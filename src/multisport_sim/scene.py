"""Programmatic regulation-size MuJoCo scene construction."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import cos, pi, sin

import mujoco

from .specs import BALLS, CAMPUS_OFFSETS, COURTS, SCENES, TABLE_TENNIS, Sport

WHITE = "0.96 0.96 0.94 1"


def _rgba(values: tuple[float, float, float, float]) -> str:
    return " ".join(f"{value:.4g}" for value in values)


def _geom(name: str, geom_type: str = "box", **attributes: object) -> str:
    attrs = {"name": name, "type": geom_type, **attributes}
    encoded = " ".join(f'{key.rstrip("_")}="{escape(str(value))}"' for key, value in attrs.items())
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
                    f"{x + radius * cos(a0)} {y + radius * sin(a0)} 0.013 "
                    f"{x + radius * cos(a1)} {y + radius * sin(a1)} 0.013"
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
    net_size = (
        f"0.012 {span / 2} {(height - bottom) / 2}"
        if along_y
        else f"{span / 2} 0.012 {(height - bottom) / 2}"
    )
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


@dataclass(frozen=True)
class BenchmarkEffector:
    """The optional kinematic striking surface one sport's harness uses.

    Every sport needs the same three things -- a mocap body parked out of play,
    one geom that is the only legal striking surface, and a size matching the
    real implement -- so they are declared here rather than written out per
    sport.  Keeping it separate from the decorative rackets leaves every default
    scene unchanged.
    """

    sport: Sport
    parked_at: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    rgba: str = "0.82 0.05 0.035 1"
    friction: str = "0.85 0.01 0.001"
    # Ball-implement contact, given explicitly rather than inherited.  The
    # scene default is a damping ratio of 0.7, which returns a ball at a
    # coefficient of restitution near 0.2: the fixture would be swinging a
    # sponge, and no reachable speed would produce a legal return.  ``None``
    # keeps the default, which is what the frozen table-tennis v0 scores were
    # measured with and must keep being measured with.
    contact_solref: tuple[float, float] | None = None
    contact_solimp: tuple[float, float, float] | None = None
    contact_friction: str = "0.72 0.01 0.001"

    @property
    def body(self) -> str:
        return f"{self.sport.value}_benchmark_paddle"

    @property
    def geom(self) -> str:
        return f"{self.sport.value}_benchmark_paddle_blade"


BENCHMARK_EFFECTORS: dict[Sport, BenchmarkEffector] = {
    # Blade thin along its own y axis, which is therefore the strike normal.
    Sport.TABLE_TENNIS: BenchmarkEffector(
        sport=Sport.TABLE_TENNIS,
        parked_at=(-1.58, 1.25, 1.0),
        half_extents=(0.085, 0.008, 0.10),
    ),
    # An ITF-legal frame is at most 0.737 x 0.318 m overall; the strung face is
    # modelled at roughly 0.26 x 0.32 m, the part that can legally return a ball.
    Sport.TENNIS: BenchmarkEffector(
        sport=Sport.TENNIS,
        # Calibrated by the drop test in tests/test_scenes.py: a ball dropped on
        # a fixed string bed rebounds at a coefficient of restitution near 0.8,
        # which is where measured ball-on-strings restitution sits.  Inheriting
        # the scene default instead gives about 0.2.
        contact_solref=(0.009, 0.115),
        contact_solimp=(0.97, 0.995, 0.0005),
        # Parked behind the baseline and outside the singles width, but inside
        # the task's declared workspace: an effector that starts outside its own
        # action space makes the environment's first observation invalid.
        parked_at=(-9.5, 4.9, 1.0),
        half_extents=(0.130, 0.010, 0.160),
        rgba="0.06 0.12 0.19 1",
        friction="0.72 0.01 0.001",
    ),
}


def _benchmark_paddle(sport: Sport) -> str:
    """Create one sport's kinematic benchmark effector.

    The body starts parked outside the playing area.  Benchmark controllers move
    it through ``mjData.mocap_pos`` and ``mjData.mocap_quat``.
    """
    effector = BENCHMARK_EFFECTORS[sport]
    position = " ".join(f"{value}" for value in effector.parked_at)
    size = " ".join(f"{value}" for value in effector.half_extents)
    return "".join(
        [
            f'<body name="{effector.body}" mocap="true" pos="{position}">',
            _geom(
                effector.geom,
                "ellipsoid",
                size=size,
                rgba=effector.rgba,
                friction=effector.friction,
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
                        f"{radius * cos(a0)} {radius * sin(a0)} {z_ring} "
                        f"{radius * cos(a1)} {radius * sin(a1)} {z_ring}"
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


def _table_tennis_net(x: float, y: float) -> list[str]:
    """Create a regulation net with a collision panel and visible cord mesh."""
    bottom = TABLE_TENNIS.table_height
    top = TABLE_TENNIS.net_top_height
    span = TABLE_TENNIS.net_span
    items = [
        _geom(
            "table_tennis_net",
            pos=f"{x} {y} {(bottom + top) / 2}",
            size=f"0.006 {span / 2} {(top - bottom) / 2}",
            rgba="0.04 0.07 0.11 0.20",
            friction="0.3 0.005 0.0001",
        ),
        _geom(
            "table_tennis_post_a",
            "cylinder",
            pos=f"{x} {y - span / 2} 0.82",
            size="0.018 0.155",
            rgba="0.06 0.07 0.08 1",
        ),
        _geom(
            "table_tennis_post_b",
            "cylinder",
            pos=f"{x} {y + span / 2} 0.82",
            size="0.018 0.155",
            rgba="0.06 0.07 0.08 1",
        ),
        _geom(
            "table_tennis_net_top_tape",
            "capsule",
            fromto=f"{x} {y - span / 2} {top} {x} {y + span / 2} {top}",
            size="0.010",
            rgba=WHITE,
            contype="0",
            conaffinity="0",
            mass="0",
        ),
    ]
    for index in range(1, 12):
        cord_y = y - span / 2 + span * index / 12
        items.append(
            _geom(
                f"table_tennis_net_vertical_{index}",
                "capsule",
                fromto=f"{x} {cord_y} {bottom} {x} {cord_y} {top}",
                size="0.0022",
                rgba="0.84 0.87 0.88 1",
                contype="0",
                conaffinity="0",
                mass="0",
            )
        )
    for index in range(1, 4):
        cord_z = bottom + (top - bottom) * index / 4
        items.append(
            _geom(
                f"table_tennis_net_horizontal_{index}",
                "capsule",
                fromto=f"{x} {y - span / 2} {cord_z} {x} {y + span / 2} {cord_z}",
                size="0.0022",
                rgba="0.84 0.87 0.88 1",
                contype="0",
                conaffinity="0",
                mass="0",
            )
        )
    return items


def _table_tennis(x: float, y: float) -> list[str]:
    items = [_court_surface(Sport.TABLE_TENNIS, x, y)]
    items += _rectangle_lines("table_tennis_zone", x, y, 8.0, 5.0)
    tabletop_z, tabletop_t = TABLE_TENNIS.table_height, 0.04
    half_length, half_width = TABLE_TENNIS.half_length, TABLE_TENNIS.half_width
    items += [
        _geom(
            "table_tennis_table",
            pos=f"{x} {y} {tabletop_z - tabletop_t / 2}",
            size=f"{half_length} {half_width} {tabletop_t / 2}",
            rgba="0.025 0.27 0.52 1",
            friction="0.35 0.003 0.0005",
            condim="6",
        ),
        _line_box(
            "table_tennis_center_line", (x, y), (half_length, 0.0015), tabletop_z + 0.006
        ),
        _line_box(
            "table_tennis_edge_north",
            (x, y + half_width - 0.010),
            (half_length, 0.010),
            tabletop_z + 0.006,
        ),
        _line_box(
            "table_tennis_edge_south",
            (x, y - half_width + 0.010),
            (half_length, 0.010),
            tabletop_z + 0.006,
        ),
        _line_box(
            "table_tennis_edge_east",
            (x + half_length - 0.010, y),
            (0.010, half_width),
            tabletop_z + 0.006,
        ),
        _line_box(
            "table_tennis_edge_west",
            (x - half_length + 0.010, y),
            (0.010, half_width),
            tabletop_z + 0.006,
        ),
        _geom(
            "table_tennis_apron_north",
            pos=f"{x} {y + 0.745} 0.69",
            size="1.33 0.0175 0.05",
            rgba="0.06 0.07 0.08 1",
        ),
        _geom(
            "table_tennis_apron_south",
            pos=f"{x} {y - 0.745} 0.69",
            size="1.33 0.0175 0.05",
            rgba="0.06 0.07 0.08 1",
        ),
    ]
    for ix in (-1.05, 1.05):
        for iy in (-0.55, 0.55):
            items.append(
                _geom(
                    f"table_leg_{ix}_{iy}",
                    pos=f"{x + ix} {y + iy} 0.35",
                    size="0.03 0.03 0.35",
                    rgba="0.15 0.16 0.17 1",
                )
            )
    items += [
        _geom(
            "table_tennis_brace_w",
            "capsule",
            fromto=f"{x - 1.05} {y - 0.55} 0.48 {x - 1.05} {y + 0.55} 0.48",
            size="0.025",
            rgba="0.06 0.07 0.08 1",
        ),
        _geom(
            "table_tennis_brace_e",
            "capsule",
            fromto=f"{x + 1.05} {y - 0.55} 0.48 {x + 1.05} {y + 0.55} 0.48",
            size="0.025",
            rgba="0.06 0.07 0.08 1",
        ),
    ]
    items += _table_tennis_net(x, y)
    items += [
        _paddle_body("table_tennis_a", (x - 1.7, y - 0.95, 0.23), True),
        _paddle_body("table_tennis_b", (x + 1.7, y + 0.95, 0.23), False),
        _ball(Sport.TABLE_TENNIS, x - 0.7, y, 1.25),
        f'<camera name="table_tennis_camera" pos="{x - 4.5} {y - 4.5} 3.3" xyaxes="0.707 -0.707 0 0.35 0.35 0.87"/>',
    ]
    return items


def _goal(prefix: str, x: float, y: float, facing: float) -> list[str]:
    depth = 2.0 * facing
    return [
        _geom(
            f"{prefix}_left_post",
            "cylinder",
            pos=f"{x} {y - 3.66} 1.22",
            size="0.06 1.22",
            rgba=WHITE,
        ),
        _geom(
            f"{prefix}_right_post",
            "cylinder",
            pos=f"{x} {y + 3.66} 1.22",
            size="0.06 1.22",
            rgba=WHITE,
        ),
        _geom(
            f"{prefix}_crossbar",
            "capsule",
            fromto=f"{x} {y - 3.66} 2.44 {x} {y + 3.66} 2.44",
            size="0.06",
            rgba=WHITE,
        ),
        _geom(
            f"{prefix}_goal_net",
            pos=f"{x + depth / 2} {y} 1.22",
            size=f"{abs(depth) / 2} 3.66 1.22",
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
        f'<camera name="football_camera" pos="{x - 60} {y - 50} 55" xyaxes="0.64 -0.77 0 0.44 0.37 0.82"/>',
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
        f'<camera name="badminton_camera" pos="{x - 9} {y - 9} 6.5" xyaxes="0.707 -0.707 0 0.36 0.36 0.86"/>',
    ]
    return items


def _basket_net(prefix: str, rim_x: float, y: float) -> list[str]:
    """Create a visible tapered basketball net from non-colliding cords."""
    items: list[str] = []
    strands, rim_z, bottom_z = 12, 3.045, 2.62
    for index in range(strands):
        angle = 2 * pi * index / strands
        next_angle = 2 * pi * (index + 1) / strands
        for suffix, top_angle, lower_angle in (
            ("a", angle, next_angle),
            ("b", next_angle, angle),
        ):
            items.append(
                _geom(
                    f"{prefix}_net_strand_{index}_{suffix}",
                    "capsule",
                    fromto=(
                        f"{rim_x + 0.225 * cos(top_angle)} {y + 0.225 * sin(top_angle)} {rim_z} "
                        f"{rim_x + 0.13 * cos(lower_angle)} {y + 0.13 * sin(lower_angle)} {bottom_z}"
                    ),
                    size="0.004",
                    rgba=WHITE,
                    contype="0",
                    conaffinity="0",
                    mass="0",
                )
            )
    for tier, (radius, z) in enumerate(((0.19, 2.90), (0.16, 2.76), (0.13, bottom_z))):
        for index in range(strands):
            a0, a1 = 2 * pi * index / strands, 2 * pi * (index + 1) / strands
            items.append(
                _geom(
                    f"{prefix}_net_ring_{tier}_{index}",
                    "capsule",
                    fromto=(
                        f"{rim_x + radius * cos(a0)} {y + radius * sin(a0)} {z} "
                        f"{rim_x + radius * cos(a1)} {y + radius * sin(a1)} {z}"
                    ),
                    size="0.0035",
                    rgba=WHITE,
                    contype="0",
                    conaffinity="0",
                    mass="0",
                )
            )
    return items


def _basket_structure(prefix: str, x: float, y: float, facing: float) -> list[str]:
    board_x = x + facing * 1.20
    rim_x = x + facing * 1.575
    support_x = x - facing * 0.45
    front_x = board_x + facing * 0.041
    items = [
        _geom(
            f"{prefix}_support",
            "cylinder",
            pos=f"{support_x} {y} 1.6",
            size="0.10 1.6",
            rgba="0.12 0.13 0.15 1",
        ),
        _geom(
            f"{prefix}_support_base",
            pos=f"{support_x} {y} 0.12",
            size="0.375 0.525 0.12",
            rgba="0.12 0.13 0.15 1",
        ),
        _geom(
            f"{prefix}_arm",
            "capsule",
            fromto=f"{support_x} {y} 3.15 {board_x} {y} 3.15",
            size="0.07",
            rgba="0.12 0.13 0.15 1",
        ),
        _geom(
            f"{prefix}_backboard",
            pos=f"{board_x} {y} 3.40",
            size="0.035 0.90 0.525",
            rgba="0.82 0.91 0.96 0.82",
            friction="0.55 0.005 0.0005",
        ),
    ]
    for name, pos, size in (
        ("board_top", (board_x, y, 3.91), (0.0425, 0.90, 0.0225)),
        ("board_bottom", (board_x, y, 2.89), (0.0425, 0.90, 0.0225)),
        ("board_left", (board_x, y - 0.88, 3.40), (0.0425, 0.0225, 0.525)),
        ("board_right", (board_x, y + 0.88, 3.40), (0.0425, 0.0225, 0.525)),
        ("target_top", (front_x, y, 3.65), (0.006, 0.295, 0.0175)),
        ("target_bottom", (front_x, y, 3.20), (0.006, 0.295, 0.0175)),
        ("target_left", (front_x, y - 0.2775, 3.425), (0.006, 0.0175, 0.225)),
        ("target_right", (front_x, y + 0.2775, 3.425), (0.006, 0.0175, 0.225)),
    ):
        items.append(
            _geom(
                f"{prefix}_{name}",
                pos=f"{pos[0]} {pos[1]} {pos[2]}",
                size=f"{size[0]} {size[1]} {size[2]}",
                rgba=WHITE,
                contype="0",
                conaffinity="0",
                mass="0",
            )
        )
    for index in range(32):
        a0, a1 = 2 * pi * index / 32, 2 * pi * (index + 1) / 32
        items.append(
            _geom(
                f"{prefix}_rim_{index}",
                "capsule",
                fromto=(
                    f"{rim_x + 0.225 * cos(a0)} {y + 0.225 * sin(a0)} 3.05 "
                    f"{rim_x + 0.225 * cos(a1)} {y + 0.225 * sin(a1)} 3.05"
                ),
                size="0.01",
                rgba="0.95 0.23 0.03 1",
                friction="0.5 0.003 0.0003",
            )
        )
    items += _basket_net(prefix, rim_x, y)
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
    items += _arc_lines(
        "basketball_three_e", x + 12.425, y, 6.75, pi - 1.20, pi + 1.20, width=0.025
    )
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
        f'<camera name="basketball_camera" pos="{x - 19} {y - 18} 12" xyaxes="0.69 -0.72 0 0.36 0.35 0.86"/>',
    ]
    return items


def _squash(x: float, y: float) -> list[str]:
    """Build a WSF-size enclosed singles court (front wall is at negative X)."""
    length, width = 9.75, 6.40
    front_x, back_x = x - length / 2, x + length / 2
    front_height, back_height = 4.57, 2.13
    wall_t = 0.10
    wall_color = "0.87 0.87 0.83 1"
    items = [_court_surface(Sport.SQUASH, x, y)]
    # The walls are collision geometry; floor markings remain visual-only.
    items += [
        _geom(
            "squash_front_wall",
            pos=f"{front_x - wall_t / 2} {y} {front_height / 2}",
            size=f"{wall_t / 2} {width / 2} {front_height / 2}",
            rgba=wall_color,
            friction="0.65 0.01 0.001",
        ),
        _geom(
            "squash_back_wall",
            pos=f"{back_x + wall_t / 2} {y} {back_height / 2}",
            size=f"{wall_t / 2} {width / 2} {back_height / 2}",
            rgba=wall_color,
            friction="0.65 0.01 0.001",
        ),
        _geom(
            "squash_side_wall_n",
            pos=f"{x} {y + width / 2 + wall_t / 2} {front_height / 2}",
            size=f"{length / 2} {wall_t / 2} {front_height / 2}",
            rgba=wall_color,
            friction="0.65 0.01 0.001",
        ),
        _geom(
            "squash_side_wall_s",
            pos=f"{x} {y - width / 2 - wall_t / 2} {front_height / 2}",
            size=f"{length / 2} {wall_t / 2} {front_height / 2}",
            rgba=wall_color,
            friction="0.65 0.01 0.001",
        ),
        _geom(
            "squash_tin",
            pos=f"{front_x + 0.004} {y} 0.48",
            size=f"0.004 {width / 2} 0.025",
            rgba="0.28 0.08 0.06 1",
            contype="0",
            conaffinity="0",
        ),
        _line_box("squash_short_line", (front_x + 4.26, y), (0.025, width / 2)),
        _line_box("squash_half_court_line", (x, y), (0.025, width / 2)),
        _ball(Sport.SQUASH, x + 1.2, y, 1.2),
        (
            f'<camera name="squash_camera" pos="{x + 4.3} {y} 2.3" '
            'xyaxes="0 1 0 0 0 1"/>'
        ),
    ]
    items += _rectangle_lines("squash_service_box_n", x + 0.80, y + 2.40, 1.60, 1.60, 0.035)
    items += _rectangle_lines("squash_service_box_s", x + 0.80, y - 2.40, 1.60, 1.60, 0.035)
    return items


BUILDERS = {
    Sport.TENNIS: _tennis,
    Sport.TABLE_TENNIS: _table_tennis,
    Sport.FOOTBALL: _football,
    Sport.BADMINTON: _badminton,
    Sport.BASKETBALL: _basketball,
    Sport.SQUASH: _squash,
}


CONTACTS = {
    # Values are (time constant, damping ratio). They were calibrated with the
    # regulation drop tests in tests/test_bounce.py at the 1 ms model timestep.
    Sport.TENNIS: ("0.012 0.10", "0.75 0.015 0.003"),
    Sport.TABLE_TENNIS: ("0.025 0.03", "0.35 0.004 0.001"),
    Sport.FOOTBALL: ("0.020 0.10", "0.80 0.018 0.015"),
    Sport.BADMINTON: ("0.008 1.00", "0.45 0.006 0.004"),
    Sport.BASKETBALL: ("0.020 0.075", "0.80 0.018 0.012"),
    Sport.SQUASH: ("0.016 0.09", "0.72 0.012 0.008"),
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


def _benchmark_contact_pair(sport: Sport) -> str:
    """The effector's own ball contact, when the sport calibrated one."""
    effector = BENCHMARK_EFFECTORS[sport]
    if effector.contact_solref is None or effector.contact_solimp is None:
        return ""
    solref = " ".join(str(value) for value in effector.contact_solref)
    solimp = " ".join(str(value) for value in effector.contact_solimp)
    return (
        f'<pair name="{sport.value}_benchmark_paddle_contact" '
        f'geom1="{effector.geom}" geom2="{sport.value}_ball_geom" condim="6" '
        f'friction="{effector.contact_friction}" solref="{solref}" solimp="{solimp}"/>'
    )


def build_xml(scene: str = "campus", *, benchmark_paddle: bool = False) -> str:
    """Build a complete MJCF string for one sport or the complete campus."""
    if scene not in SCENES:
        raise ValueError(f"unknown scene {scene!r}; expected one of {', '.join(SCENES)}")
    if benchmark_paddle and (
        scene == "campus" or Sport(scene) not in BENCHMARK_EFFECTORS
    ):
        available = ", ".join(sorted(sport.value for sport in BENCHMARK_EFFECTORS))
        raise ValueError(
            f"benchmark_paddle is not available for the {scene!r} scene; "
            f"it exists for: {available}"
        )

    if scene == "campus":
        sports = list(Sport)
        offsets = dict(CAMPUS_OFFSETS)
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
    if benchmark_paddle:
        world.append(_benchmark_paddle(Sport(scene)))
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
  <worldbody>{"".join(world)}</worldbody>
  <contact>{_contact_pairs(sports)}{_benchmark_contact_pair(Sport(scene)) if benchmark_paddle else ""}</contact>
</mujoco>"""


def build_model(scene: str = "campus", *, benchmark_paddle: bool = False) -> mujoco.MjModel:
    """Compile and return a MuJoCo model, raising with MJCF diagnostics on errors."""
    return mujoco.MjModel.from_xml_string(
        build_xml(scene, benchmark_paddle=benchmark_paddle)
    )
