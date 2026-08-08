"""Authoritative SI-unit dimensions and physical coefficients used by the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import pi


class Sport(str, Enum):
    TENNIS = "tennis"
    TABLE_TENNIS = "table_tennis"
    FOOTBALL = "football"
    BADMINTON = "badminton"
    BASKETBALL = "basketball"


SCENES = ("campus", *(sport.value for sport in Sport))


@dataclass(frozen=True)
class BallSpec:
    mass: float
    radius: float
    drag_coefficient: float
    restitution: float
    rolling_friction: float
    color: tuple[float, float, float, float]
    launch_velocity: tuple[float, float, float]

    @property
    def cross_section(self) -> float:
        return pi * self.radius**2


@dataclass(frozen=True)
class CourtSpec:
    length: float
    width: float
    surface: str
    color: tuple[float, float, float, float]


# Regulation midpoint values. Restitution is the effective normal coefficient used
# to calibrate MuJoCo contact softness; it is intentionally surface-specific in scene.py.
BALLS: dict[Sport, BallSpec] = {
    Sport.TENNIS: BallSpec(
        mass=0.0577,
        radius=0.0335,
        drag_coefficient=0.55,
        # PhysX contact restitution calibrated to the ITF 2.54 m drop test.
        restitution=0.777,
        rolling_friction=0.003,
        color=(0.78, 1.0, 0.08, 1.0),
        launch_velocity=(12.0, 0.5, 7.0),
    ),
    Sport.TABLE_TENNIS: BallSpec(
        mass=0.0027,
        radius=0.020,
        drag_coefficient=0.47,
        restitution=0.89,
        rolling_friction=0.001,
        color=(1.0, 0.45, 0.06, 1.0),
        launch_velocity=(5.5, 0.25, 2.3),
    ),
    Sport.FOOTBALL: BallSpec(
        mass=0.430,
        radius=0.110,
        drag_coefficient=0.25,
        restitution=0.72,
        rolling_friction=0.015,
        color=(0.96, 0.96, 0.96, 1.0),
        launch_velocity=(18.0, 1.0, 9.0),
    ),
    Sport.BADMINTON: BallSpec(
        mass=0.0050,
        radius=0.0135,  # cork/base collision radius; skirt is represented separately
        drag_coefficient=0.58,
        restitution=0.30,
        rolling_friction=0.004,
        color=(0.96, 0.96, 0.90, 1.0),
        launch_velocity=(18.0, 0.0, 8.0),
    ),
    Sport.BASKETBALL: BallSpec(
        mass=0.600,
        radius=0.120,
        drag_coefficient=0.50,
        # 0.795 yields the FIBA 1.035-1.085 m PhysX rebound window after
        # aerodynamic losses at the 240 Hz Isaac timestep.
        restitution=0.795,
        rolling_friction=0.012,
        color=(0.93, 0.32, 0.055, 1.0),
        launch_velocity=(8.5, 0.0, 8.3),
    ),
}


COURTS: dict[Sport, CourtSpec] = {
    Sport.TENNIS: CourtSpec(23.77, 10.97, "hard court", (0.10, 0.36, 0.58, 1.0)),
    Sport.TABLE_TENNIS: CourtSpec(8.0, 5.0, "indoor sport floor", (0.34, 0.16, 0.08, 1.0)),
    Sport.FOOTBALL: CourtSpec(105.0, 68.0, "natural grass", (0.12, 0.43, 0.16, 1.0)),
    Sport.BADMINTON: CourtSpec(13.40, 6.10, "synthetic mat", (0.10, 0.48, 0.40, 1.0)),
    Sport.BASKETBALL: CourtSpec(28.0, 15.0, "hardwood", (0.72, 0.47, 0.23, 1.0)),
}


AIR_DENSITY = 1.225
SHUTTLE_SKIRT_RADIUS = 0.033
SHUTTLE_CENTER_OF_PRESSURE_OFFSET = 0.045
