"""Aerodynamic forces omitted by rigid-body contact models.

MuJoCo handles gravity, inertia, friction and compliant contact. This module adds
quadratic air drag, a bounded Magnus lift term for spinning balls, and the large,
orientation-dependent drag/stabilising torque of a shuttlecock.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import mujoco
import numpy as np

from .specs import (
    AIR_DENSITY,
    BALLS,
    SHUTTLE_CENTER_OF_PRESSURE_OFFSET,
    SHUTTLE_SKIRT_RADIUS,
    Sport,
)


@dataclass
class Atmosphere:
    density: float = AIR_DENSITY
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    magnus_scale: float = 0.20


def quadratic_drag(
    velocity: np.ndarray, area: float, coefficient: float, density: float = AIR_DENSITY
) -> np.ndarray:
    """Return ``-0.5 rho Cd A |v| v`` in newtons."""
    speed = float(np.linalg.norm(velocity))
    if speed < 1e-9:
        return np.zeros(3)
    return -0.5 * density * coefficient * area * speed * velocity


class Aerodynamics:
    """Apply passive aerodynamic loads to every named sports projectile in a model."""

    def __init__(self, model: mujoco.MjModel, atmosphere: Atmosphere | None = None):
        self.model = model
        self.atmosphere = atmosphere or Atmosphere()
        self._body_ids: dict[Sport, int] = {}
        for sport in Sport:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{sport.value}_ball")
            if body_id >= 0:
                self._body_ids[sport] = body_id

    def apply(self, data: mujoco.MjData) -> None:
        wind = np.asarray(self.atmosphere.wind)
        for sport, body_id in self._body_ids.items():
            velocity = data.cvel[body_id, 3:6] - wind
            angular_velocity = data.cvel[body_id, 0:3]
            spec = BALLS[sport]

            if sport is Sport.BADMINTON:
                force, torque = self._shuttle_loads(data, body_id, velocity)
            else:
                force = quadratic_drag(
                    velocity,
                    spec.cross_section,
                    spec.drag_coefficient,
                    self.atmosphere.density,
                )
                torque = np.zeros(3)
                speed = float(np.linalg.norm(velocity))
                if speed > 0.1:
                    # Spin parameter S=r*omega/v; Cl is bounded to avoid unstable loads.
                    spin_axis = np.cross(angular_velocity, velocity)
                    cl = min(0.35, self.atmosphere.magnus_scale * spec.radius * np.linalg.norm(angular_velocity) / speed)
                    spin_norm = float(np.linalg.norm(spin_axis))
                    if spin_norm > 1e-9:
                        force += (
                            0.5
                            * self.atmosphere.density
                            * spec.cross_section
                            * speed**2
                            * cl
                            * spin_axis
                            / spin_norm
                        )

            data.xfrc_applied[body_id, :3] += force
            data.xfrc_applied[body_id, 3:] += torque

    def _shuttle_loads(
        self, data: mujoco.MjData, body_id: int, velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        speed = float(np.linalg.norm(velocity))
        if speed < 1e-9:
            return np.zeros(3), np.zeros(3)

        # Local +Z points from cork to skirt. Broadside area is higher than nose-on area.
        axis = data.xmat[body_id].reshape(3, 3)[:, 2]
        flow_direction = velocity / speed
        axial = abs(float(np.dot(axis, flow_direction)))
        full_area = pi * SHUTTLE_SKIRT_RADIUS**2
        projected_area = full_area * (0.35 + 0.65 * axial)
        force = quadratic_drag(
            velocity,
            projected_area,
            BALLS[Sport.BADMINTON].drag_coefficient,
            self.atmosphere.density,
        )

        # Drag acts behind the centre of mass, naturally pointing the cork into flight.
        cp_offset = axis * SHUTTLE_CENTER_OF_PRESSURE_OFFSET
        stabilising_torque = np.cross(cp_offset, force)
        angular_damping = -1.5e-5 * data.cvel[body_id, 0:3]
        return force, stabilising_torque + angular_damping

