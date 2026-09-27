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
        self._free_joints: dict[int, tuple[int, int]] = {}
        for sport in Sport:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{sport.value}_ball")
            if body_id >= 0:
                self._body_ids[sport] = body_id
                joint = int(model.body_jntadr[body_id])
                if joint >= 0 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                    self._free_joints[body_id] = (
                        int(model.jnt_qposadr[joint]),
                        int(model.jnt_dofadr[joint]),
                    )

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
        """Drag of the skirt, evaluated where it acts: at the centre of pressure.

        The flow a shuttle's skirt sees is the velocity of the centre of
        pressure, ``v_com + omega x r_cp``, not of its centre of mass.  The
        difference is the shuttle's pitch damping: a tumbling shuttle's skirt
        sweeps through the air and is resisted.  Evaluating drag at the centre
        of mass drops that damping, and the stabilising torque is then a stiff
        undamped spring -- struck at 25 m/s it tumbled into a numerical blow-up
        within 5 ms.

        State is read from ``qpos``/``qvel`` rather than ``xmat``/``cvel``,
        which still hold the previous step's pre-integration values when this
        runs; a one-step-stale restoring torque on a spring this stiff is
        negative damping.
        """
        del velocity
        joint = self._free_joints.get(body_id)
        if joint is None:  # pragma: no cover - every shuttle has a free joint
            return np.zeros(3), np.zeros(3)
        qpos_address, dof_address = joint
        quaternion = data.qpos[qpos_address + 3 : qpos_address + 7]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion / np.linalg.norm(quaternion))
        rotation = rotation.reshape(3, 3)
        # Free-joint angular velocity is in the body frame; the linear part is
        # the velocity of the body origin, which is not the centre of mass.
        angular_velocity = rotation @ data.qvel[dof_address + 3 : dof_address + 6]
        com_offset = rotation @ self.model.body_ipos[body_id]
        com_velocity = data.qvel[dof_address : dof_address + 3] + np.cross(
            angular_velocity, com_offset
        )

        # Local +Z points from cork to skirt. Broadside area is higher than nose-on area.
        axis = rotation[:, 2]
        cp_offset = axis * SHUTTLE_CENTER_OF_PRESSURE_OFFSET
        flow = com_velocity + np.cross(angular_velocity, cp_offset) - np.asarray(
            self.atmosphere.wind
        )
        speed = float(np.linalg.norm(flow))
        if speed < 1e-9:
            return np.zeros(3), np.zeros(3)
        axial = abs(float(np.dot(axis, flow / speed)))
        full_area = pi * SHUTTLE_SKIRT_RADIUS**2
        projected_area = full_area * (0.35 + 0.65 * axial)
        force = quadratic_drag(
            flow,
            projected_area,
            BALLS[Sport.BADMINTON].drag_coefficient,
            self.atmosphere.density,
        )
        # Force applied at the centre of pressure = force at the centre of
        # mass plus this moment.
        stabilising_torque = np.cross(cp_offset, force)
        angular_damping = -1.5e-5 * angular_velocity
        return force, stabilising_torque + angular_damping