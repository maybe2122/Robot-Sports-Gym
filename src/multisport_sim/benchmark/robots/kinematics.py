"""Damped least-squares inverse kinematics over an attached arm's joints.

Two callers need this and they need it for different reasons, which is why it
lives on its own:

* an adapter that accepts :class:`~multisport_sim.benchmark.robot.ControlMode`
  ``EFFECTOR_POSE`` has to turn a pose into joint targets, and the solver it
  uses becomes part of what a submission measured;
* a scripted baseline plans its own swing and must do so with its *own* solver
  on its *own* model copy, exactly as a real policy would -- never by reaching
  into the simulator the benchmark is stepping.

The solver works on a scratch :class:`mujoco.MjData`, never on the data the
benchmark is integrating, so calling it can never perturb an episode.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class IKResult:
    """Outcome of one solve, including the residuals when it did not converge.

    A failed solve returns the best configuration it reached rather than
    raising: a controller running at 200 Hz should degrade into tracking, not
    into an exception mid-episode.
    """

    qpos: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    iterations: int
    converged: bool


class IKSolver:
    """Position-and-orientation IK for one site driven by one joint set.

    The solver clamps every iterate into the joint range, so a returned
    configuration is always commandable; it cannot "succeed" by leaving the
    robot's own limits.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        site_name: str,
        joint_names: Sequence[str],
        damping: float = 5e-2,
        position_tolerance_m: float = 2e-3,
        orientation_tolerance_rad: float = 2e-2,
        max_iterations: int = 100,
        step_scale: float = 0.6,
        stall_step_rad: float = 1e-7,
        position_low: Sequence[float] | None = None,
        position_high: Sequence[float] | None = None,
    ) -> None:
        self.model = model
        self.site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name))
        if self.site_id < 0:
            raise ValueError(f"model has no site named {site_name!r}")
        names = tuple(joint_names)
        if not names:
            raise ValueError("joint_names must not be empty")
        self.joint_ids = []
        for name in names:
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if joint_id < 0:
                raise ValueError(f"model has no joint named {name!r}")
            self.joint_ids.append(joint_id)
        self.joint_names = names
        self.qpos_adr = np.array(
            [int(model.jnt_qposadr[joint_id]) for joint_id in self.joint_ids], dtype=int
        )
        self.dof_adr = np.array(
            [int(model.jnt_dofadr[joint_id]) for joint_id in self.joint_ids], dtype=int
        )
        self.lower = np.array(
            [float(model.jnt_range[joint_id][0]) for joint_id in self.joint_ids]
        )
        self.upper = np.array(
            [float(model.jnt_range[joint_id][1]) for joint_id in self.joint_ids]
        )
        # A task may publish a *narrower* range than the hardware allows -- a
        # humanoid's striking arm is kept out of its own torso that way -- and a
        # solution outside it is not commandable, so the solver is bounded by it
        # here rather than having its answers clipped afterwards.  Widening is
        # refused: the model's range is the hardware's, and no task may exceed it.
        if position_low is not None:
            self.lower = self._tighten(position_low, self.lower, widen=np.greater_equal)
        if position_high is not None:
            self.upper = self._tighten(position_high, self.upper, widen=np.less_equal)
        if np.any(self.lower >= self.upper):
            raise ValueError("position_low must stay strictly below position_high")
        self.damping = float(damping)
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.max_iterations = int(max_iterations)
        self.step_scale = float(step_scale)
        # An unreachable target leaves damped least squares at a fixed point it
        # then creeps toward for the rest of the budget: on the Panda strike
        # targets the last 200 of 300 iterations moved no joint by more than
        # 1e-7 rad and cost ~6 ms of a 5 ms control period.  Stopping there
        # returns the same configuration to well under a microradian.
        self.stall_step_rad = float(stall_step_rad)
        self._scratch = mujoco.MjData(model)

    def _tighten(
        self,
        values: Sequence[float],
        bound: np.ndarray,
        *,
        widen: Any,
    ) -> np.ndarray:
        """Accept a per-joint limit override that stays inside the model's own."""
        override = np.asarray(values, dtype=float)
        if override.shape != bound.shape or not np.all(np.isfinite(override)):
            raise ValueError(
                "position limit overrides must hold one finite value per joint"
            )
        if not np.all(widen(override, bound)):
            raise ValueError(
                "position limit overrides must lie inside the model's joint ranges"
            )
        return override

    def forward(
        self,
        qpos: Sequence[float],
        *,
        context_qpos: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the site's world position and ``wxyz`` quaternion.

        ``context_qpos`` is a full-model configuration that everything *outside*
        the solved joints is placed at.  A fixed-base arm has no use for it --
        the model's own defaults already put its base where it is bolted -- but
        a floating-base robot's effector pose depends on where its pelvis
        currently stands, and solving against the asset's nominal base would
        answer a question about a robot that is not there.
        """
        data = self._scratch
        self._seed_context(context_qpos)
        data.qpos[self.qpos_adr] = np.asarray(qpos, dtype=float)
        mujoco.mj_kinematics(self.model, data)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[self.site_id])
        return data.site_xpos[self.site_id].copy(), quat

    def _seed_context(self, context_qpos: Sequence[float] | None) -> None:
        """Reset the scratch data, optionally to a supplied full configuration.

        Without a context the scratch starts from ``qpos0``, which is where a
        bolted-down arm always is.  With one, every joint the solver does not
        drive -- a floating base, a held posture -- starts where the robot
        actually is, so the solution is expressed in the robot's real stance
        rather than in the asset's nominal one.
        """
        data = self._scratch
        mujoco.mj_resetData(self.model, data)
        if context_qpos is None:
            return
        values = np.asarray(context_qpos, dtype=float)
        if values.shape != (self.model.nq,):
            raise ValueError(
                f"context_qpos must hold {self.model.nq} values, one per model "
                f"coordinate; received {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("context_qpos must contain finite numbers")
        data.qpos[:] = values

    def solve(
        self,
        target_position: Sequence[float],
        target_quaternion: Sequence[float] | None = None,
        *,
        target_axis: Sequence[float] | None = None,
        initial_qpos: Sequence[float] | None = None,
        context_qpos: Sequence[float] | None = None,
    ) -> IKResult:
        """Solve for ``target_position`` and, optionally, an orientation.

        Orientation can be constrained two ways and the choice matters:

        * ``target_quaternion`` fixes the site frame completely (three
          rotational constraints);
        * ``target_axis`` fixes only where the site's z axis points, leaving
          rotation *about* that axis free (two constraints).

        A paddle only cares about its face normal, so ``target_axis`` is the
        right constraint for a strike: it spends one fewer degree of freedom and
        leaves the arm's redundancy available for reaching, instead of forcing
        it into a contorted configuration to satisfy a spin angle nobody asked
        for.

        ``initial_qpos`` seeds the iteration; passing the robot's current
        configuration keeps successive solves continuous, which is what stops a
        tracking controller from jumping between IK branches mid-swing.

        ``context_qpos`` places everything the solver does not drive.  It is the
        difference between a bolted arm and a robot that stands: a humanoid's
        blade pose is measured from its pelvis, so a solve that ignored where
        the pelvis currently is would return joint angles for a robot standing
        somewhere else.
        """
        model = self.model
        data = self._scratch
        target_position = np.asarray(target_position, dtype=float)
        if target_position.shape != (3,) or not np.all(np.isfinite(target_position)):
            raise ValueError("target_position must be three finite numbers")
        if target_quaternion is not None and target_axis is not None:
            raise ValueError("pass target_quaternion or target_axis, not both")
        use_axis = target_axis is not None
        if use_axis:
            target_axis = np.asarray(target_axis, dtype=float)
            if target_axis.shape != (3,) or not np.all(np.isfinite(target_axis)):
                raise ValueError("target_axis must be three finite numbers")
            axis_norm = float(np.linalg.norm(target_axis))
            if axis_norm <= 1e-12:
                raise ValueError("target_axis must have non-zero norm")
            target_axis = target_axis / axis_norm
        use_orientation = target_quaternion is not None
        if use_orientation:
            target_quaternion = np.asarray(target_quaternion, dtype=float)
            if target_quaternion.shape != (4,) or not np.all(np.isfinite(target_quaternion)):
                raise ValueError("target_quaternion must be four finite numbers")
            norm = float(np.linalg.norm(target_quaternion))
            if norm <= 1e-12:
                raise ValueError("target_quaternion must have non-zero norm")
            target_quaternion = target_quaternion / norm

        self._seed_context(context_qpos)
        if initial_qpos is not None:
            seed = np.asarray(initial_qpos, dtype=float)
            if seed.shape != (len(self.joint_ids),):
                raise ValueError("initial_qpos must have one value per joint")
            data.qpos[self.qpos_adr] = np.clip(seed, self.lower, self.upper)

        rows = 6 if (use_orientation or use_axis) else 3
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        error = np.zeros(rows)
        site_quat = np.empty(4)
        conjugate = np.empty(4)
        difference = np.empty(4)
        rotation = np.empty(3)

        position_error = float("inf")
        orientation_error = 0.0
        iteration = 0
        for iteration in range(1, self.max_iterations + 1):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            position_residual = target_position - data.site_xpos[self.site_id]
            position_error = float(np.linalg.norm(position_residual))
            error[:3] = position_residual

            if use_orientation:
                mujoco.mju_mat2Quat(site_quat, data.site_xmat[self.site_id])
                mujoco.mju_negQuat(conjugate, site_quat)
                mujoco.mju_mulQuat(difference, target_quaternion, conjugate)
                mujoco.mju_quat2Vel(rotation, difference, 1.0)
                orientation_error = float(np.linalg.norm(rotation))
                error[3:] = rotation
            elif use_axis:
                # The site's third column is its z axis, which the paddle site
                # is built to align with the blade normal.
                current_axis = data.site_xmat[self.site_id].reshape(3, 3)[:, 2]
                cross = np.cross(current_axis, target_axis)
                sine = float(np.linalg.norm(cross))
                cosine = float(np.clip(current_axis @ target_axis, -1.0, 1.0))
                angle = float(np.arctan2(sine, cosine))
                if sine > 1e-9:
                    rotation[:] = cross / sine * angle
                elif cosine < 0.0:
                    # Exactly anti-parallel: any perpendicular axis turns it
                    # around, and the next iteration leaves the singularity.
                    perpendicular = np.array([0.0, 0.0, 1.0])
                    if abs(float(current_axis @ perpendicular)) > 0.95:
                        perpendicular = np.array([0.0, 1.0, 0.0])
                    axis = np.cross(current_axis, perpendicular)
                    rotation[:] = axis / np.linalg.norm(axis) * angle
                else:
                    rotation[:] = 0.0
                orientation_error = abs(angle)
                error[3:] = rotation

            if (
                position_error <= self.position_tolerance_m
                and orientation_error <= self.orientation_tolerance_rad
            ):
                return IKResult(
                    qpos=data.qpos[self.qpos_adr].copy(),
                    position_error_m=position_error,
                    orientation_error_rad=orientation_error,
                    iterations=iteration,
                    converged=True,
                )

            mujoco.mj_jacSite(model, data, jacp, jacr, self.site_id)
            jacobian = np.vstack((jacp[:, self.dof_adr], jacr[:, self.dof_adr]))[:rows]
            # Damped least squares: (J J^T + lambda^2 I)^-1 stays finite at a
            # singularity, where the pseudo-inverse would command an unbounded
            # joint velocity.
            gram = jacobian @ jacobian.T
            gram[np.diag_indices_from(gram)] += self.damping**2
            try:
                delta = jacobian.T @ np.linalg.solve(gram, error)
            except np.linalg.LinAlgError:  # pragma: no cover - guarded by damping
                break
            previous = data.qpos[self.qpos_adr].copy()
            data.qpos[self.qpos_adr] = np.clip(
                previous + self.step_scale * delta, self.lower, self.upper
            )
            if float(np.max(np.abs(data.qpos[self.qpos_adr] - previous))) < self.stall_step_rad:
                break

        return IKResult(
            qpos=data.qpos[self.qpos_adr].copy(),
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            iterations=iteration,
            converged=False,
        )


def blade_quaternion(normal: Sequence[float]) -> np.ndarray:
    """Return a ``wxyz`` quaternion whose local z axis points along ``normal``.

    The paddle site is built with its z axis along the blade normal, so aiming
    the blade is a single-axis constraint: the rotation about the normal is
    left free for the solver to use as redundancy.
    """
    direction = np.asarray(normal, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise ValueError("normal must have non-zero norm")
    direction = direction / length
    # Pick any reference not parallel to the normal, then build an orthonormal
    # frame around it.
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(direction @ reference)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(reference, direction)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(direction, x_axis)
    matrix = np.column_stack((x_axis, y_axis, direction)).reshape(9)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, matrix)
    return quat


__all__ = ["IKResult", "IKSolver", "blade_quaternion"]
