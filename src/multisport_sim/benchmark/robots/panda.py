"""Franka Emika Panda adapter for the table-tennis return task.

The arm is attached to the existing table-tennis scene at load time with
:class:`mujoco.MjSpec`, which keeps the upstream Menagerie files untouched and
gives every Panda joint, actuator and geom the ``rb_`` prefix.  A benchmark
paddle is added as a child of the arm's own ``attachment`` body, so the blade is
a rigid part of the kinematic chain: contact with the ball is a real physical
strike produced by the arm's dynamics, not a mocap teleport.

Why this arm.  MuJoCo Menagerie and Isaac Lab both ship a Panda under a clear
licence, which is the only reason the same robot can run the same task on both
backends without a second asset pipeline.  Its seven joints also leave one
degree of redundancy after the blade's position and normal are fixed, which a
six-axis arm would not.

Its cost is honest and documented: the datasheet caps joint speed at 2.175 rad/s
for the proximal joints, so the fastest shots in the bank are near or past what
this embodiment can intercept.  That is a property of the robot, and the
benchmark reports it as a score rather than hiding it by slowing the ball down.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from ...scene import build_xml
from ...specs import Sport
from ..assets import FRANKA_PANDA, AssetSource, require_asset
from ..robot import (
    ControlMode,
    EffectorPoseCommand,
    JointCommand,
    JointLimits,
    RobotCommand,
    RobotObservation,
    SafetyLimits,
    UnsupportedControlMode,
    WorkspaceBox,
)
from ..sensors import WORLD_FRAME, CameraSpec
from .kinematics import IKSolver

IK_SETTINGS = {"damping": 0.02, "max_iterations": 300, "step_scale": 0.9}
"""Solver settings that converge on 100% of the measured strike plane.

They are shared by the adapter and by any baseline that plans its own swing, so
two controllers using "the reference IK" mean the same thing.
"""

PREFIX = "rb_"
JOINT_NAMES = tuple(f"{PREFIX}joint{index}" for index in range(1, 8))
ACTUATOR_NAMES = tuple(f"{PREFIX}actuator{index}" for index in range(1, 8))
PADDLE_BODY = f"{PREFIX}benchmark_paddle"
PADDLE_GEOM = f"{PREFIX}benchmark_paddle_blade"
PADDLE_SITE = f"{PREFIX}benchmark_paddle_center"
PEDESTAL_GEOM = f"{PREFIX}pedestal"

# Franka Emika Panda datasheet limits.  They are absent from the MJCF -- which
# declares only position ranges and actuator force ranges -- so the benchmark
# supplies them here and every safety check reads them from one place.
PANDA_VELOCITY_LIMIT = (2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100)
PANDA_TORQUE_LIMIT = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)

# Ready pose: blade parked in front of the arm, face normal toward the
# opponent.  It is not an arbitrary "home" -- it is the configuration that
# minimises the *worst-case* joint travel time to any point of the strike plane
# the frozen shot bank actually crosses, because travel time is what decides
# whether this arm can intercept at all.  Derived by
# ``scripts/calibrate_reachability.py`` and frozen here so every reset is
# bit-identical across machines.
#
# The objective is measured against the solver's *actual* behaviour when seeded
# from this pose, not against an idealised solution set.  That distinction is
# the whole difficulty: a redundant arm has many configurations reaching the
# same blade pose, and a ready pose scored against solutions the solver would
# never return from it looks twice as good as it is.  Worst-case travel here is
# 0.51 s against a shortest available intercept window of 0.53 s.
PANDA_READY_QPOS = (
    -1.50953,
    1.51280,
    1.65599,
    -2.62808,
    -0.94185,
    1.87744,
    2.89730,
)

# Blade normal along the arm's local z at the paddle site; see ``_paddle_site``.
BLADE_HALF_EXTENTS = (0.085, 0.008, 0.10)


@dataclass(frozen=True)
class PandaMount:
    """Where the arm is bolted down, in the task frame.

    The default was chosen by the reachability calibration in
    ``scripts/calibrate_reachability.py``: it covers 97.5% of the strike plane
    that the frozen shot bank actually crosses.  Moving the mount changes what
    the task measures, so it is versioned with the task configuration rather
    than being a runtime knob.
    """

    position: tuple[float, float, float] = (-1.95, 0.0, 0.75)
    pedestal_radius_m: float = 0.13

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.position)
        if len(values) != 3 or not all(np.isfinite(values)):
            raise ValueError("position must be three finite numbers")
        object.__setattr__(self, "position", values)
        radius = float(self.pedestal_radius_m)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("pedestal_radius_m must be greater than zero")
        object.__setattr__(self, "pedestal_radius_m", radius)
        if values[2] <= 0.0:
            raise ValueError("the mount must stand above the floor")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "pedestal_radius_m": self.pedestal_radius_m,
        }


def build_panda_table_tennis_model(
    *,
    mount: PandaMount | None = None,
    source: AssetSource = FRANKA_PANDA,
    cameras: Sequence[CameraSpec] = (),
) -> mujoco.MjModel:
    """Compile the table-tennis scene with a Panda and its paddle attached.

    ``cameras`` adds one MuJoCo camera per declared :class:`CameraSpec`, placed
    from the spec rather than from the scene file.  The spec stays the single
    source of truth for the vision track: a report that names a camera pose and
    field of view names the pose and field of view that were rendered.

    Raises :class:`~multisport_sim.benchmark.assets.AssetUnavailableError` when
    the Menagerie checkout is missing, with the commands that fix it.
    """
    placement = mount or PandaMount()
    model_path = require_asset(source)

    spec = mujoco.MjSpec.from_string(build_xml(Sport.TABLE_TENNIS.value))
    arm = mujoco.MjSpec.from_file(str(model_path))
    # A compiled model holds one impratio.  The upstream arm asks for 10 and the
    # sports scene for 1; attaching would silently keep the scene's value and
    # warn.  Resolving it explicitly here makes the choice reviewable: the ball's
    # calibrated bounce and spin depend on the scene value, and the arm has no
    # frictional contact in this task, so the scene wins.  The decision is
    # recorded in the asset's modification list.
    arm.option.impratio = spec.option.impratio
    _attach_paddle(arm)
    frame = spec.worldbody.add_frame(pos=list(placement.position))
    frame.attach_body(arm.worldbody.first_body(), PREFIX, "")

    _add_pedestal(spec, placement)
    _add_blade_contact(spec)
    # After attachment, so a body-mounted camera can name the arm's own frames.
    _add_cameras(spec, cameras)
    return spec.compile()


def _add_cameras(spec: mujoco.MjSpec, cameras: Sequence[CameraSpec]) -> None:
    """Place one MuJoCo camera per declared spec, on the world or on a body."""
    for camera in cameras:
        if not isinstance(camera, CameraSpec):
            raise TypeError("cameras must be CameraSpec instances")
        if camera.mount == WORLD_FRAME:
            parent = spec.worldbody
        else:
            parent = next((body for body in spec.bodies if body.name == camera.mount), None)
            if parent is None:
                known = ", ".join(sorted(body.name for body in spec.bodies if body.name))
                raise ValueError(
                    f"camera {camera.name!r} is mounted on unknown frame "
                    f"{camera.mount!r}; the model has: {known}"
                )
        handle = parent.add_camera()
        handle.name = camera.name
        handle.pos = list(camera.position)
        handle.quat = list(camera.quaternion)
        handle.fovy = camera.fovy_deg


CAMERA_MOUNT_PADDLE = PADDLE_BODY
"""The arm frame an on-board camera rides; named here so specs need no prefix."""


# Blade-rubber contact, calibrated by the drop test in
# ``tests/test_robot_scene.py``: a ball dropped on a fixed blade rebounds with a
# coefficient of restitution of roughly 0.82-0.86, which is where measured
# inverted rubber sits.  Without a dedicated pair the blade would inherit the
# scene's default damping ratio of 0.7 and return the ball at e = 0.17 -- the
# arm would be swinging a piece of foam, and no achievable blade speed would
# produce a legal return.
#
# The pair is added to the *robot* model only.  The v0 mocap fixture keeps the
# contact it was frozen with, because changing it would silently move every v0
# score; that the two tasks differ here is one more reason they are versioned
# separately.
BLADE_CONTACT_SOLREF = (0.023, 0.042)
# solimp takes five values in a pair: dmin, dmax, width, midpoint, power.
BLADE_CONTACT_SOLIMP = (0.96, 0.99, 0.001, 0.5, 2.0)
BLADE_CONTACT_FRICTION = (0.85, 0.85, 0.01, 0.001, 0.001)


def _add_blade_contact(spec: mujoco.MjSpec) -> None:
    """Give the ball-blade pair rubber's restitution instead of the default."""
    pair = spec.add_pair()
    pair.name = "robot_paddle_ball_contact"
    pair.geomname1 = PADDLE_GEOM
    pair.geomname2 = f"{Sport.TABLE_TENNIS.value}_ball_geom"
    pair.condim = 6
    pair.friction = list(BLADE_CONTACT_FRICTION)
    pair.solref = list(BLADE_CONTACT_SOLREF)
    pair.solimp = list(BLADE_CONTACT_SOLIMP)


def _attach_paddle(arm: mujoco.MjSpec) -> None:
    """Add the benchmark blade to the arm's flange, before attachment.

    The blade is an ellipsoid that is thin along its body y axis, so the strike
    face normal is that axis.  The site is rotated -90 degrees about x, which
    puts the *site's* z axis along the blade normal; aiming the paddle is then a
    single-axis constraint for the IK solver.
    """
    attachment = next((body for body in arm.bodies if body.name == "attachment"), None)
    if attachment is None:  # pragma: no cover - upstream asset contract
        raise RuntimeError("the Panda asset no longer exposes an 'attachment' body")

    blade = attachment.add_body(name="benchmark_paddle", pos=[0.0, 0.0, 0.13])
    geom = blade.add_geom()
    geom.name = "benchmark_paddle_blade"
    geom.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    geom.size = list(BLADE_HALF_EXTENTS)
    geom.rgba = [0.82, 0.05, 0.035, 1.0]
    geom.friction = [0.85, 0.01, 0.001]
    # A regulation blade plus rubber is about 170 g; the arm has to accelerate
    # it, so it carries real mass rather than the MJCF default density.
    geom.mass = 0.17

    site = blade.add_site()
    site.name = "benchmark_paddle_center"
    site.size = [0.005, 0.005, 0.005]
    site.quat = [0.7071067811865476, -0.7071067811865476, 0.0, 0.0]


def _add_pedestal(spec: mujoco.MjSpec, mount: PandaMount) -> None:
    """Stand the arm on a fixed column instead of floating it in mid-air."""
    x, y, z = mount.position
    geom = spec.worldbody.add_geom()
    geom.name = PEDESTAL_GEOM
    geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    geom.pos = [x, y, z / 2.0]
    geom.size = [mount.pedestal_radius_m, z / 2.0, 0.0]
    geom.rgba = [0.20, 0.21, 0.24, 1.0]


class PandaTableTennisAdapter:
    """A :class:`~multisport_sim.benchmark.robot.RobotAdapter` over a MuJoCo Panda.

    The adapter owns no simulation loop: it reads and writes one
    :class:`mujoco.MjData` that a backend steps. Position and velocity commands
    use the asset's affine position actuators; torque commands temporarily
    configure those general actuators as unit-gain motors, bounded by the same
    force ranges. ``EFFECTOR_POSE`` is served by this adapter's own damped
    least-squares solver. Values are never silently reinterpreted with another
    mode's units.
    """

    robot_id = "franka-panda-tabletennis-v1"

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        mount: PandaMount | None = None,
        workspace: WorkspaceBox | None = None,
        ready_qpos: Sequence[float] = PANDA_READY_QPOS,
        source: AssetSource = FRANKA_PANDA,
    ) -> None:
        self.model = model
        self.data = data
        self.mount = mount or PandaMount()
        self.source = source
        self._joint_ids = tuple(
            self._require(mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES
        )
        self._actuator_ids = tuple(
            self._require(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES
        )
        actuator_index = np.asarray(self._actuator_ids, dtype=int)
        # Menagerie uses affine ``general`` actuators as joint servos. Torque
        # mode changes these model arrays, so keep exact copies to make every
        # transition back to a servo bit-for-bit reversible.
        self._position_gainprm = model.actuator_gainprm[actuator_index].copy()
        self._position_biasprm = model.actuator_biasprm[actuator_index].copy()
        self._position_ctrlrange = model.actuator_ctrlrange[actuator_index].copy()
        self._active_control_mode = ControlMode.JOINT_POSITION
        self._qpos_adr = np.array(
            [int(model.jnt_qposadr[joint]) for joint in self._joint_ids], dtype=int
        )
        self._dof_adr = np.array(
            [int(model.jnt_dofadr[joint]) for joint in self._joint_ids], dtype=int
        )
        self._site_id = self._require(mujoco.mjtObj.mjOBJ_SITE, PADDLE_SITE)
        self.paddle_geom_id = self._require(mujoco.mjtObj.mjOBJ_GEOM, PADDLE_GEOM)
        self._robot_geom_ids = frozenset(self._collect_robot_geoms())

        ready = np.asarray(ready_qpos, dtype=float)
        if ready.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(ready)):
            raise ValueError("ready_qpos must hold one finite value per joint")
        self._limits = JointLimits(
            joint_names=JOINT_NAMES,
            position_low=tuple(float(model.jnt_range[j][0]) for j in self._joint_ids),
            position_high=tuple(float(model.jnt_range[j][1]) for j in self._joint_ids),
            velocity_limit=PANDA_VELOCITY_LIMIT,
            torque_limit=PANDA_TORQUE_LIMIT,
        )
        self.ready_qpos = self._limits.clamp_positions(ready)
        self._safety = SafetyLimits(joints=self._limits, workspace=workspace)
        self._solver: IKSolver | None = None
        self._contact_labels: dict[int, str] = {}

    # -- identity -----------------------------------------------------------

    @property
    def joint_names(self) -> tuple[str, ...]:
        return JOINT_NAMES

    @property
    def dof(self) -> int:
        return len(JOINT_NAMES)

    @property
    def effector_name(self) -> str:
        return "racket"

    @property
    def control_modes(self) -> frozenset[ControlMode]:
        return frozenset(
            {
                ControlMode.JOINT_POSITION,
                ControlMode.JOINT_VELOCITY,
                ControlMode.JOINT_TORQUE,
                ControlMode.EFFECTOR_POSE,
            }
        )

    @property
    def safety_limits(self) -> SafetyLimits:
        return self._safety

    @property
    def joint_limits(self) -> JointLimits:
        return self._limits

    @property
    def solver(self) -> IKSolver:
        """The adapter's own IK solver, built on first use."""
        if self._solver is None:
            self._solver = IKSolver(
                self.model,
                site_name=PADDLE_SITE,
                joint_names=JOINT_NAMES,
                **IK_SETTINGS,
            )
        return self._solver

    def set_contact_labels(self, labels: dict[int, str]) -> None:
        """Tell the adapter which geom ids carry which semantic label.

        The backend owns scene semantics, so it hands the mapping down instead
        of the adapter guessing from geom names -- which is exactly the coupling
        the rule engine forbids.
        """
        self._contact_labels = dict(labels)

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Place the arm in its frozen ready pose with zero velocity."""
        self._configure_actuators(ControlMode.JOINT_POSITION)
        self.data.qpos[self._qpos_adr] = self.ready_qpos
        self.data.qvel[self._dof_adr] = 0.0
        self.data.ctrl[list(self._actuator_ids)] = self.ready_qpos

    def observe(self) -> RobotObservation:
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, self.data.site_xmat[self._site_id])
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._site_id)
        linear_velocity = jacp @ self.data.qvel
        return RobotObservation(
            time_s=float(self.data.time),
            joint_positions=tuple(float(v) for v in self.data.qpos[self._qpos_adr]),
            joint_velocities=tuple(float(v) for v in self.data.qvel[self._dof_adr]),
            applied_torque=tuple(
                float(self.data.qfrc_actuator[dof]) for dof in self._dof_adr
            ),
            effector_position=tuple(float(v) for v in self.data.site_xpos[self._site_id]),
            effector_quaternion=tuple(float(v) for v in quaternion),
            effector_linear_velocity=tuple(float(v) for v in linear_velocity),
            contacts=self.robot_contacts(),
        )

    def robot_contacts(self) -> tuple[str, ...]:
        """Semantic labels of everything the robot's own geoms are touching.

        Ball contacts are excluded: striking the ball is the task, not a safety
        breach.  The blade hitting the table is.
        """
        labels: set[str] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if float(contact.dist) > 0.0:
                continue
            if geom1 in self._robot_geom_ids:
                other = geom2
            elif geom2 in self._robot_geom_ids:
                other = geom1
            else:
                continue
            if other in self._robot_geom_ids:
                labels.add("self")
                continue
            label = self._contact_labels.get(other)
            if label is not None and label != "ball":
                labels.add(label)
        return tuple(sorted(labels))

    def apply(self, command: RobotCommand) -> None:
        """Send one command to the arm's actuators."""
        if isinstance(command, JointCommand):
            if command.mode not in self.control_modes:
                raise UnsupportedControlMode(
                    self.robot_id, command.mode, self.control_modes
                )
            if len(command) != self.dof:
                raise ValueError(
                    f"command has {len(command)} targets but the robot has {self.dof} joints"
                )
            self._configure_actuators(command.mode)
            if command.mode is ControlMode.JOINT_POSITION:
                targets = self._limits.clamp_positions(command.targets)
            elif command.mode is ControlMode.JOINT_VELOCITY:
                targets = self._integrate_velocity(command.targets)
            else:
                limits = np.asarray(self._limits.torque_limit)
                targets = tuple(
                    float(value)
                    for value in np.clip(np.asarray(command.targets), -limits, limits)
                )
            self.data.ctrl[list(self._actuator_ids)] = targets
            return

        if isinstance(command, EffectorPoseCommand):
            if ControlMode.EFFECTOR_POSE not in self.control_modes:  # pragma: no cover
                raise UnsupportedControlMode(
                    self.robot_id, ControlMode.EFFECTOR_POSE, self.control_modes
                )
            self._configure_actuators(ControlMode.EFFECTOR_POSE)
            current = self.data.qpos[self._qpos_adr].copy()
            result = self.solver.solve(
                command.position, command.quaternion, initial_qpos=current
            )
            self.data.ctrl[list(self._actuator_ids)] = self._limits.clamp_positions(
                result.qpos
            )
            return

        raise TypeError(
            "PandaTableTennisAdapter expects a JointCommand or an EffectorPoseCommand; "
            f"received {type(command).__name__}"
        )

    def _configure_actuators(self, mode: ControlMode) -> None:
        """Select servo or direct-torque semantics for the shared actuators.

        Menagerie's servo computes ``gain * ctrl - kp * q - kd * qvel``.
        Direct torque instead needs unit gain and zero bias. Each adapter owns
        its compiled model, and restoring captured arrays prevents model drift
        across repeated mode changes.
        """
        if mode is self._active_control_mode:
            return
        actuator_index = np.asarray(self._actuator_ids, dtype=int)
        if mode is ControlMode.JOINT_TORQUE:
            self.model.actuator_gainprm[actuator_index] = 0.0
            self.model.actuator_gainprm[actuator_index, 0] = 1.0
            self.model.actuator_biasprm[actuator_index] = 0.0
            limits = np.asarray(self._limits.torque_limit)
            self.model.actuator_ctrlrange[actuator_index, 0] = -limits
            self.model.actuator_ctrlrange[actuator_index, 1] = limits
        else:
            self.model.actuator_gainprm[actuator_index] = self._position_gainprm
            self.model.actuator_biasprm[actuator_index] = self._position_biasprm
            self.model.actuator_ctrlrange[actuator_index] = self._position_ctrlrange
        self._active_control_mode = mode

    def _integrate_velocity(self, rates: Sequence[float]) -> tuple[float, ...]:
        """Turn a velocity command into the next position setpoint.

        The Panda MJCF exposes position actuators only.  Rather than pretend a
        velocity mode exists at the hardware level, the adapter integrates the
        request over one physics step against the *datasheet* speed limit, so a
        policy cannot obtain motion the real arm could not deliver.
        """
        timestep = float(self.model.opt.timestep)
        current = self.data.qpos[self._qpos_adr]
        limited = np.clip(
            np.asarray(rates, dtype=float),
            -np.asarray(self._limits.velocity_limit),
            np.asarray(self._limits.velocity_limit),
        )
        return self._limits.clamp_positions(current + limited * timestep)

    # -- reporting ----------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "effector": self.effector_name,
            "dof": self.dof,
            "joint_names": list(JOINT_NAMES),
            "control_modes": sorted(mode.value for mode in self.control_modes),
            "default_control_mode": ControlMode.JOINT_POSITION.value,
            "ready_qpos": list(self.ready_qpos),
            "mount": self.mount.to_dict(),
            "effector_pose_solver": "damped-least-squares-ik",
            "asset": self.source.to_dict(),
            "safety_limits": self._safety.to_dict(),
        }

    # -- internals ----------------------------------------------------------

    def _require(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = int(mujoco.mj_name2id(self.model, object_type, name))
        if object_id < 0:
            raise RuntimeError(f"the robot model is missing {name!r}")
        return object_id

    def _collect_robot_geoms(self) -> list[int]:
        """Every geom belonging to the arm, found by its attachment prefix."""
        found: list[int] = []
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body is not None and body.startswith(PREFIX):
                found.append(geom_id)
        if not found:  # pragma: no cover - protects against a prefix change
            raise RuntimeError("no robot geoms were found in the compiled model")
        return found


__all__ = [
    "ACTUATOR_NAMES",
    "BLADE_HALF_EXTENTS",
    "JOINT_NAMES",
    "PADDLE_GEOM",
    "PADDLE_SITE",
    "PANDA_READY_QPOS",
    "PANDA_TORQUE_LIMIT",
    "PANDA_VELOCITY_LIMIT",
    "PEDESTAL_GEOM",
    "PREFIX",
    "PandaMount",
    "PandaTableTennisAdapter",
    "build_panda_table_tennis_model",
]
