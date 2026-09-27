"""Unitree G1 adapter for the table-tennis return task.

This is the benchmark's second embodiment, and it exists to prove a claim the
first one could only assert: that the task, the judge, the frozen shot bank and
the metrics do not know what robot is playing.  Everything below the adapter is
shared with the Panda; what changes is the number of joints, their limits, where
the blade hangs and how far it can reach.

Three decisions here are load-bearing and none of them is free.

**The pelvis is fixed.**  The upstream model stands on a free joint, and a
free-standing G1 handed a table-tennis task would be scored on whether it stays
upright, which is a different benchmark.  The free joint is deleted at load
time, so the pelvis is rigidly held at the mount and the feet rest on the floor.
That makes this a *fixed-base humanoid upper body*, and the task id says so.
Locomotion and balance are out of scope for this task, not solved by it.

**The policy commands ten joints, not twenty-nine.**  The action space is the
waist (3) and the right arm (7).  The legs and the left arm are present with
their real mass, inertia and collision geometry -- they are part of what the
right arm has to accelerate and must not collide with -- but they are held at
the asset's own ``stand`` keyframe by their position actuators.  A benchmark
that published all 29 joints while 19 of them can only hold still would be
inviting policies to learn a 19-dimensional no-op.

**The observation is 39 numbers, and that is the point.**  The Panda's is 33.
Neither is written down anywhere: both come from
:meth:`~..task_config.TableTennisReturnTaskConfig.observation_layout`, whose
field *names* are identical across the two robots and whose field *sizes* are
not.  See ``docs/POLICY_INTERFACE.md``.

The honest cost, measured rather than asserted: a fixed-base G1's right hand
sweeps a far smaller volume than a Panda's, and the reachability calibration in
``scripts/calibrate_reachability.py --robot g1`` reports how much of the strike
plane it actually covers.  A low score there is a property of this embodiment on
this bank, and the benchmark reports it as a number instead of moving the ball.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from ...scene import build_xml
from ...specs import Sport
from ..assets import UNITREE_G1, AssetSource, require_asset
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

IK_SETTINGS = {"damping": 0.05, "max_iterations": 300, "step_scale": 0.7}
"""Solver settings for the G1's arm.

Damped more heavily than the Panda's.  Ten joints driving a blade whose reachable
set barely contains the strike plane leaves the solver working near the boundary
of that set, where a lightly damped step oscillates instead of converging.
"""

PREFIX = "g1_"

# The asset names actuators after the joints they drive, so one tuple serves
# both once the attachment prefix is applied.
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
TASK_JOINTS = WAIST_JOINTS + RIGHT_ARM_JOINTS
JOINT_NAMES = tuple(f"{PREFIX}{name}" for name in TASK_JOINTS)
ACTUATOR_NAMES = JOINT_NAMES

WRIST_BODY = "right_wrist_yaw_link"
PADDLE_BODY = f"{PREFIX}benchmark_paddle"
PADDLE_GEOM = f"{PREFIX}benchmark_paddle_blade"
PADDLE_SITE = f"{PREFIX}benchmark_paddle_center"
PEDESTAL_GEOM = f"{PREFIX}pedestal"

# The asset's own ``stand`` keyframe, minus the free joint it no longer has.
# Ordered like the model's joint declaration: left leg, right leg, waist, left
# arm, right arm.
STAND_QPOS = (
    (0.0,) * 12
    + (0.0, 0.0, 0.0)
    + (0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0)
    + (0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0)
)

# Joint speed envelope.  Unlike the Panda's, this is NOT a datasheet figure:
# the G1 MJCF declares per-joint ``actuatorfrcrange`` but no speed limit, and
# Unitree does not publish a per-joint maximum this benchmark could cite.  So
# the benchmark declares one uniform conservative value, enforces it like any
# other limit, and marks it unverified in ``describe()`` and therefore in every
# report.  A guessed number that is labelled a guess is usable; a guessed number
# that reads as a datasheet is not.
G1_VELOCITY_LIMIT_RAD_S = 10.0
G1_VELOCITY_LIMIT_SOURCE = "unverified_placeholder"

# Ready pose for the ten commanded joints, in ``TASK_JOINTS`` order.  Frozen by
# ``scripts/calibrate_reachability.py --robot g1``, by the same Chebyshev-centre
# iteration the Panda's uses.
#
# The number it produces is not the interesting part.  Worst-case joint travel
# from this pose to any reachable strike point is 0.067 s against a shortest
# crossing time of 0.558 s -- an eightfold margin, where the Panda's was 7 ms.
# Travel time is simply not this robot's constraint.  Reach is: from here the
# arm can put the blade on 96 of 120 measured crossings, and only 84 of those
# without folding some part of itself into some other part.
G1_READY_QPOS_REACHABLE = 96
G1_READY_QPOS_COLLISION_FREE = 84
G1_READY_QPOS_SAMPLE = 120
G1_READY_QPOS = (
    0.42068,
    0.17477,
    0.36833,
    -2.18122,
    -0.63251,
    -0.79535,
    1.97035,
    1.52509,
    -0.23185,
    -1.07621,
)

BLADE_HALF_EXTENTS = (0.085, 0.008, 0.10)


SHOULDER_ROLL_JOINT = "right_shoulder_roll_joint"
SHOULDER_ROLL_SOLVER_MAX_RAD = -0.10
"""Upper bound the *solver* keeps the right shoulder in, in radians.

Measured, not guessed.  With the joint's full range available the damped
least-squares solver frequently returns the branch that adducts the shoulder and
folds the arm across the torso: on 120 real strike points from the dev split,
only 40 of the solutions it returned were free of robot-on-robot contact.
Holding the shoulder in abduction raises that to 85 while *increasing* the
number of points reached, because a folded solution is one the servo cannot
track anyway -- the arm jams on the torso and the blade never arrives.

This bound lives in the solver, not in the action space.  The published action
space stays the robot's own joint range, because that is the contract: a policy
may fold this arm into the torso if it wants to, and the safety monitor will
score the resulting self-collision.  What a *baseline* does not have to do is
walk into it, and the adapter's own EFFECTOR_POSE mode is a baseline facility.
"""


def solver_position_bounds(
    model: mujoco.MjModel,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Model joint ranges, with the shoulder held in abduction for the solver."""
    low: list[float] = []
    high: list[float] = []
    for name in JOINT_NAMES:
        joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if joint < 0:  # pragma: no cover - model contract
            raise RuntimeError(f"the robot model is missing joint {name!r}")
        lower, upper = (float(v) for v in model.jnt_range[joint])
        if name == f"{PREFIX}{SHOULDER_ROLL_JOINT}":
            upper = min(upper, SHOULDER_ROLL_SOLVER_MAX_RAD)
        low.append(lower)
        high.append(upper)
    return tuple(low), tuple(high)


@dataclass(frozen=True)
class G1Mount:
    """Where the robot stands, in the task frame.

    ``position`` is where the *soles* go, not the pelvis: the asset already
    places its pelvis 0.793 m above its own origin, so a mount at ``z = 0`` puts
    the feet on the floor at the height the model was authored for.  Raising it
    stands the robot on a plinth, which the pedestal geom draws.
    """

    position: tuple[float, float, float] = (-2.00, -0.05, 0.10)
    pedestal_radius_m: float = 0.16

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.position)
        if len(values) != 3 or not all(np.isfinite(values)):
            raise ValueError("position must be three finite numbers")
        object.__setattr__(self, "position", values)
        radius = float(self.pedestal_radius_m)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("pedestal_radius_m must be greater than zero")
        object.__setattr__(self, "pedestal_radius_m", radius)
        if values[2] < 0.0:
            raise ValueError("the mount must not stand below the floor")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "pedestal_radius_m": self.pedestal_radius_m,
        }


def build_g1_table_tennis_model(
    *,
    mount: G1Mount | None = None,
    source: AssetSource = UNITREE_G1,
    cameras: Sequence[CameraSpec] = (),
    fixed_base: bool = True,
) -> mujoco.MjModel:
    """Compile the table-tennis scene with a fixed-base G1 and its paddle.

    Raises :class:`~multisport_sim.benchmark.assets.AssetUnavailableError` when
    the Menagerie checkout is missing, with the commands that fix it.
    """
    placement = mount or G1Mount()
    model_path = require_asset(source)

    spec = mujoco.MjSpec.from_string(build_xml(Sport.TABLE_TENNIS.value))
    robot = mujoco.MjSpec.from_file(str(model_path))
    # One compiled model holds one impratio; the scene's calibrated bounce
    # depends on its value, so the scene wins.  Recorded as a modification.
    robot.option.impratio = spec.option.impratio
    if fixed_base:
        _fix_base(robot)
    else:
        for key in list(robot.keys):
            robot.delete(key)
    _attach_paddle(robot)
    frame = spec.worldbody.add_frame(pos=list(placement.position))
    frame.attach_body(robot.worldbody.first_body(), PREFIX, "")

    _add_pedestal(spec, placement)
    _add_blade_contact(spec)
    _add_cameras(spec, cameras)
    return spec.compile()


def _fix_base(robot: mujoco.MjSpec) -> None:
    """Delete the pelvis free joint, and the keyframe that sized itself to it.

    Without this the humanoid is scored on balance rather than on the shot.  The
    keyframe has to go with it: its ``qpos`` counts the free joint's seven
    numbers, and MuJoCo rejects the attachment rather than silently truncating.
    Its pose is not lost -- it is copied into :data:`STAND_QPOS`.
    """
    pelvis = robot.worldbody.first_body()
    free = next(
        (joint for joint in pelvis.joints if joint.type == mujoco.mjtJoint.mjJNT_FREE),
        None,
    )
    if free is None:  # pragma: no cover - upstream asset contract
        raise RuntimeError("the G1 asset no longer roots its pelvis on a free joint")
    robot.delete(free)
    for key in list(robot.keys):
        robot.delete(key)


def _attach_paddle(robot: mujoco.MjSpec) -> None:
    """Bolt the benchmark blade to the right wrist, before attachment.

    The blade is an ellipsoid thin along its body y axis, so that axis is the
    strike normal; the site is rotated -90 degrees about x so the *site's* z
    axis is the blade normal.  Aiming is then a single-axis constraint, exactly
    as on the Panda -- which is what lets one scripted swing drive both robots.
    """
    wrist = next((body for body in robot.bodies if body.name == WRIST_BODY), None)
    if wrist is None:  # pragma: no cover - upstream asset contract
        raise RuntimeError(f"the G1 asset no longer exposes a {WRIST_BODY!r} body")

    # Past the rubber hand, along the forearm's local +x, where a held paddle
    # would sit rather than where the hand itself is.
    blade = wrist.add_body(name="benchmark_paddle", pos=[0.145, -0.003, 0.0])
    geom = blade.add_geom()
    geom.name = "benchmark_paddle_blade"
    geom.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    geom.size = list(BLADE_HALF_EXTENTS)
    geom.rgba = [0.82, 0.05, 0.035, 1.0]
    geom.friction = [0.85, 0.01, 0.001]
    geom.mass = 0.17

    site = blade.add_site()
    site.name = "benchmark_paddle_center"
    site.size = [0.005, 0.005, 0.005]
    site.quat = [0.7071067811865476, -0.7071067811865476, 0.0, 0.0]


# Identical to the Panda's blade-rubber pair, and deliberately so: two robots
# whose blades bounced the ball differently would not be playing the same task.
BLADE_CONTACT_SOLREF = (0.023, 0.042)
BLADE_CONTACT_SOLIMP = (0.96, 0.99, 0.001, 0.5, 2.0)
BLADE_CONTACT_FRICTION = (0.85, 0.85, 0.01, 0.001, 0.001)


def _add_blade_contact(spec: mujoco.MjSpec) -> None:
    pair = spec.add_pair()
    pair.name = "robot_paddle_ball_contact"
    pair.geomname1 = PADDLE_GEOM
    pair.geomname2 = f"{Sport.TABLE_TENNIS.value}_ball_geom"
    pair.condim = 6
    pair.friction = list(BLADE_CONTACT_FRICTION)
    pair.solref = list(BLADE_CONTACT_SOLREF)
    pair.solimp = list(BLADE_CONTACT_SOLIMP)


def _add_pedestal(spec: mujoco.MjSpec, mount: G1Mount) -> None:
    """Draw the plinth the robot stands on, when it stands on one."""
    x, y, z = mount.position
    if z <= 0.0:
        return
    geom = spec.worldbody.add_geom()
    geom.name = PEDESTAL_GEOM
    geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    geom.pos = [x, y, z / 2.0]
    geom.size = [mount.pedestal_radius_m, z / 2.0, 0.0]
    geom.rgba = [0.20, 0.21, 0.24, 1.0]


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
"""The robot frame an on-board camera rides; named here so specs need no prefix."""


class G1TableTennisAdapter:
    """A :class:`~multisport_sim.benchmark.robot.RobotAdapter` over a MuJoCo G1.

    The adapter presents ten commanded joints.  The other nineteen are real --
    they have mass, they collide, and the safety monitor watches the arm against
    them -- but they are held at the asset's stand pose and are not part of the
    action space.  ``describe()`` lists them under ``held_joints`` so a report
    never leaves a reader guessing which joints a policy actually drove.
    """

    robot_id = "unitree-g1-tabletennis-v1"

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        mount: G1Mount | None = None,
        workspace: WorkspaceBox | None = None,
        ready_qpos: Sequence[float] = G1_READY_QPOS,
        source: AssetSource = UNITREE_G1,
    ) -> None:
        self.model = model
        self.data = data
        self.mount = mount or G1Mount()
        self.source = source
        self._joint_ids = tuple(
            self._require(mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES
        )
        self._actuator_ids = tuple(
            self._require(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES
        )
        actuator_index = np.asarray(self._actuator_ids, dtype=int)
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
        self._held = self._collect_held_joints()

        ready = np.asarray(ready_qpos, dtype=float)
        if ready.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(ready)):
            raise ValueError("ready_qpos must hold one finite value per commanded joint")
        # Torque limits come from the asset's own ``actuatorfrcrange`` rather
        # than from a number typed here, so they cannot drift from the model a
        # result was produced with.  The asset declares them on the *joints*
        # (25 N.m at the shoulder and elbow, 5 at the wrist, 88 at the waist
        # yaw), which is where MuJoCo keeps them; the actuators' own force range
        # is unset and reads as zero.
        torque = tuple(
            float(model.jnt_actfrcrange[joint][1]) for joint in self._joint_ids
        )
        if any(value <= 0.0 for value in torque):  # pragma: no cover - asset contract
            raise RuntimeError(
                "the G1 asset no longer declares actuatorfrcrange on its joints"
            )
        self._limits = JointLimits(
            joint_names=JOINT_NAMES,
            position_low=tuple(float(model.jnt_range[j][0]) for j in self._joint_ids),
            position_high=tuple(float(model.jnt_range[j][1]) for j in self._joint_ids),
            velocity_limit=(G1_VELOCITY_LIMIT_RAD_S,) * len(JOINT_NAMES),
            torque_limit=torque,
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
        if self._solver is None:
            low, high = solver_position_bounds(self.model)
            self._solver = IKSolver(
                self.model,
                site_name=PADDLE_SITE,
                joint_names=JOINT_NAMES,
                position_low=low,
                position_high=high,
                **IK_SETTINGS,
            )
        return self._solver

    def set_contact_labels(self, labels: dict[int, str]) -> None:
        self._contact_labels = dict(labels)

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Stand the robot up, then place the commanded joints in the ready pose."""
        self._configure_actuators(ControlMode.JOINT_POSITION)
        for qpos_adr, actuator, value in self._held:
            self.data.qpos[qpos_adr] = value
            self.data.ctrl[actuator] = value
        self.data.qpos[self._qpos_adr] = self.ready_qpos
        self.data.qvel[:] = 0.0
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
            # The torque the joint actually received, not the servo's raw
            # output.  The G1 asset declares its limits with ``actuatorfrcrange``
            # on the *joints*, and MuJoCo enforces those in ``qfrc_actuator``
            # while leaving ``actuator_force`` unclamped.  Reading the latter
            # reported wrist torques of 250 N.m against a 5 N.m limit and failed
            # every episode on a safety violation that never physically
            # happened.  For the Panda the two are identical, which is why this
            # went unnoticed until a second robot arrived.
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

        Self-contact *is* reported, and that is a deliberate choice for this
        embodiment.  Measured at the frozen ready pose the robot touches nothing
        -- not itself, not the floor -- so any robot-on-robot contact during an
        episode is the arm folding into the torso, which is exactly the failure
        the solver bound above exists to avoid and exactly the thing a policy
        that ignores it should be scored for.  If the pose ever changes so that
        the arms rest on the torso at rest, this becomes a permanent violation
        and the test that pins the ready pose will say so.

        The floor is excluded, because this robot stands on it.  A Panda bolted
        to a pedestal touching the floor would be a collision; a humanoid with
        its feet on the ground is the intended configuration, and scoring it as
        a breach would fail every episode on step one.
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
            if label is not None and label not in {"ball", "floor"}:
                labels.add(label)
        return tuple(sorted(labels))

    def apply(self, command: RobotCommand) -> None:
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
            "G1TableTennisAdapter expects a JointCommand or an EffectorPoseCommand; "
            f"received {type(command).__name__}"
        )

    def _configure_actuators(self, mode: ControlMode) -> None:
        """Select servo or direct-torque semantics for the commanded actuators.

        Only the ten commanded actuators are touched.  The held joints stay
        servos whatever the policy's control mode is, because "hold the legs
        still" is the benchmark's job and not something a torque-mode policy
        should be able to switch off.
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
        """Turn a velocity request into the next position setpoint."""
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
            "held_joints": [name for _, _, name in self._held_names()],
            "base": "fixed pelvis; locomotion and balance are out of scope",
            "control_modes": sorted(mode.value for mode in self.control_modes),
            "default_control_mode": ControlMode.JOINT_POSITION.value,
            "ready_qpos": list(self.ready_qpos),
            "mount": self.mount.to_dict(),
            "effector_pose_solver": "damped-least-squares-ik",
            "velocity_limit_rad_s": G1_VELOCITY_LIMIT_RAD_S,
            "velocity_limit_source": G1_VELOCITY_LIMIT_SOURCE,
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
        found: list[int] = []
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body is not None and body.startswith(PREFIX):
                found.append(geom_id)
        if not found:  # pragma: no cover - protects against a prefix change
            raise RuntimeError("no robot geoms were found in the compiled model")
        return found

    def _held_names(self) -> list[tuple[int, int, str]]:
        """Every prefixed actuator that is not in the action space."""
        commanded = set(ACTUATOR_NAMES)
        out: list[tuple[int, int, str]] = []
        for actuator in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            if name is None or not name.startswith(PREFIX) or name in commanded:
                continue
            joint = int(self.model.actuator_trnid[actuator][0])
            out.append((int(self.model.jnt_qposadr[joint]), actuator, name))
        return out

    def _collect_held_joints(self) -> tuple[tuple[int, int, float], ...]:
        """Held joints paired with the stand-pose angle each is kept at."""
        stand = dict(zip(_model_joint_order(self.model), STAND_QPOS, strict=False))
        held: list[tuple[int, int, float]] = []
        for qpos_adr, actuator, name in self._held_names():
            held.append((qpos_adr, actuator, float(stand.get(name, 0.0))))
        return tuple(held)


def _model_joint_order(model: mujoco.MjModel) -> list[str]:
    """Prefixed robot joint names in the model's own declaration order.

    :data:`STAND_QPOS` is the upstream keyframe, so it is indexed by this order
    and not by anything this module chose.
    """
    names: list[str] = []
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if (name is not None and name.startswith(PREFIX)
                and model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE):
            names.append(name)
    return names


__all__ = [
    "ACTUATOR_NAMES",
    "BLADE_HALF_EXTENTS",
    "G1_READY_QPOS",
    "G1_VELOCITY_LIMIT_RAD_S",
    "IK_SETTINGS",
    "JOINT_NAMES",
    "PADDLE_GEOM",
    "PADDLE_SITE",
    "STAND_QPOS",
    "TASK_JOINTS",
    "G1Mount",
    "G1TableTennisAdapter",
    "build_g1_table_tennis_model",
]
