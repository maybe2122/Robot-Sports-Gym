"""MuJoCo backend that runs a real robot arm instead of a mocap fixture.

This backend is the embodied counterpart of
:class:`~multisport_sim.benchmark.backends.mujoco.MujocoShotBackend`.  Both feed
the same judge the same semantic contacts, so the two agree on what a legal
return is; they differ only in what produces the strike.  Here the blade is the
last link of an actuated arm, which means a hit costs the policy a reachable
configuration, a feasible trajectory and real actuator torque.

Safety is enforced here rather than in the task, because only the backend can
see joint states and contacts on the same physics step.  A violation is a
terminal outcome with ``failure_reason="safety"``: it counts in the denominator
like any miss.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import mujoco
import numpy as np

from ...physics import Aerodynamics, Atmosphere
from ...specs import Sport
from ..robot import (
    ControlMode,
    JointCommand,
    RobotCommand,
    RobotObservation,
    SafetyLimits,
    SafetyMonitor,
    SafetyViolation,
    WorkspaceBox,
)
from ..robots import g1 as g1_module
from ..robots.panda import (
    PADDLE_GEOM,
    PandaMount,
    PandaTableTennisAdapter,
    build_panda_table_tennis_model,
)
from ..sensors import CameraSpec, SensorReadings, SensorSpec, SensorSuite
from ..task_config import TaskFrame
from ..types import BallState, SemanticContact, ShotSpec, Vec3
from .base import ShotBackend
from .mujoco import PROFILES


@runtime_checkable
class SafetyAwareBackend(ShotBackend, Protocol):
    """A backend that can report envelope breaches for the current step.

    The runner checks for this protocol instead of assuming every backend has
    a robot: the mocap fixture legitimately has no safety envelope, and asking
    it for one should not be an error.
    """

    def safety_violations(self) -> tuple[SafetyViolation, ...]:
        ...

    def robot_observation(self) -> RobotObservation:
        ...


@dataclass(frozen=True)
class RobotShotObservation:
    """What a controller sees each control step on an embodied task.

    Ball state is privileged truth used by the state track.  The vision track
    reads :attr:`sensors` instead; the two halves are kept separate rather than
    flattened so that a wrapper can withhold one of them and a report can say
    which one a result was produced with.
    """

    time_s: float
    ball: BallState
    robot: RobotObservation
    sensors: SensorReadings | None = None


class MujocoEmbodiedTableTennisBackend:
    """The table-tennis return task driven by an actuated robot.

    The scene, ball physics, aerodynamics and semantic contact mapping are
    unchanged from the mocap backend; the paddle geom now belongs to the robot.

    Nothing here names an embodiment.  A subclass supplies four class
    attributes -- how to build the model, how to build the adapter, which mount
    type it takes and which geom is its blade -- and gets the whole backend.
    That the second robot needed no change to this file is the evidence that the
    task is robot-agnostic; a copy of this class per robot would have looked the
    same on day one and diverged by the third.
    """

    backend_name = "mujoco-embodied"
    mount_type: type = PandaMount
    paddle_geom = PADDLE_GEOM

    @staticmethod
    def build_model(*, mount: object, cameras: Sequence[CameraSpec]) -> mujoco.MjModel:
        raise NotImplementedError

    @staticmethod
    def build_adapter(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        mount: object,
        workspace: WorkspaceBox | None,
    ) -> object:
        raise NotImplementedError

    def __init__(
        self,
        *,
        mount: object | None = None,
        workspace: WorkspaceBox | None = None,
        wind: Vec3 = (0.0, 0.0, 0.0),
        terminate_on_safety: bool = True,
        sensors: Sequence[SensorSpec] = (),
    ) -> None:
        self.sport = Sport.TABLE_TENNIS
        self.profile = PROFILES[self.sport]
        self.mount = mount if mount is not None else self.mount_type()
        self.sensor_specs = tuple(sensors)
        cameras = tuple(spec for spec in self.sensor_specs if isinstance(spec, CameraSpec))
        self.model = self.build_model(mount=self.mount, cameras=cameras)
        self.data = mujoco.MjData(self.model)
        self.terminate_on_safety = bool(terminate_on_safety)

        wind_vector = tuple(float(value) for value in wind)
        if len(wind_vector) != 3 or not all(np.isfinite(wind_vector)):
            raise ValueError("wind must contain three finite numbers")
        self.aerodynamics = Aerodynamics(self.model, Atmosphere(wind=wind_vector))

        joint_id = self._require(mujoco.mjtObj.mjOBJ_JOINT, self.profile.ball_joint)
        self._ball_qpos_address = int(self.model.jnt_qposadr[joint_id])
        self._ball_dof_address = int(self.model.jnt_dofadr[joint_id])
        self._ball_geom_id = self._require(mujoco.mjtObj.mjOBJ_GEOM, self.profile.ball_geom)

        self.adapter = self.build_adapter(
            self.model, self.data, mount=self.mount, workspace=workspace
        )
        # The robot's blade replaces the mocap fixture as the only geom that can
        # produce a ``robot_racket`` event.
        semantic_names = dict(self.profile.semantic_geom_names())
        semantic_names.pop(self.profile.effector_geom, None)
        semantic_names[self.paddle_geom] = "robot_racket"
        self._semantic_geoms: dict[int, str] = {}
        for name, category in semantic_names.items():
            geom_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name))
            if geom_id < 0:
                # The mocap fixture is absent from the robot scene by design;
                # every other named geom must exist.
                raise RuntimeError(f"the robot model is missing geom {name!r}")
            self._semantic_geoms[geom_id] = category
        self._semantic_geoms[self._ball_geom_id] = "ball"
        self.adapter.set_contact_labels(self._semantic_geoms)

        self.monitor = SafetyMonitor(self.adapter.safety_limits)
        # Built after the semantic map: a contact sensor reports the same
        # categories the judge scores, or the two would disagree about what the
        # blade just touched.
        self.sensors: SensorSuite | None = None
        if self.sensor_specs:
            from .mujoco_sensors import MujocoSensorSuite

            self.sensors = MujocoSensorSuite(
                self.model,
                self.data,
                self.sensor_specs,
                semantic_geoms=self._semantic_geoms,
            )
        self.reset()

    # -- backend contract ---------------------------------------------------

    @property
    def task_frame(self) -> TaskFrame:
        """Identity: the single-sport scene builds the table at the origin."""
        return TaskFrame()

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def safety_limits(self) -> SafetyLimits:
        return self.adapter.safety_limits

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0
        self.adapter.reset()
        self.monitor.reset()
        if self.sensors is not None:
            self.sensors.reset()
        mujoco.mj_forward(self.model, self.data)

    def launch_ball(self, shot: ShotSpec) -> None:
        if shot.sport != self.sport.value:
            raise ValueError(
                f"the table-tennis robot backend cannot launch sport {shot.sport!r}"
            )
        qpos = self._ball_qpos_address
        dof = self._ball_dof_address
        self.data.qpos[qpos : qpos + 3] = shot.position
        self.data.qpos[qpos + 3 : qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[dof : dof + 3] = shot.linear_velocity
        self.data.qvel[dof + 3 : dof + 6] = shot.angular_velocity
        self.data.xfrc_applied[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_ball_state(self) -> BallState:
        qpos = self._ball_qpos_address
        dof = self._ball_dof_address
        return BallState(
            position=tuple(float(v) for v in self.data.qpos[qpos : qpos + 3]),
            linear_velocity=tuple(float(v) for v in self.data.qvel[dof : dof + 3]),
            angular_velocity=tuple(float(v) for v in self.data.qvel[dof + 3 : dof + 6]),
        )

    def semantic_contacts(self) -> tuple[SemanticContact, ...]:
        """Ball contacts normalized to benchmark categories.

        Identical in meaning to the mocap backend's: only the arm's blade maps
        to ``robot_racket``, so brushing a decorative paddle or the pedestal can
        never be scored as a strike.
        """
        contacts: list[SemanticContact] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if self._ball_geom_id == geom1:
                other_geom, normal_sign = geom2, 1.0
            elif self._ball_geom_id == geom2:
                other_geom, normal_sign = geom1, -1.0
            else:
                continue
            category = self._semantic_geoms.get(other_geom)
            if category is None or category == "ball" or float(contact.dist) > 0.0:
                continue
            contacts.append(
                SemanticContact.between(
                    "ball",
                    category,
                    position=tuple(float(v) for v in contact.pos),
                    normal=tuple(normal_sign * float(v) for v in contact.frame[:3]),
                )
            )
        return tuple(contacts)

    def robot_observation(self) -> RobotObservation:
        return self.adapter.observe()

    def read_sensors(self) -> SensorReadings:
        """Sample the declared suite, honouring each sensor's own rate.

        Raises when the backend was built without sensors: a vision policy must
        fail loudly rather than quietly fall back to privileged state.
        """
        if self.sensors is None:
            raise RuntimeError(
                "this backend was built without sensors; pass sensors=... to run "
                "the vision track"
            )
        return self.sensors.sample(self.time)

    def observe(self) -> RobotShotObservation:
        """Privileged truth plus, when a suite is attached, its sensor readings.

        Both are handed over on every step.  Which of them a controller is
        allowed to read is the *track's* rule, recorded in the report, not
        something the backend can enforce -- and a wrapper that hides the
        privileged half is how the vision track enforces it.
        """
        return RobotShotObservation(
            time_s=self.time,
            ball=self.get_ball_state(),
            robot=self.adapter.observe(),
            sensors=self.read_sensors() if self.sensors is not None else None,
        )

    def apply_action(self, action: RobotCommand | None) -> None:
        """Forward one command to the adapter; ``None`` holds the last setpoint.

        Holding is the right no-op for a position-controlled arm: zeroing the
        control would command the arm to fold to zero angles at full torque.
        """
        if action is None:
            return
        self.adapter.apply(action)

    def step(self, action: RobotCommand | None = None) -> None:
        """Apply passive aerodynamics to the ball and advance one physics tick."""
        if action is not None:
            self.apply_action(action)
        self.data.xfrc_applied[:] = 0.0
        self.aerodynamics.apply(self.data)
        mujoco.mj_step(self.model, self.data)

    # -- safety -------------------------------------------------------------

    def safety_violations(self) -> tuple[SafetyViolation, ...]:
        """Envelope breaches at the current physics step, accumulated in the monitor."""
        return self.monitor.check(self.adapter.observe())

    @property
    def safety_violation_count(self) -> int:
        return self.monitor.count

    def hold_command(self) -> JointCommand:
        """A command that keeps the arm exactly where it is.

        Baselines and tests need a well-defined "do nothing" that is not "fall
        over"; the no-op controller emits this.
        """
        observation = self.adapter.observe()
        return JointCommand(observation.joint_positions, ControlMode.JOINT_POSITION)

    def describe(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "mujoco_version": mujoco.__version__,
            "physics_dt": self.timestep,
            "sport": self.sport.value,
            "robot": self.adapter.describe(),
            "terminate_on_safety": self.terminate_on_safety,
        }

    def _require(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = int(mujoco.mj_name2id(self.model, object_type, name))
        if object_id < 0:
            raise RuntimeError(f"the robot benchmark model is missing {name!r}")
        return object_id


class MujocoPandaTableTennisBackend(MujocoEmbodiedTableTennisBackend):
    """The table-tennis return task driven by an actuated Franka Panda."""

    backend_name = "mujoco-panda"
    mount_type = PandaMount
    paddle_geom = PADDLE_GEOM

    @staticmethod
    def build_model(*, mount: object, cameras: Sequence[CameraSpec]) -> mujoco.MjModel:
        return build_panda_table_tennis_model(mount=mount, cameras=cameras)

    @staticmethod
    def build_adapter(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        mount: object,
        workspace: WorkspaceBox | None,
    ) -> object:
        return PandaTableTennisAdapter(model, data, mount=mount, workspace=workspace)


class MujocoG1TableTennisBackend(MujocoEmbodiedTableTennisBackend):
    """The same task driven by a fixed-base Unitree G1.

    Every line of behaviour is inherited.  What this class contains is the
    answer to "which robot", and nothing else.
    """

    backend_name = "mujoco-g1"
    mount_type = g1_module.G1Mount
    paddle_geom = g1_module.PADDLE_GEOM

    @staticmethod
    def build_model(*, mount: object, cameras: Sequence[CameraSpec]) -> mujoco.MjModel:
        return g1_module.build_g1_table_tennis_model(mount=mount, cameras=cameras)

    @staticmethod
    def build_adapter(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        mount: object,
        workspace: WorkspaceBox | None,
    ) -> object:
        return g1_module.G1TableTennisAdapter(
            model, data, mount=mount, workspace=workspace
        )


__all__ = [
    "MujocoEmbodiedTableTennisBackend",
    "MujocoG1TableTennisBackend",
    "MujocoPandaTableTennisBackend",
    "RobotShotObservation",
    "SafetyAwareBackend",
]
