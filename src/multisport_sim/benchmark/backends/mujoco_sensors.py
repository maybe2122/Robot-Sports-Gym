"""MuJoCo reference implementation of the backend-neutral sensor suite.

Everything simulator-specific about sensing lives here: renderer lifetimes,
``mjData`` field names, contact-force extraction and frame lookups.  A task, a
perception pipeline or a policy sees only the types in
:mod:`multisport_sim.benchmark.sensors`, so the same code runs against an Isaac
suite that produces the same readings from its own renderer.

Two choices are worth stating because they change what a measurement means:

* Rate is enforced by :class:`~multisport_sim.benchmark.sensors.SensorSchedule`,
  not by the caller.  A camera declared at 60 Hz is rendered at 60 Hz however
  often ``sample`` is called, and the frame returned in between is the previous
  one, carrying the earlier ``time_s``.  A policy that ignores that timestamp is
  making an error the hardware would have made for it.
* The IMU reports *proper* acceleration, gravity included, in the sensor's own
  frame -- what an accelerometer measures, not the body's kinematic
  acceleration.  This needs ``mj_rnePostConstraint``, which is run before the
  reading is taken.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import mujoco
import numpy as np

from ..sensors import (
    WORLD_FRAME,
    CameraFrame,
    CameraSpec,
    ContactReading,
    ContactSensorSpec,
    FrameTransformReading,
    FrameTransformSpec,
    ImuReading,
    ImuSpec,
    JointTorqueReading,
    JointTorqueSpec,
    SensorReading,
    SensorReadings,
    SensorSchedule,
    SensorSpec,
    SensorUnavailable,
    describe_specs,
)

_FRAME_OBJECTS = (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_SITE)


class MujocoSensorSuite:
    """A :class:`~multisport_sim.benchmark.sensors.SensorSuite` over ``mjData``.

    ``camera_names`` maps a spec name to the MuJoCo camera that realizes it,
    for models whose attachment prefixes the name.  Without an entry the spec
    name is used, which is the common case.

    ``semantic_geoms`` is the backend's geom-id to category map -- the same one
    the judge scores contacts with -- so a contact sensor reports ``"ball"``
    rather than a geom id no policy should ever see.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        specs: Sequence[SensorSpec],
        *,
        camera_names: Mapping[str, str] | None = None,
        semantic_geoms: Mapping[int, str] | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self._specs = tuple(specs)
        self._schedule = SensorSchedule(self._specs)
        self._camera_names = dict(camera_names or {})
        self._semantic_geoms = dict(semantic_geoms or {})
        self._renderers: dict[tuple[int, int, bool], mujoco.Renderer] = {}
        self._latest: dict[str, SensorReading] = {}

        # Resolve every name once, at construction: a suite that only fails on
        # the step a sensor is first due would hide a typo until mid-episode.
        self._camera_ids: dict[str, int] = {}
        self._frames: dict[str, tuple[mujoco.mjtObj, int]] = {}
        self._body_geoms: dict[str, tuple[int, ...]] = {}
        self._joint_dofs: dict[str, tuple[int, ...]] = {}
        for spec in self._specs:
            self._resolve(spec)

    # -- construction helpers ----------------------------------------------

    def _resolve(self, spec: SensorSpec) -> None:
        if isinstance(spec, CameraSpec):
            name = self._camera_names.get(spec.name, spec.name)
            camera_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name))
            if camera_id < 0:
                raise SensorUnavailable(spec.name, f"the model has no camera {name!r}")
            self._camera_ids[spec.name] = camera_id
            return
        if isinstance(spec, JointTorqueSpec):
            addresses: list[int] = []
            for joint in spec.joint_names:
                joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint))
                if joint_id < 0:
                    raise SensorUnavailable(spec.name, f"the model has no joint {joint!r}")
                addresses.append(int(self.model.jnt_dofadr[joint_id]))
            self._joint_dofs[spec.name] = tuple(addresses)
            return
        if isinstance(spec, ContactSensorSpec):
            body_id = self._require_body(spec)
            start = int(self.model.body_geomadr[body_id])
            count = int(self.model.body_geomnum[body_id])
            if count <= 0:
                raise SensorUnavailable(spec.name, f"body {spec.mount!r} has no geoms to touch")
            self._body_geoms[spec.name] = tuple(range(start, start + count))
            return
        if isinstance(spec, ImuSpec):
            self._frames[spec.name] = (mujoco.mjtObj.mjOBJ_BODY, self._require_body(spec))
            return
        if isinstance(spec, FrameTransformSpec):
            self._frames[spec.name] = self._require_frame(spec.name, spec.target)
            if spec.mount != WORLD_FRAME:
                self._frames[f"{spec.name}::mount"] = self._require_frame(spec.name, spec.mount)
            return
        raise SensorUnavailable(spec.name, f"unsupported sensor type {type(spec).__name__}")

    def _require_body(self, spec: SensorSpec) -> int:
        body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, spec.mount))
        if body_id < 0:
            raise SensorUnavailable(spec.name, f"the model has no body {spec.mount!r}")
        return body_id

    def _require_frame(self, sensor: str, name: str) -> tuple[mujoco.mjtObj, int]:
        for object_type in _FRAME_OBJECTS:
            index = int(mujoco.mj_name2id(self.model, object_type, name))
            if index >= 0:
                return (object_type, index)
        raise SensorUnavailable(sensor, f"the model has no body or site {name!r}")

    # -- suite contract -----------------------------------------------------

    @property
    def specs(self) -> tuple[SensorSpec, ...]:
        return self._specs

    def reset(self) -> None:
        self._schedule.reset()
        self._latest.clear()

    def sample(self, time_s: float) -> SensorReadings:
        """Read every sensor whose period has elapsed; hold the rest.

        The returned mapping always contains one reading per declared sensor
        once each has fired: a policy should not have to special-case the first
        few steps of an episode, only to read the timestamps.
        """
        due = self._schedule.due(time_s)
        if due:
            if any(isinstance(spec, ImuSpec) for spec in due):
                # cacc is only meaningful after the post-constraint RNE pass.
                mujoco.mj_rnePostConstraint(self.model, self.data)
            for spec in due:
                self._latest[spec.name] = self._read(spec, float(time_s))
            self._schedule.mark(due, time_s)
        return SensorReadings(self._latest.values())

    def describe(self) -> dict[str, Any]:
        return {
            **describe_specs(self._specs),
            "implementation": "mujoco",
            "mujoco_version": mujoco.__version__,
        }

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    # -- per-sensor reads ---------------------------------------------------

    def _read(self, spec: SensorSpec, time_s: float) -> SensorReading:
        if isinstance(spec, CameraSpec):
            return self._read_camera(spec, time_s)
        if isinstance(spec, ImuSpec):
            return self._read_imu(spec, time_s)
        if isinstance(spec, ContactSensorSpec):
            return self._read_contact(spec, time_s)
        if isinstance(spec, JointTorqueSpec):
            return self._read_joint_torque(spec, time_s)
        if isinstance(spec, FrameTransformSpec):
            return self._read_frame_transform(spec, time_s)
        raise SensorUnavailable(spec.name, f"unsupported sensor type {type(spec).__name__}")

    def _renderer(self, height: int, width: int, *, depth: bool) -> mujoco.Renderer:
        """One renderer per resolution; creating them per frame is far slower."""
        key = (height, width, depth)
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            if depth:
                renderer.enable_depth_rendering()
            self._renderers[key] = renderer
        return renderer

    def _read_camera(self, spec: CameraSpec, time_s: float) -> CameraFrame:
        camera_id = self._camera_ids[spec.name]
        colour = self._renderer(spec.height, spec.width, depth=False)
        colour.update_scene(self.data, camera=camera_id)
        rgb = np.array(colour.render(), dtype=np.uint8)
        depth = None
        if spec.depth:
            metric = self._renderer(spec.height, spec.width, depth=True)
            metric.update_scene(self.data, camera=camera_id)
            depth = np.array(metric.render(), dtype=np.float32)
        return CameraFrame(name=spec.name, time_s=time_s, rgb=rgb, depth=depth)

    def _read_imu(self, spec: ImuSpec, time_s: float) -> ImuReading:
        _, body_id = self._frames[spec.name]
        velocity = np.zeros(6)
        acceleration = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 1
        )
        mujoco.mj_objectAcceleration(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, acceleration, 1
        )
        quaternion = np.asarray(self.data.xquat[body_id], dtype=np.float64)
        # This is already *proper* acceleration -- what a real accelerometer
        # reads, zero in free fall and +g at rest.  ``mj_rnePostConstraint``
        # seeds the world body's ``cacc`` with -gravity, exactly so that the
        # local linear part comes out in accelerometer convention; subtracting
        # gravity again here would double-count it.
        return ImuReading(
            name=spec.name,
            time_s=time_s,
            linear_acceleration=tuple(float(v) for v in acceleration[3:]),
            angular_velocity=tuple(float(v) for v in velocity[:3]),
            orientation=tuple(float(v) for v in quaternion),
        )

    def _read_contact(self, spec: ContactSensorSpec, time_s: float) -> ContactReading:
        watched = set(self._body_geoms[spec.name])
        forces: dict[str, float] = {}
        wrench = np.zeros(6)
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            if first in watched:
                other = second
            elif second in watched:
                other = first
            else:
                continue
            category = self._semantic_geoms.get(other)
            if category is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            forces[category] = forces.get(category, 0.0) + abs(float(wrench[0]))
        categories = tuple(sorted(forces))
        return ContactReading(
            name=spec.name,
            time_s=time_s,
            categories=categories,
            forces_n=tuple(forces[name] for name in categories),
        )

    def _read_joint_torque(self, spec: JointTorqueSpec, time_s: float) -> JointTorqueReading:
        addresses = self._joint_dofs[spec.name]
        return JointTorqueReading(
            name=spec.name,
            time_s=time_s,
            joint_names=spec.joint_names,
            torques_nm=tuple(float(self.data.qfrc_actuator[address]) for address in addresses),
        )

    def _frame_pose(self, frame: tuple[mujoco.mjtObj, int]) -> tuple[np.ndarray, np.ndarray]:
        object_type, index = frame
        if object_type == mujoco.mjtObj.mjOBJ_BODY:
            position = np.asarray(self.data.xpos[index], dtype=np.float64)
            rotation = np.asarray(self.data.xmat[index], dtype=np.float64).reshape(3, 3)
        else:
            position = np.asarray(self.data.site_xpos[index], dtype=np.float64)
            rotation = np.asarray(self.data.site_xmat[index], dtype=np.float64).reshape(3, 3)
        return position, rotation

    def _read_frame_transform(
        self, spec: FrameTransformSpec, time_s: float
    ) -> FrameTransformReading:
        target_position, target_rotation = self._frame_pose(self._frames[spec.name])
        if spec.mount == WORLD_FRAME:
            position, rotation = target_position, target_rotation
        else:
            mount_position, mount_rotation = self._frame_pose(self._frames[f"{spec.name}::mount"])
            position = mount_rotation.T @ (target_position - mount_position)
            rotation = mount_rotation.T @ target_rotation
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
        return FrameTransformReading(
            name=spec.name,
            time_s=time_s,
            position=tuple(float(v) for v in position),
            quaternion=tuple(float(v) for v in quaternion),
        )


__all__ = ["MujocoSensorSuite"]
