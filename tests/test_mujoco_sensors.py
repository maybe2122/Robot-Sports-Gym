"""The MuJoCo sensor suite, against a scene small enough to reason about.

These tests deliberately avoid the benchmark scene and the Panda asset: what is
being checked is that each reading means what the contract says -- an
accelerometer that reads +g at rest and zero in free fall, a contact sensor
that reports semantic categories, a camera that honours its declared rate --
and a four-geom model makes the expected answer something you can work out by
hand.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from multisport_sim.benchmark.backends.mujoco_sensors import MujocoSensorSuite
from multisport_sim.benchmark.sensors import (
    WORLD_FRAME,
    CameraSpec,
    ContactSensorSpec,
    FrameTransformSpec,
    ImuSpec,
    JointTorqueSpec,
    SensorUnavailable,
    look_at_quaternion,
)

SCENE = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.2 0.2 0.2 1"/>
    <body name="faller" pos="0 0 1.5">
      <freejoint name="fall"/>
      <geom name="marker" type="sphere" size="0.25" rgba="1 0 0 1"/>
      <site name="marker_site" pos="0 0 0.25" size="0.01"/>
    </body>
    <body name="arm" pos="1 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="link" type="capsule" fromto="0 0 0 0.4 0 0" size="0.05" rgba="0 0 1 1"/>
    </body>
    <camera name="watcher" pos="0 -2 1.5" mode="targetbody" target="faller"/>
  </worldbody>
  <actuator>
    <motor name="hinge_motor" joint="hinge" gear="1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _semantic_geoms(model: mujoco.MjModel) -> dict[int, str]:
    names = {"floor": "floor", "marker": "ball", "link": "robot_racket"}
    return {
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)): category
        for name, category in names.items()
    }


def test_a_camera_renders_the_declared_resolution(scene) -> None:
    model, data = scene
    camera = CameraSpec(name="watcher", rate_hz=60.0, width=48, height=32, fovy_deg=60.0)
    suite = MujocoSensorSuite(model, data, [camera])

    frame = suite.sample(0.0).camera("watcher")

    assert frame.rgb.shape == (32, 48, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.time_s == 0.0
    # The red sphere is in view, so some pixel is dominated by red.
    red = frame.rgb[..., 0].astype(int) - frame.rgb[..., 2].astype(int)
    assert red.max() > 60
    suite.close()


def test_a_camera_holds_its_frame_between_its_own_ticks(scene) -> None:
    """A 10 Hz camera on a 100 Hz loop returns the old frame, with the old time."""
    model, data = scene
    camera = CameraSpec(name="watcher", rate_hz=10.0, width=16, height=16)
    suite = MujocoSensorSuite(model, data, [camera])

    first = suite.sample(0.0).camera("watcher")
    held = suite.sample(0.01).camera("watcher")
    fresh = suite.sample(0.10).camera("watcher")

    assert held.time_s == first.time_s == 0.0
    assert fresh.time_s == pytest.approx(0.10)
    suite.close()


def test_depth_is_produced_only_when_it_is_declared(scene) -> None:
    model, data = scene
    plain = CameraSpec(name="watcher", rate_hz=60.0, width=16, height=16)
    ranging = CameraSpec(name="watcher", rate_hz=60.0, width=16, height=16, depth=True)

    without = MujocoSensorSuite(model, data, [plain])
    with_depth = MujocoSensorSuite(model, data, [ranging])

    assert without.sample(0.0).camera("watcher").depth is None
    frame = with_depth.sample(0.0).camera("watcher")
    assert frame.depth is not None
    assert frame.depth.shape == (16, 16)
    assert float(frame.depth.min()) > 0.0
    without.close()
    with_depth.close()


def test_the_imu_reports_proper_acceleration_not_kinematic_acceleration(scene) -> None:
    """At rest it reads +g; in free fall it reads zero.  That is what an IMU does."""
    model, data = scene
    imu = ImuSpec(name="imu", rate_hz=1000.0, mount="faller")
    suite = MujocoSensorSuite(model, data, [imu])

    # Free fall: the body is unsupported at t=0.
    mujoco.mj_forward(model, data)
    falling = suite.sample(0.0).imu("imu")
    assert np.linalg.norm(falling.linear_acceleration) == pytest.approx(0.0, abs=1e-6)

    # Let it land, then read again once it is resting on the floor.
    for _ in range(2000):
        mujoco.mj_step(model, data)
    suite.reset()
    resting = suite.sample(float(data.time)).imu("imu")
    assert resting.linear_acceleration[2] == pytest.approx(9.81, abs=0.2)
    suite.close()


def test_the_contact_sensor_reports_semantic_categories_with_forces(scene) -> None:
    model, data = scene
    sensor = ContactSensorSpec(name="touch", rate_hz=1000.0, mount="faller")
    suite = MujocoSensorSuite(model, data, [sensor], semantic_geoms=_semantic_geoms(model))

    assert suite.sample(0.0).contact("touch").categories == ()

    for _ in range(2000):
        mujoco.mj_step(model, data)
    suite.reset()
    reading = suite.sample(float(data.time)).contact("touch")

    assert reading.categories == ("floor",)
    assert reading.forces_n[0] > 0.0
    suite.close()


def test_joint_torque_reads_what_the_actuator_delivered(scene) -> None:
    model, data = scene
    sensor = JointTorqueSpec(
        name="torque", rate_hz=1000.0, mount="arm", joint_names=("hinge",)
    )
    suite = MujocoSensorSuite(model, data, [sensor])

    data.ctrl[0] = 3.0
    mujoco.mj_step(model, data)
    reading = suite.sample(float(data.time)).joint_torque("torque")

    assert reading.joint_names == ("hinge",)
    assert reading.torques_nm[0] == pytest.approx(3.0, abs=1e-6)
    suite.close()


def test_a_frame_transform_is_expressed_in_its_mount(scene) -> None:
    model, data = scene
    world = FrameTransformSpec(
        name="site_in_world", rate_hz=1000.0, mount=WORLD_FRAME, target="marker_site"
    )
    relative = FrameTransformSpec(
        name="site_in_arm", rate_hz=1000.0, mount="arm", target="marker_site"
    )
    suite = MujocoSensorSuite(model, data, [world, relative])

    readings = suite.sample(0.0)
    absolute = readings.frame_transform("site_in_world").position
    in_arm = readings.frame_transform("site_in_arm").position

    assert absolute == pytest.approx((0.0, 0.0, 1.75), abs=1e-6)
    # The arm body sits at (1, 0, 0.5) with no rotation.
    assert in_arm == pytest.approx((-1.0, 0.0, 1.25), abs=1e-6)
    suite.close()


def test_every_name_is_resolved_at_construction(scene) -> None:
    """A typo must fail when the suite is built, not mid-episode."""
    model, data = scene

    with pytest.raises(SensorUnavailable, match="no camera"):
        MujocoSensorSuite(model, data, [CameraSpec(name="nope", rate_hz=10.0)])
    with pytest.raises(SensorUnavailable, match="no body"):
        MujocoSensorSuite(model, data, [ImuSpec(name="imu", rate_hz=10.0, mount="nope")])
    with pytest.raises(SensorUnavailable, match="no joint"):
        MujocoSensorSuite(
            model,
            data,
            [JointTorqueSpec(name="t", rate_hz=10.0, mount="arm", joint_names=("nope",))],
        )
    with pytest.raises(SensorUnavailable, match="no body or site"):
        MujocoSensorSuite(
            model,
            data,
            [FrameTransformSpec(name="p", rate_hz=10.0, target="nope")],
        )


def test_a_camera_name_may_be_remapped_for_a_prefixed_model(scene) -> None:
    model, data = scene
    spec = CameraSpec(name="court", rate_hz=10.0, width=8, height=8)
    suite = MujocoSensorSuite(model, data, [spec], camera_names={"court": "watcher"})

    assert suite.sample(0.0).camera("court").rgb.shape == (8, 8, 3)
    suite.close()


def test_describe_names_the_suite_and_its_implementation(scene) -> None:
    model, data = scene
    suite = MujocoSensorSuite(
        model,
        data,
        [CameraSpec(name="watcher", rate_hz=30.0, width=8, height=8)],
    )

    described = suite.describe()

    assert described["implementation"] == "mujoco"
    assert described["count"] == 1
    assert described["sensors"][0]["kind"] == "camera"
    suite.close()


def test_camera_pose_from_a_spec_matches_where_it_was_asked_to_look() -> None:
    """The spec, not the scene file, decides where a declared camera points."""
    eye = (0.0, -2.0, 1.0)
    target = (0.0, 0.0, 0.5)
    spec = CameraSpec(
        name="declared",
        rate_hz=30.0,
        width=8,
        height=8,
        position=eye,
        quaternion=look_at_quaternion(eye, target),
    )
    body = mujoco.MjSpec.from_string(SCENE)
    handle = body.worldbody.add_camera()
    handle.name = spec.name
    handle.pos = list(spec.position)
    handle.quat = list(spec.quaternion)
    handle.fovy = spec.fovy_deg
    model = body.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, spec.name))
    rotation = np.asarray(data.cam_xmat[camera_id]).reshape(3, 3)
    forward = -rotation[:, 2]
    expected = np.asarray(target) - np.asarray(eye)

    assert forward == pytest.approx(expected / np.linalg.norm(expected), abs=1e-6)
