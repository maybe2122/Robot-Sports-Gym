"""The backend-neutral sensor contracts.

Nothing here needs a simulator or an asset.  What is being tested is the part
of the sensor layer that decides what a measurement *means*: that a spec cannot
declare something incoherent, that rate is enforced in one place so two
backends cannot disagree about how many frames an episode had, and that a
reading of the wrong kind fails loudly instead of being unpacked.
"""

from __future__ import annotations

import numpy as np
import pytest

from multisport_sim.benchmark.sensors import (
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
    SensorKind,
    SensorReadings,
    SensorSchedule,
    SensorSuite,
    SensorUnavailable,
    look_at_quaternion,
    quaternion_matrix,
)


def test_camera_spec_publishes_what_a_submission_must_reproduce() -> None:
    camera = CameraSpec(
        name="left", rate_hz=120.0, width=320, height=240, fovy_deg=70.0, depth=True
    )

    assert camera.kind is SensorKind.CAMERA
    assert camera.shape == (240, 320, 3)
    assert camera.period_s == pytest.approx(1.0 / 120.0)
    published = camera.to_dict()
    for field in ("width", "height", "fovy_deg", "rate_hz", "position", "quaternion"):
        assert field in published


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"height": -4},
        {"fovy_deg": 0.0},
        {"fovy_deg": 180.0},
        {"rate_hz": 0.0},
        {"name": "  "},
        {"quaternion": (0.0, 0.0, 0.0, 0.0)},
    ],
)
def test_an_incoherent_camera_is_refused(kwargs: dict) -> None:
    defaults = {"name": "left", "rate_hz": 60.0}
    with pytest.raises(ValueError):
        CameraSpec(**{**defaults, **kwargs})


def test_an_imu_bolted_to_the_world_is_refused() -> None:
    """A world-fixed IMU would read zero forever and look like a working sensor."""
    with pytest.raises(ValueError, match="moving frame"):
        ImuSpec(name="imu", rate_hz=200.0, mount=WORLD_FRAME)


def test_joint_torque_spec_requires_distinct_named_joints() -> None:
    with pytest.raises(ValueError, match="at least one joint"):
        JointTorqueSpec(name="torque", rate_hz=200.0, mount="link", joint_names=())
    with pytest.raises(ValueError, match="unique"):
        JointTorqueSpec(
            name="torque", rate_hz=200.0, mount="link", joint_names=("a", "a")
        )


def test_schedule_fires_every_sensor_once_and_then_at_its_own_rate() -> None:
    fast = ImuSpec(name="imu", rate_hz=200.0, mount="link")
    slow = CameraSpec(name="cam", rate_hz=50.0)
    schedule = SensorSchedule([fast, slow])

    # The first sample is always due: a policy must not step blind.
    assert {spec.name for spec in schedule.due(0.0)} == {"imu", "cam"}
    schedule.mark(schedule.due(0.0), 0.0)

    assert [spec.name for spec in schedule.due(0.005)] == ["imu"]
    schedule.mark(schedule.due(0.005), 0.005)
    assert [spec.name for spec in schedule.due(0.010)] == ["imu"]
    assert {spec.name for spec in schedule.due(0.020)} == {"imu", "cam"}

    schedule.reset()
    assert {spec.name for spec in schedule.due(0.020)} == {"imu", "cam"}


def test_duplicate_sensor_names_are_refused_by_the_schedule() -> None:
    with pytest.raises(ValueError, match="unique"):
        SensorSchedule([ImuSpec(name="x", rate_hz=1.0, mount="a"), CameraSpec(name="x", rate_hz=1.0)])


def test_readings_are_addressed_by_name_and_typed_on_access() -> None:
    frame = CameraFrame(name="cam", time_s=0.25, rgb=np.zeros((4, 6, 3), np.uint8))
    torque = JointTorqueReading(
        name="torque", time_s=0.25, joint_names=("a", "b"), torques_nm=(1.0, -2.0)
    )
    readings = SensorReadings([frame, torque])

    assert set(readings) == {"cam", "torque"}
    assert readings.camera("cam").rgb.shape == (4, 6, 3)
    assert readings.joint_torque("torque").torques_nm == (1.0, -2.0)

    with pytest.raises(SensorUnavailable, match="expected ImuReading"):
        readings.imu("cam")
    with pytest.raises(SensorUnavailable, match="no reading was produced"):
        readings.camera("missing")


def test_a_camera_frame_must_look_like_a_camera_frame() -> None:
    with pytest.raises(ValueError, match="uint8"):
        CameraFrame(name="cam", time_s=0.0, rgb=np.zeros((2, 2, 3), np.float32))
    with pytest.raises(ValueError, match="height, width, 3"):
        CameraFrame(name="cam", time_s=0.0, rgb=np.zeros((2, 2), np.uint8))
    with pytest.raises(ValueError, match="depth must match"):
        CameraFrame(
            name="cam",
            time_s=0.0,
            rgb=np.zeros((2, 2, 3), np.uint8),
            depth=np.zeros((3, 3), np.float32),
        )


def test_contact_reading_pairs_every_category_with_a_force() -> None:
    reading = ContactReading(
        name="touch", time_s=0.1, categories=("ball", "table"), forces_n=(0.4, 2.0)
    )

    assert reading.touching("ball")
    assert not reading.touching("net")
    with pytest.raises(ValueError, match="forces_n"):
        ContactReading(name="touch", time_s=0.1, categories=("ball",), forces_n=())
    with pytest.raises(ValueError, match="non-negative"):
        ContactReading(name="touch", time_s=0.1, categories=("ball",), forces_n=(-1.0,))


def test_look_at_quaternion_points_the_camera_where_it_says() -> None:
    eye = (-0.7, -2.1, 1.55)
    target = (-0.4, 0.0, 1.0)

    rotation = quaternion_matrix(look_at_quaternion(eye, target))
    # A camera looks along its own -z axis, with +y up.
    forward = -rotation[:, 2]
    expected = np.asarray(target) - np.asarray(eye)
    expected = expected / np.linalg.norm(expected)

    assert forward == pytest.approx(expected, abs=1e-9)
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-9)
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0)
    assert float(rotation[2, 1]) > 0.0  # the image's up axis points upward


def test_look_at_quaternion_refuses_a_degenerate_view() -> None:
    with pytest.raises(ValueError, match="eye and target must differ"):
        look_at_quaternion((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="parallel"):
        look_at_quaternion((0.0, 0.0, 2.0), (0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))


def test_frame_transform_and_imu_readings_normalize_their_quaternions() -> None:
    pose = FrameTransformReading(
        name="pose", time_s=0.0, position=(1.0, 2.0, 3.0), quaternion=(2.0, 0.0, 0.0, 0.0)
    )
    imu = ImuReading(name="imu", time_s=0.0, orientation=(0.0, 3.0, 0.0, 0.0))

    assert pose.quaternion == (1.0, 0.0, 0.0, 0.0)
    assert imu.orientation == (0.0, 1.0, 0.0, 0.0)


def test_a_minimal_object_satisfies_the_suite_protocol() -> None:
    """The protocol is what an Isaac implementation has to match, so it is checked."""

    class Suite:
        specs = ()

        def reset(self) -> None:
            return None

        def sample(self, time_s: float) -> SensorReadings:
            del time_s
            return SensorReadings(())

        def describe(self) -> dict:
            return {}

    assert isinstance(Suite(), SensorSuite)
    assert not isinstance(object(), SensorSuite)


def test_a_contact_sensor_declares_the_frame_it_rides() -> None:
    sensor = ContactSensorSpec(name="blade", rate_hz=200.0, mount="paddle")

    assert sensor.kind is SensorKind.CONTACT
    assert sensor.to_dict()["mount"] == "paddle"


def test_a_frame_transform_needs_a_target() -> None:
    with pytest.raises(ValueError, match="target"):
        FrameTransformSpec(name="pose", rate_hz=100.0)
