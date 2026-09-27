"""Depth geometry and the G1's real camera-to-control path."""

from dataclasses import replace

import numpy as np
import pytest

from multisport_sim.benchmark.assets import UNITREE_G1, asset_available
from multisport_sim.benchmark.sensors import CameraFrame, CameraSpec, SensorReadings
from multisport_sim.benchmark.vision import RGBDBallTracker, vision_sensors_for


def test_depth_is_optical_axis_distance_and_duplicate_frames_are_skipped():
    camera = CameraSpec(name="depth", rate_hz=120.0, width=21, height=21, depth=True)
    rgb = np.zeros((21, 21, 3), dtype=np.uint8)
    rgb[10, 15] = (255, 115, 15)
    frame = CameraFrame(
        name="depth", time_s=0.0, rgb=rgb, depth=np.full((21, 21), 2.0, dtype=np.float32)
    )
    tracker = RGBDBallTracker(camera)
    readings = SensorReadings([frame])
    detection = tracker.detect(readings)
    assert detection is not None
    assert detection.position[0] > 0.5
    assert detection.position[2] < -2.0
    assert np.linalg.norm(
        np.array(detection.position) - np.array([2 * 5 / (10.5 / np.tan(np.pi / 6)), 0.0, -2.0])
    ) == pytest.approx(0.02)
    assert tracker.detect(readings) is None
    assert tracker.estimate() is None
    tracker.detect(SensorReadings([replace(frame, time_s=0.01)]))
    assert tracker.estimate() is not None
    tracker.reset()
    assert tracker.estimate() is None


def test_invalid_depth_is_not_a_ball_detection():
    camera = CameraSpec(name="depth", rate_hz=120.0, width=3, height=3, depth=True)
    rgb = np.full((3, 3, 3), (255, 115, 15), dtype=np.uint8)
    for value in (0.0, -1.0, np.nan, np.inf):
        tracker = RGBDBallTracker(camera)
        frame = CameraFrame(name="depth", time_s=0.0, rgb=rgb, depth=np.full((3, 3), value))
        assert tracker.detect(SensorReadings([frame])) is None


@pytest.mark.skipif(not asset_available(UNITREE_G1), reason="G1 asset not installed")
def test_g1_sensor_mounts_resolve_and_render_depth():
    from multisport_sim.benchmark.backends.mujoco_robot import MujocoG1TableTennisBackend
    from multisport_sim.benchmark.vision import VisionTrackBackend

    backend = MujocoG1TableTennisBackend(sensors=vision_sensors_for("g1", depth=True))
    try:
        observation = VisionTrackBackend(backend).observe()
        assert not hasattr(observation, "ball")
        assert len(observation.robot.joint_positions) == 10
        frame = observation.sensors.camera("ball_camera_left")
        assert frame.depth.shape == (240, 320)
        assert np.isfinite(frame.depth).all()
    finally:
        backend.sensors.close()


@pytest.mark.skipif(not asset_available(UNITREE_G1), reason="G1 asset not installed")
@pytest.mark.parametrize("perception", ["stereo", "rgbd"])
def test_g1_vision_gym_contract(perception):
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env

    env = gym.make("MultiSportRobot/TableTennisReturn-G1-Vision-v1", perception=perception)
    try:
        check_env(env.unwrapped, skip_render_check=True)
        observation, info = env.reset(seed=7)
        assert "ball" not in observation
        assert info["privileged_ball_state"] is False
        assert env.action_space.shape == (10,)
        assert ("ball_camera_left_depth" in observation) == (perception == "rgbd")
    finally:
        env.close()
