"""The depth-only path cannot depend on RGB colour or segmentation IDs."""

from dataclasses import replace

import numpy as np
import pytest

from multisport_sim.benchmark.assets import UNITREE_G1, asset_available
from multisport_sim.benchmark.depth_vision import DepthBallTracker
from multisport_sim.benchmark.sensors import CameraFrame, SensorReadings
from multisport_sim.benchmark.vision import LEFT_CAMERA


def test_sphere_detection_is_independent_of_rgb():
    camera = replace(LEFT_CAMERA, depth=True)
    background = np.full((240, 320), 5.0, dtype=np.float32)
    depth = background.copy()
    depth[119:122, 159:162] = 2.0
    positions = []
    for colour in (0, 255):
        tracker = DepthBallTracker(camera, background)
        frame = CameraFrame(
            name=camera.name,
            time_s=0.0,
            depth=depth,
            rgb=np.full((240, 320, 3), colour, dtype=np.uint8),
        )
        found = tracker.detect(SensorReadings([frame]))
        assert found is not None
        assert found.pixels == (9, 0)
        positions.append(found.position)
        assert tracker.detect(SensorReadings([frame])) is None
        tracker.reset()
        assert tracker.estimate() is None
    assert positions[0] == positions[1]


def test_large_foreground_and_invalid_depth_do_not_become_a_ball():
    camera = replace(LEFT_CAMERA, depth=True)
    background = np.full((240, 320), 5.0, dtype=np.float32)
    depth = background.copy()
    depth[100:140, 140:180] = 2.0
    depth[5:8, 5:8] = np.nan
    frame = CameraFrame(
        name=camera.name, time_s=0.0, depth=depth, rgb=np.zeros((240, 320, 3), dtype=np.uint8)
    )
    assert DepthBallTracker(camera, background).detect(SensorReadings([frame])) is None


@pytest.mark.skipif(not asset_available(UNITREE_G1), reason="G1 asset not installed")
def test_gym_depth_mode_has_no_rgb_and_policy_wrapper_blacks_it_out():
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env

    env = gym.make("MultiSportRobot/TableTennisReturn-G1-Standing-Vision-v2", perception="depth")
    try:
        check_env(env.unwrapped, skip_render_check=True)
        obs, _ = env.reset(seed=0)
        assert "ball_camera_left" not in obs
        assert obs["ball_camera_left_depth"].shape == (240, 320)
        assert obs["base_imu"].shape == (10,)
        policy_obs = env.unwrapped._vision.observe()
        assert not policy_obs.sensors.camera("ball_camera_left").rgb.any()
    finally:
        env.close()
