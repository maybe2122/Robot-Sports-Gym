"""The vision track: segmentation, triangulation, and the rule the track exists for.

Most of this needs no simulator -- the perception maths is checked against
images and rays constructed by hand, where the right answer is known exactly.
The two tests at the end need the Panda asset and check the only claims that
matter in the end: the pipeline localizes a real ball to centimetres, and a
vision-track controller is physically unable to read privileged state.
"""

from __future__ import annotations

import numpy as np
import pytest

from multisport_sim.benchmark.assets import FRANKA_PANDA, asset_available
from multisport_sim.benchmark.sensors import CameraFrame, CameraSpec, SensorReadings
from multisport_sim.benchmark.vision import (
    LEFT_CAMERA,
    RIGHT_CAMERA,
    TABLE_TENNIS_VISION_SENSORS,
    BallDetection,
    StereoBallTracker,
    VisionObservation,
    VisionTrackBackend,
    ball_mask,
    describe_track,
    largest_blob,
    pixel_ray,
    triangulate,
)

BALL_RGB = (255, 115, 15)
BLADE_RGB = (209, 13, 9)
TABLE_RGB = (6, 69, 133)
FLOOR_RGB = (87, 41, 20)


def _image(background: tuple[int, int, int], patches=()) -> np.ndarray:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :] = background
    for colour, (row, column, size) in patches:
        image[row : row + size, column : column + size] = colour
    return image


def test_the_mask_finds_the_ball_and_nothing_else_in_the_scene() -> None:
    """The blade is the trap: it is red and bright, and only green-blue rejects it."""
    image = _image(
        TABLE_RGB,
        [
            (BALL_RGB, (10, 12, 3)),
            (BLADE_RGB, (25, 40, 8)),
            (FLOOR_RGB, (0, 0, 6)),
        ],
    )

    mask = ball_mask(image)

    assert mask.sum() == 9  # only the 3x3 ball patch
    rows, columns = np.nonzero(mask)
    assert rows.min() == 10 and columns.min() == 12


@pytest.mark.parametrize("colour", [BLADE_RGB, TABLE_RGB, FLOOR_RGB, (0, 0, 0)])
def test_nothing_else_in_the_palette_reads_as_a_ball(colour: tuple[int, int, int]) -> None:
    assert not ball_mask(_image(colour)).any()


def test_the_mask_rejects_a_malformed_image() -> None:
    with pytest.raises(ValueError, match="height, width, 3"):
        ball_mask(np.zeros((4, 4), dtype=np.uint8))


def test_the_largest_blob_wins_and_an_implausible_one_is_discarded() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1, 1] = True  # a one-pixel speck
    mask[5:8, 5:8] = True  # the ball

    blob = largest_blob(mask)

    assert blob is not None
    assert blob[2] == 9
    assert blob[0] == pytest.approx(6.0)
    assert blob[1] == pytest.approx(6.0)
    assert largest_blob(np.zeros((4, 4), dtype=bool)) is None
    # A segmentation covering the frame has found something other than a ball.
    assert largest_blob(np.ones((80, 80), dtype=bool)) is None


def test_a_centre_pixel_ray_points_where_the_camera_points() -> None:
    origin, direction = pixel_ray(
        LEFT_CAMERA, (LEFT_CAMERA.width - 1) / 2.0, (LEFT_CAMERA.height - 1) / 2.0
    )

    assert origin == pytest.approx(np.asarray(LEFT_CAMERA.position))
    aim = np.asarray((-0.40, 0.0, 1.00)) - np.asarray(LEFT_CAMERA.position)
    assert direction == pytest.approx(aim / np.linalg.norm(aim), abs=1e-9)


def test_two_rays_recover_the_point_they_were_cast_from() -> None:
    """Project a known point into both cameras, then triangulate it back."""
    truth = np.asarray((-0.6, 0.15, 1.05))
    rays = []
    for spec in (LEFT_CAMERA, RIGHT_CAMERA):
        direction = truth - np.asarray(spec.position)
        rays.append((np.asarray(spec.position), direction / np.linalg.norm(direction)))

    point, residual = triangulate(rays[0][0], rays[0][1], rays[1][0], rays[1][1])

    assert point == pytest.approx(truth, abs=1e-9)
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_rays_that_do_not_meet_report_a_residual() -> None:
    _, residual = triangulate(
        np.asarray((0.0, -1.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((1.0, 0.0, 0.5)),
        np.asarray((0.0, 0.0, 1.0)),
    )

    assert residual == pytest.approx(1.0, abs=1e-9)


def _stereo_readings(time_s: float, position: np.ndarray) -> SensorReadings:
    """Render two synthetic frames with the ball's true projection lit up."""
    frames = []
    for spec in (LEFT_CAMERA, RIGHT_CAMERA):
        image = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
        image[:, :] = TABLE_RGB
        focal = (spec.height / 2.0) / np.tan(np.radians(spec.fovy_deg) / 2.0)
        from multisport_sim.benchmark.sensors import quaternion_matrix

        local = quaternion_matrix(spec.quaternion).T @ (position - np.asarray(spec.position))
        u = (spec.width - 1) / 2.0 + focal * local[0] / -local[2]
        v = (spec.height - 1) / 2.0 - focal * local[1] / -local[2]
        row, column = round(v), round(u)
        image[row : row + 1, column : column + 1] = BALL_RGB
        frames.append(CameraFrame(name=spec.name, time_s=time_s, rgb=image))
    return SensorReadings(frames)


def test_the_tracker_estimates_velocity_from_a_short_history() -> None:
    tracker = StereoBallTracker()
    start = np.asarray((0.6, 0.05, 1.10))
    velocity = np.asarray((-5.0, 0.4, -0.3))

    for step in range(5):
        time_s = 0.01 * step
        tracker.detect(_stereo_readings(time_s, start + velocity * time_s))

    estimate = tracker.estimate()

    assert estimate is not None
    position, measured = estimate
    assert np.asarray(position) == pytest.approx(start + velocity * 0.04, abs=0.02)
    assert np.asarray(measured) == pytest.approx(velocity, rel=0.05, abs=0.2)


def test_the_tracker_ignores_a_frame_it_has_already_seen() -> None:
    """At 120 Hz sensors on a 200 Hz loop the same image arrives twice."""
    tracker = StereoBallTracker()
    readings = _stereo_readings(0.0, np.asarray((0.6, 0.0, 1.1)))

    assert isinstance(tracker.detect(readings), BallDetection)
    assert tracker.detect(readings) is None
    assert len(tracker.detections) == 1
    assert tracker.estimate() is None  # one sample is not a velocity

    tracker.reset()
    assert tracker.detections == ()


def test_an_undetectable_frame_yields_no_detection() -> None:
    tracker = StereoBallTracker()
    blank = SensorReadings(
        CameraFrame(
            name=spec.name,
            time_s=0.0,
            rgb=np.zeros((spec.height, spec.width, 3), dtype=np.uint8),
        )
        for spec in (LEFT_CAMERA, RIGHT_CAMERA)
    )

    assert tracker.detect(blank) is None


def test_the_vision_observation_has_no_ball() -> None:
    """The track's whole definition is this absence, so it is asserted."""
    assert "ball" not in VisionObservation.__dataclass_fields__
    assert set(VisionObservation.__dataclass_fields__) == {"time_s", "robot", "sensors"}


def test_the_wrapper_refuses_a_backend_without_sensors() -> None:
    class Sensorless:
        sensors = None

    with pytest.raises(ValueError, match="built with sensors"):
        VisionTrackBackend(Sensorless())


def test_the_declared_suite_is_reported_in_full() -> None:
    described = describe_track()

    assert described["track"] == "vision"
    assert described["privileged_ball_state"] is False
    names = {sensor["name"] for sensor in described["sensors"]}
    assert {"ball_camera_left", "ball_camera_right"} <= names
    assert len(described["sensors"]) == len(TABLE_TENNIS_VISION_SENSORS)


def test_the_stereo_pair_is_wide_enough_to_triangulate() -> None:
    """Depth here comes from geometry; a narrow baseline would make it noise."""
    baseline = np.linalg.norm(
        np.asarray(LEFT_CAMERA.position) - np.asarray(RIGHT_CAMERA.position)
    )

    assert baseline > 3.0
    assert LEFT_CAMERA.rate_hz == RIGHT_CAMERA.rate_hz


@pytest.fixture(scope="module")
def vision_backend():
    from multisport_sim.benchmark.backends.mujoco_robot import (
        MujocoPandaTableTennisBackend,
    )

    instance = MujocoPandaTableTennisBackend(sensors=TABLE_TENNIS_VISION_SENSORS)
    yield instance
    instance.sensors.close()


@pytest.mark.skipif(
    not asset_available(FRANKA_PANDA),
    reason="the Franka Panda asset is not installed; see docs/ROBOT_LAYER.md",
)
class TestAgainstTheRealScene:
    """The claims that only a rendered ball in a real scene can support."""

    def test_the_pipeline_localizes_a_real_ball_to_centimetres(self, vision_backend) -> None:
        from multisport_sim.benchmark.shot_bank import ShotBank

        shot = next(s for s in ShotBank.from_resource(split="dev") if s.level == "L2")
        vision = VisionTrackBackend(vision_backend)
        tracker = StereoBallTracker()
        vision_backend.reset()
        tracker.reset()
        vision_backend.launch_ball(shot)

        errors = []
        for _ in range(240):
            vision_backend.step()
            detection = tracker.detect(vision.observe().sensors)
            truth = np.asarray(vision_backend.get_ball_state().position)
            if detection is not None:
                errors.append(float(np.linalg.norm(np.asarray(detection.position) - truth)))
            if truth[0] < -1.7:
                break

        assert len(errors) > 20
        assert float(np.median(errors)) < 0.03

    def test_a_vision_controller_cannot_reach_privileged_state(self, vision_backend) -> None:
        vision = VisionTrackBackend(vision_backend)
        vision_backend.reset()

        observation = vision.observe()

        assert isinstance(observation, VisionObservation)
        assert not hasattr(observation, "ball")
        # Everything else the runner needs is still delegated.
        assert vision.timestep == vision_backend.timestep
        assert vision.safety_violations() == ()

    def test_the_gymnasium_vision_environment_passes_the_checker(self) -> None:
        import gymnasium as gym
        from gymnasium.utils.env_checker import check_env

        from multisport_sim.benchmark import register_envs

        register_envs()
        env = gym.make("MultiSportRobot/TableTennisReturn-Panda-Vision-v1").unwrapped
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()

    def test_training_and_scoring_read_the_same_vision_observation(self) -> None:
        """One packing function, so a policy is scored on what it trained on."""
        import gymnasium as gym

        from multisport_sim.benchmark import register_envs

        register_envs()
        env = gym.make("MultiSportRobot/TableTennisReturn-Panda-Vision-v1").unwrapped
        try:
            shot_id = next(shot.shot_id for shot in env.shot_bank if shot.level == "L2")
            observation, info = env.reset(seed=0, options={"shot_id": shot_id})
            assert env.observation_space.contains(observation)
            assert info["track"] == "vision"
            assert info["privileged_ball_state"] is False
            assert set(observation) == {
                "ball_camera_left",
                "ball_camera_right",
                "joint_positions",
                "joint_velocities",
                "blade_pose",
                "frame_age_s",
            }
            assert observation["ball_camera_left"].shape == LEFT_CAMERA.shape
            assert float(observation["frame_age_s"][0]) == pytest.approx(0.0)

            # One control step at 200 Hz on a 120 Hz camera leaves the frame stale.
            stepped, _, _, _, _ = env.step(np.asarray(env.action_space.sample()))
            assert float(stepped["frame_age_s"][0]) > 0.0
        finally:
            env.close()

    def test_the_declared_cameras_exist_in_the_compiled_model(self, vision_backend) -> None:
        import mujoco

        for spec in TABLE_TENNIS_VISION_SENSORS:
            if not isinstance(spec, CameraSpec):
                continue
            index = mujoco.mj_name2id(vision_backend.model, mujoco.mjtObj.mjOBJ_CAMERA, spec.name)
            assert index >= 0
            assert float(vision_backend.model.cam_fovy[index]) == pytest.approx(spec.fovy_deg)
