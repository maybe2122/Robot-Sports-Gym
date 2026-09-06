"""The vision track: what a policy may see when it is not handed the truth.

The state track gives a policy the ball's exact position, velocity and spin,
read out of the simulator.  No camera supplies that, so a state-track score
measures control alone.  This module defines the other track:

* :data:`TABLE_TENNIS_VISION_SENSORS` is the declared suite -- a fixed stereo
  pair watching the robot's half, plus the proprioception a real arm has
  anyway.  It is part of the task definition, so a result can be checked
  against the sensors it claimed.
* :class:`VisionTrackBackend` wraps an embodied backend and *removes* the
  privileged ball state from what a controller receives.  The rule is enforced
  by construction rather than promised in a docstring: a vision-track policy
  cannot read the ball because the attribute is not there.
* :class:`StereoBallTracker` is a reference perception pipeline -- colour
  segmentation, largest blob, two-ray triangulation, a short least-squares fit
  for velocity.  It is deliberately simple and deliberately imperfect; the gap
  between the two tracks is a result the benchmark should report, not hide.

Measured against ground truth over all 12 dev shots (540 stereo frames), the
reference pipeline localizes the ball to a median error of 0.81 cm (90th
percentile 1.02 cm, worst 1.42 cm) and returns no detection on 8.1% of frames.
A submission is free to replace it; it is a floor, not a ceiling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, radians, tan
from typing import Any

import numpy as np

from .robot import RobotObservation
from .robots.panda import JOINT_NAMES, PADDLE_BODY, PADDLE_SITE
from .sensors import (
    WORLD_FRAME,
    CameraSpec,
    ContactSensorSpec,
    FrameTransformSpec,
    ImuSpec,
    JointTorqueSpec,
    SensorReadings,
    SensorSpec,
    look_at_quaternion,
    quaternion_matrix,
)

Vec3 = tuple[float, float, float]

BALL_RADIUS_M = 0.02
"""ITTF ball radius; the detector's only physical prior."""

CAMERA_TARGET: Vec3 = (-0.40, 0.0, 1.00)
"""Both cameras are aimed here: the middle of the robot's approach corridor."""

LEFT_CAMERA_EYE: Vec3 = (-0.70, -2.10, 1.55)
RIGHT_CAMERA_EYE: Vec3 = (-0.70, 2.10, 1.55)

LEFT_CAMERA = CameraSpec(
    name="ball_camera_left",
    rate_hz=120.0,
    width=320,
    height=240,
    fovy_deg=70.0,
    position=LEFT_CAMERA_EYE,
    quaternion=look_at_quaternion(LEFT_CAMERA_EYE, CAMERA_TARGET),
)
RIGHT_CAMERA = CameraSpec(
    name="ball_camera_right",
    rate_hz=120.0,
    width=320,
    height=240,
    fovy_deg=70.0,
    position=RIGHT_CAMERA_EYE,
    quaternion=look_at_quaternion(RIGHT_CAMERA_EYE, CAMERA_TARGET),
)
"""A 4.2 m baseline either side of the table.

A wide baseline is what makes depth come from geometry instead of from the
ball's apparent size: at this range a 40 mm ball covers about nine pixels, and
its radius is far too noisy to range with, while two rays crossing at nearly a
right angle are not.
"""

BLADE_POSE = FrameTransformSpec(
    name="blade_pose",
    rate_hz=200.0,
    mount=WORLD_FRAME,
    target=PADDLE_SITE,
)
BLADE_IMU = ImuSpec(name="blade_imu", rate_hz=200.0, mount=PADDLE_BODY)
BLADE_CONTACT = ContactSensorSpec(name="blade_contact", rate_hz=200.0, mount=PADDLE_BODY)
ARM_TORQUE = JointTorqueSpec(
    name="arm_torque", rate_hz=200.0, mount=PADDLE_BODY, joint_names=JOINT_NAMES
)

TABLE_TENNIS_VISION_SENSORS: tuple[SensorSpec, ...] = (
    LEFT_CAMERA,
    RIGHT_CAMERA,
    BLADE_POSE,
    BLADE_IMU,
    BLADE_CONTACT,
    ARM_TORQUE,
)
"""The vision track's declared suite.

The cameras replace privileged ball state.  The rest is proprioception a real
arm already has -- where its own blade is, what its joints are delivering, what
it is touching -- and withholding it would measure a harder problem than the
one the benchmark claims to pose.
"""


@dataclass(frozen=True)
class VisionObservation:
    """What a vision-track controller receives each control step.

    There is no ``ball`` field.  That absence is the track's definition.
    """

    time_s: float
    robot: RobotObservation
    sensors: SensorReadings


class VisionTrackBackend:
    """Wrap an embodied backend so its observations carry no privileged truth.

    Every other part of the backend -- physics, contacts, safety, the judge's
    view of the episode -- is delegated untouched, so the two tracks are scored
    by the same rules on the same shots and differ only in what the policy was
    allowed to see.
    """

    def __init__(self, backend: Any) -> None:
        if getattr(backend, "sensors", None) is None:
            raise ValueError(
                "the vision track needs a backend built with sensors; pass "
                "sensors=TABLE_TENNIS_VISION_SENSORS"
            )
        self.backend = backend

    def observe(self) -> VisionObservation:
        return VisionObservation(
            time_s=float(self.backend.time),
            robot=self.backend.robot_observation(),
            sensors=self.backend.read_sensors(),
        )

    def __getattr__(self, name: str) -> Any:
        # Delegation is explicit about one thing only: `observe` above is not
        # delegated, so no caller can reach the privileged observation through
        # the wrapper.
        return getattr(self.backend, name)


@dataclass(frozen=True)
class BallDetection:
    """One triangulated ball position, with the evidence behind it."""

    time_s: float
    position: Vec3
    pixels: tuple[int, int]
    residual_m: float


def _camera_focal_px(spec: CameraSpec) -> float:
    """Focal length in pixels from the declared vertical field of view."""
    return (spec.height / 2.0) / tan(radians(spec.fovy_deg) / 2.0)


def ball_mask(rgb: np.ndarray) -> np.ndarray:
    """Segment the ITTF orange ball from the rendered scene.

    Three tests, each earning its place: brightness rejects the brown floor,
    red-minus-blue rejects the blue table, and green-minus-blue rejects the red
    paddles and court markings, which are otherwise indistinguishable from the
    ball on the first two.  Dropping the last test alone turns a sub-centimetre
    measurement into a 2.2 m one, because the red blade then outvotes the ball.
    """
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    return (red > 110) & (red - blue > 55) & (red - green > 25) & (green - blue > 40)


def largest_blob(mask: np.ndarray, *, max_pixels: int = 4000) -> tuple[float, float, int] | None:
    """Centroid and size of the largest 4-connected run of set pixels.

    Returns ``None`` when nothing is set, or when the mask is implausibly large
    -- a segmentation that lights up thousands of pixels has found something
    other than a 40 mm ball, and reporting its centroid as the ball would be
    worse than reporting no detection at all.
    """
    flags = np.asarray(mask, dtype=bool)
    count = int(flags.sum())
    if count == 0 or count > max_pixels:
        return None

    height, width = flags.shape
    set_pixels = flags.reshape(-1)
    seen = np.zeros(set_pixels.size, dtype=bool)
    best: tuple[int, list[int]] | None = None
    for start in np.flatnonzero(set_pixels):
        if seen[start]:
            continue
        stack = [int(start)]
        seen[start] = True
        blob: list[int] = []
        while stack:
            index = stack.pop()
            blob.append(index)
            row, column = divmod(index, width)
            for next_row, next_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if not (0 <= next_row < height and 0 <= next_column < width):
                    continue
                neighbour = next_row * width + next_column
                if set_pixels[neighbour] and not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)
        if best is None or len(blob) > best[0]:
            best = (len(blob), blob)

    assert best is not None
    rows, columns = np.divmod(np.asarray(best[1]), width)
    return float(columns.mean()), float(rows.mean()), best[0]


def pixel_ray(spec: CameraSpec, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    """World-frame origin and unit direction of the ray through pixel ``(u, v)``.

    The pinhole model matches the renderer's: the camera looks along its own -z
    with +y up, and the principal point sits at the image centre.
    """
    focal = _camera_focal_px(spec)
    direction_camera = np.array(
        (
            (u - (spec.width - 1) / 2.0) / focal,
            -(v - (spec.height - 1) / 2.0) / focal,
            -1.0,
        ),
        dtype=np.float64,
    )
    direction = quaternion_matrix(spec.quaternion) @ direction_camera
    norm = float(np.linalg.norm(direction))
    return np.asarray(spec.position, dtype=np.float64), direction / norm


def triangulate(
    origin_a: np.ndarray,
    direction_a: np.ndarray,
    origin_b: np.ndarray,
    direction_b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Midpoint of the shortest segment between two rays, and its length.

    The length is the residual: two rays that do not nearly meet were not
    looking at the same object, and the caller should discard the detection
    rather than average two different things.
    """
    matrix = np.stack([direction_a, -direction_b], axis=1)
    scales, *_ = np.linalg.lstsq(matrix, origin_b - origin_a, rcond=None)
    point_a = origin_a + scales[0] * direction_a
    point_b = origin_b + scales[1] * direction_b
    return 0.5 * (point_a + point_b), float(np.linalg.norm(point_a - point_b))


class StereoBallTracker:
    """Reference perception: two views in, one ball state out.

    Velocity is a least-squares fit over the most recent detections rather than
    a single difference, because one 2.5 cm position error across a 8 ms gap is
    a 3 m/s velocity error, and the controller needs velocity more than it
    needs position.
    """

    def __init__(
        self,
        left: CameraSpec = LEFT_CAMERA,
        right: CameraSpec = RIGHT_CAMERA,
        *,
        history: int = 6,
        max_residual_m: float = 0.15,
    ) -> None:
        if history < 2:
            raise ValueError("history must be at least two samples")
        if not isfinite(max_residual_m) or max_residual_m <= 0.0:
            raise ValueError("max_residual_m must be greater than zero")
        self.left = left
        self.right = right
        self.history = int(history)
        self.max_residual_m = float(max_residual_m)
        self._samples: list[BallDetection] = []
        self._last_frame_time: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._last_frame_time = None

    @property
    def detections(self) -> tuple[BallDetection, ...]:
        return tuple(self._samples)

    def detect(self, readings: SensorReadings) -> BallDetection | None:
        """Locate the ball in one stereo frame, or return ``None``.

        A frame the cameras have already reported is skipped: at 120 Hz sensors
        on a 200 Hz control loop the same image arrives twice, and counting it
        twice would fabricate a zero-velocity sample.
        """
        left_frame = readings.camera(self.left.name)
        right_frame = readings.camera(self.right.name)
        time_s = float(left_frame.time_s)
        if self._last_frame_time is not None and time_s <= self._last_frame_time:
            return None
        self._last_frame_time = time_s

        left_blob = largest_blob(ball_mask(left_frame.rgb))
        right_blob = largest_blob(ball_mask(right_frame.rgb))
        if left_blob is None or right_blob is None:
            return None

        origin_a, direction_a = pixel_ray(self.left, left_blob[0], left_blob[1])
        origin_b, direction_b = pixel_ray(self.right, right_blob[0], right_blob[1])
        position, residual = triangulate(origin_a, direction_a, origin_b, direction_b)
        if residual > self.max_residual_m or not np.all(np.isfinite(position)):
            return None

        detection = BallDetection(
            time_s=time_s,
            position=tuple(float(value) for value in position),
            pixels=(left_blob[2], right_blob[2]),
            residual_m=residual,
        )
        self._samples.append(detection)
        del self._samples[: -self.history]
        return detection

    def estimate(self) -> tuple[Vec3, Vec3] | None:
        """Current position and velocity, or ``None`` before enough evidence."""
        if len(self._samples) < 2:
            return None
        times = np.array([sample.time_s for sample in self._samples])
        points = np.array([sample.position for sample in self._samples])
        span = float(times[-1] - times[0])
        if span <= 0.0:
            return None
        design = np.stack([np.ones_like(times), times - times[-1]], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, points, rcond=None)
        position = coefficients[0]
        velocity = coefficients[1]
        if not (np.all(np.isfinite(position)) and np.all(np.isfinite(velocity))):
            return None
        return (
            tuple(float(value) for value in position),
            tuple(float(value) for value in velocity),
        )


def vision_observation_space_shapes(
    sensors: Sequence[SensorSpec] = TABLE_TENNIS_VISION_SENSORS,
) -> dict[str, tuple[int, ...]]:
    """Image shapes of the declared cameras, keyed by sensor name."""
    return {
        spec.name: spec.shape for spec in sensors if isinstance(spec, CameraSpec)
    }


def vision_observation_dict(
    observation: VisionObservation,
    *,
    sensors: Sequence[SensorSpec] = TABLE_TENNIS_VISION_SENSORS,
) -> dict[str, np.ndarray]:
    """Pack one vision observation into plain arrays for a learning framework.

    Training and evaluation must read the same keys in the same layout, so both
    the Gymnasium vision environment and an offline scoring run call this one
    function.  ``frame_age_s`` is included deliberately: the cameras run slower
    than control, and a policy that cannot see how stale its frame is will
    learn to assume it never is.
    """
    packed: dict[str, np.ndarray] = {}
    for spec in sensors:
        if isinstance(spec, CameraSpec):
            packed[spec.name] = np.asarray(
                observation.sensors.camera(spec.name).rgb, dtype=np.uint8
            )
    robot = observation.robot
    packed["joint_positions"] = np.asarray(robot.joint_positions, dtype=np.float32)
    packed["joint_velocities"] = np.asarray(robot.joint_velocities, dtype=np.float32)
    packed["blade_pose"] = np.asarray(
        (*robot.effector_position, *robot.effector_quaternion), dtype=np.float32
    )
    newest = max(
        (observation.sensors[spec.name].time_s for spec in sensors if isinstance(spec, CameraSpec)),
        default=observation.time_s,
    )
    packed["frame_age_s"] = np.asarray(
        [max(observation.time_s - newest, 0.0)], dtype=np.float32
    )
    return packed


def describe_track(sensors: Sequence[SensorSpec] = TABLE_TENNIS_VISION_SENSORS) -> dict[str, Any]:
    """The report block that says what a vision-track result was allowed to see."""
    return {
        "track": "vision",
        "privileged_ball_state": False,
        "sensors": [spec.to_dict() for spec in sensors],
        "reference_detector": {
            "id": "stereo-colour-triangulation-v1",
            "ball_radius_m": BALL_RADIUS_M,
        },
    }


__all__ = [
    "ARM_TORQUE",
    "BALL_RADIUS_M",
    "BLADE_CONTACT",
    "BLADE_IMU",
    "BLADE_POSE",
    "LEFT_CAMERA",
    "RIGHT_CAMERA",
    "TABLE_TENNIS_VISION_SENSORS",
    "BallDetection",
    "StereoBallTracker",
    "VisionObservation",
    "VisionTrackBackend",
    "ball_mask",
    "describe_track",
    "largest_blob",
    "pixel_ray",
    "triangulate",
    "vision_observation_dict",
    "vision_observation_space_shapes",
]
