"""Depth-only ball tracking from a fixed calibrated camera and a background frame."""

from math import isfinite, log, pi

import numpy as np

from .sensors import CameraSpec, SensorReadings, quaternion_matrix
from .vision import BALL_RADIUS_M, BallDetection, RGBDBallTracker, _camera_focal_px, pixel_ray


def _components(mask):
    height, width = mask.shape
    remaining = mask.ravel().copy()
    for start in np.flatnonzero(remaining):
        if not remaining[start]:
            continue
        stack, pixels = [int(start)], []
        remaining[start] = False
        while stack:
            index = stack.pop()
            pixels.append(index)
            row, column = divmod(index, width)
            for r, c in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if 0 <= r < height and 0 <= c < width and remaining[r * width + c]:
                    remaining[r * width + c] = False
                    stack.append(r * width + c)
        yield np.divmod(np.asarray(pixels), width)


class DepthBallTracker(RGBDBallTracker):
    """Reject large robot surfaces; identify a small moving sphere using depth only.

    The background is an actual depth capture before the first serve, not a
    segmentation-ID buffer or ball ground truth. It remains fixed across resets.
    This reference assumes an orange-ball-sized sphere on the table-tennis
    approach corridor, but never reads RGB or assumes a colour.
    """

    def __init__(self, camera: CameraSpec, background: np.ndarray, **kwargs):
        super().__init__(camera, **kwargs)
        self.background = np.asarray(background, dtype=np.float32).copy()
        if self.background.shape != (camera.height, camera.width):
            raise ValueError("depth background must match the configured camera")
        if not np.all(np.isfinite(self.background)) or np.any(self.background <= 0):
            raise ValueError("depth background must contain positive finite metric depth")

    def detect(self, readings: SensorReadings) -> BallDetection | None:
        frame = readings.camera(self.left.name)
        if frame.depth is None:
            raise ValueError("depth-only tracking requires a metric depth frame")
        depth = np.asarray(frame.depth)
        if depth.shape != self.background.shape:
            raise ValueError("depth frame shape differs from calibration")
        time_s = float(frame.time_s)
        if self._last_frame_time is not None and time_s <= self._last_frame_time:
            return None
        self._last_frame_time = time_s
        mask = np.isfinite(depth) & (depth > 0) & (depth < self.background - 0.03)
        candidates = []
        focal = _camera_focal_px(self.left)
        estimate = self.estimate()
        optical = quaternion_matrix(self.left.quaternion) @ np.array([0.0, 0.0, -1.0])
        for rows, columns in _components(mask):
            count = len(rows)
            if not 2 <= count <= 150:
                continue
            height, width = np.ptp(rows) + 1, np.ptp(columns) + 1
            if max(height, width) > 2.5 * min(height, width):
                continue
            values = depth[rows, columns]
            if np.ptp(values) > 2.5 * BALL_RADIUS_M:
                continue
            z = float(np.median(values))
            if not isfinite(z) or z <= 0:
                continue
            expected = pi * (focal * BALL_RADIUS_M / z) ** 2
            if not 0.25 * expected <= count <= 2.5 * expected:
                continue
            origin, ray = pixel_ray(self.left, float(columns.mean()), float(rows.mean()))
            position = origin + ray * (z / float(ray @ optical) + BALL_RADIUS_M)
            if np.any(position < [-2.3, -1.0, 0.65]) or np.any(position > [2.5, 1.0, 1.8]):
                continue
            score = abs(log(count / expected))
            if estimate is not None:
                dt = time_s - self._samples[-1].time_s
                predicted = np.asarray(estimate[0]) + np.asarray(estimate[1]) * dt
                error = float(np.linalg.norm(position - predicted))
                if dt < 0.1 and error > 0.3:
                    continue
                score += 10 * error
            candidates.append((score, position, count))
        if not candidates:
            return None
        _, position, count = min(candidates, key=lambda c: c[0])
        detection = BallDetection(time_s, tuple(position), (count, 0), 0.0)
        self._samples.append(detection)
        del self._samples[: -self.history]
        return detection
