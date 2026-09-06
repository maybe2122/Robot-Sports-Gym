"""Backend-neutral sensor contracts for the observation tracks.

The state track hands a policy privileged truth: the ball's exact position and
velocity, read straight out of the simulator.  Nothing on a real table can
supply that, so a score measured on it says nothing about perception.  The
vision track exists to close that gap, and this module is the boundary that
lets both tracks run the same task, the same judge and the same report:

* A :class:`SensorSpec` declares what a sensor *is* -- its name, its rate, what
  it is mounted on -- in terms no simulator owns.  A task or a submission can
  therefore state its sensor suite before any backend exists.
* A reading carries its own ``time_s``.  Sensors run slower than physics and
  slower than control, so a policy that reads a 60 Hz camera on a 200 Hz
  control step must be able to see that the frame it holds is stale rather
  than silently treating it as current.
* :class:`SensorSuite` is the only interface a task uses.  MuJoCo implements it
  by rendering and reading ``mjData``; an Isaac implementation supplies the same
  names and the same reading types from its own tiled renderer and sensors.

Nothing here imports a simulator.  ``numpy`` is used because an image is an
array, not because a backend is present.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from numbers import Real
from typing import Any, Protocol, runtime_checkable

import numpy as np

Vec3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]

WORLD_FRAME = "world"
"""The mount name meaning "fixed in the task frame, not carried by the robot"."""


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _positive(value: object, *, field: str) -> float:
    result = _finite(value, field=field)
    if result <= 0.0:
        raise ValueError(f"{field} must be greater than zero")
    return result


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _finite_vector(values: object, *, size: int | None, field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must contain finite numbers")
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{field} must contain finite numbers") from exc
    if size is not None and len(items) != size:
        raise ValueError(f"{field} must contain exactly {size} finite numbers")
    return tuple(_finite(item, field=f"{field}[{index}]") for index, item in enumerate(items))


def _unit_quaternion(values: object, *, field: str) -> Quaternion:
    components = _finite_vector(values, size=4, field=field)
    norm = sqrt(sum(value * value for value in components))
    if norm <= 1e-12:
        raise ValueError(f"{field} must have non-zero norm")
    w, x, y, z = (value / norm for value in components)
    return (w, x, y, z)


class SensorKind(str, Enum):
    """What a sensor measures.

    The string values reach reports and submission manifests, so a vision-track
    result can be checked against the suite it claimed to use.
    """

    CAMERA = "camera"
    IMU = "imu"
    CONTACT = "contact"
    JOINT_TORQUE = "joint_torque"
    FRAME_TRANSFORM = "frame_transform"


class SensorUnavailable(RuntimeError):
    """Raised when a suite cannot produce a sensor it declared.

    A distinct type matters: a policy that caught a bare ``KeyError`` and
    carried on would report a score for a perception pipeline that never ran.
    """

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"sensor {name!r} is unavailable: {reason}")
        self.sensor_name = name
        self.reason = reason


@dataclass(frozen=True)
class SensorSpec:
    """What every sensor declares, whatever it measures.

    ``rate_hz`` is the sensor's own sampling rate, independent of the control
    rate.  A suite samples a sensor only when its period has elapsed, so a
    60 Hz camera on a 200 Hz control loop returns the same frame for three
    consecutive steps -- which is what the hardware would do.

    ``mount`` names the frame the sensor rides on: :data:`WORLD_FRAME` for a
    fixed installation, or a robot frame name for an on-board sensor.  The
    backend resolves that name; nothing here knows what a body is.
    """

    name: str
    rate_hz: float
    mount: str = WORLD_FRAME

    kind: SensorKind = SensorKind.FRAME_TRANSFORM

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field="name"))
        object.__setattr__(self, "rate_hz", _positive(self.rate_hz, field="rate_hz"))
        object.__setattr__(self, "mount", _identifier(self.mount, field="mount"))
        kind = self.kind if isinstance(self.kind, SensorKind) else SensorKind(self.kind)
        object.__setattr__(self, "kind", kind)

    @property
    def period_s(self) -> float:
        return 1.0 / self.rate_hz

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "rate_hz": self.rate_hz,
            "mount": self.mount,
        }


@dataclass(frozen=True)
class CameraSpec(SensorSpec):
    """A pinhole camera, declared in the units a submission can reproduce.

    Resolution and vertical field of view are part of the task definition, not
    a rendering detail: a policy trained at 320x240 and scored at 64x48 has not
    been scored on what it trained on.  They therefore appear in the report.
    """

    width: int = 160
    height: int = 120
    fovy_deg: float = 60.0
    depth: bool = False
    position: Vec3 = (0.0, 0.0, 0.0)
    quaternion: Quaternion = (1.0, 0.0, 0.0, 0.0)
    kind: SensorKind = SensorKind.CAMERA

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not SensorKind.CAMERA:
            raise ValueError("CameraSpec requires SensorKind.CAMERA")
        object.__setattr__(self, "width", _positive_int(self.width, field="width"))
        object.__setattr__(self, "height", _positive_int(self.height, field="height"))
        fovy = _positive(self.fovy_deg, field="fovy_deg")
        if fovy >= 180.0:
            raise ValueError("fovy_deg must be less than 180")
        object.__setattr__(self, "fovy_deg", fovy)
        object.__setattr__(self, "depth", bool(self.depth))
        object.__setattr__(
            self, "position", _finite_vector(self.position, size=3, field="position")
        )
        object.__setattr__(
            self, "quaternion", _unit_quaternion(self.quaternion, field="quaternion")
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "width": self.width,
            "height": self.height,
            "fovy_deg": self.fovy_deg,
            "depth": self.depth,
            "position": list(self.position),
            "quaternion": list(self.quaternion),
        }


@dataclass(frozen=True)
class ImuSpec(SensorSpec):
    """A strapdown IMU: linear acceleration, angular rate, orientation."""

    kind: SensorKind = SensorKind.IMU

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not SensorKind.IMU:
            raise ValueError("ImuSpec requires SensorKind.IMU")
        if self.mount == WORLD_FRAME:
            raise ValueError("an IMU must be mounted on a moving frame, not the world")


@dataclass(frozen=True)
class ContactSensorSpec(SensorSpec):
    """Reports which semantic categories the mounted body is touching.

    It reports *semantic* names -- ``ball``, ``table``, ``floor`` -- because a
    policy that keyed on geom ids would be reading the scene file, not the
    world.
    """

    kind: SensorKind = SensorKind.CONTACT

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not SensorKind.CONTACT:
            raise ValueError("ContactSensorSpec requires SensorKind.CONTACT")


@dataclass(frozen=True)
class JointTorqueSpec(SensorSpec):
    """Measures the torque the actuators actually delivered, joint by joint."""

    joint_names: tuple[str, ...] = ()
    kind: SensorKind = SensorKind.JOINT_TORQUE

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not SensorKind.JOINT_TORQUE:
            raise ValueError("JointTorqueSpec requires SensorKind.JOINT_TORQUE")
        names = tuple(self.joint_names)
        if not names:
            raise ValueError("joint_names must contain at least one joint")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("joint_names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique")
        object.__setattr__(self, "joint_names", names)

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "joint_names": list(self.joint_names)}


@dataclass(frozen=True)
class FrameTransformSpec(SensorSpec):
    """The pose of :attr:`target` expressed in :attr:`mount`'s frame.

    This is the sensor a real system gets from its kinematic chain and its
    calibration, and it is the honest way for a policy to know where its own
    blade is without reading simulator state.
    """

    target: str = ""
    kind: SensorKind = SensorKind.FRAME_TRANSFORM

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not SensorKind.FRAME_TRANSFORM:
            raise ValueError("FrameTransformSpec requires SensorKind.FRAME_TRANSFORM")
        object.__setattr__(self, "target", _identifier(self.target, field="target"))


@dataclass(frozen=True)
class SensorReading:
    """The part of a reading every sensor shares."""

    name: str
    time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field="name"))
        object.__setattr__(self, "time_s", _finite(self.time_s, field="time_s"))


@dataclass(frozen=True)
class CameraFrame(SensorReading):
    """One rendered frame: ``uint8`` RGB, optionally with a metric depth map."""

    rgb: np.ndarray = None  # type: ignore[assignment]
    depth: np.ndarray | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        rgb = np.asarray(self.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (height, width, 3)")
        if rgb.dtype != np.uint8:
            raise ValueError("rgb must be uint8; a policy sees what a camera sends")
        object.__setattr__(self, "rgb", rgb)
        if self.depth is not None:
            depth = np.asarray(self.depth, dtype=np.float32)
            if depth.shape != rgb.shape[:2]:
                raise ValueError("depth must match the rgb frame's height and width")
            object.__setattr__(self, "depth", depth)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.rgb.shape)  # type: ignore[return-value]


@dataclass(frozen=True)
class ImuReading(SensorReading):
    """Proper acceleration and angular rate in the sensor's own frame."""

    linear_acceleration: Vec3 = (0.0, 0.0, 0.0)
    angular_velocity: Vec3 = (0.0, 0.0, 0.0)
    orientation: Quaternion = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        for field_name in ("linear_acceleration", "angular_velocity"):
            object.__setattr__(
                self,
                field_name,
                _finite_vector(getattr(self, field_name), size=3, field=field_name),
            )
        object.__setattr__(
            self, "orientation", _unit_quaternion(self.orientation, field="orientation")
        )


@dataclass(frozen=True)
class ContactReading(SensorReading):
    """Semantic categories currently touched, with their normal forces."""

    categories: tuple[str, ...] = ()
    forces_n: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        categories = tuple(self.categories)
        if any(not isinstance(name, str) or not name.strip() for name in categories):
            raise ValueError("categories must be non-empty strings")
        object.__setattr__(self, "categories", categories)
        forces = _finite_vector(self.forces_n, size=len(categories), field="forces_n")
        if any(value < 0.0 for value in forces):
            raise ValueError("forces_n must be non-negative")
        object.__setattr__(self, "forces_n", forces)

    def touching(self, category: str) -> bool:
        return category in self.categories


@dataclass(frozen=True)
class JointTorqueReading(SensorReading):
    """Delivered actuator torque, ordered like the spec's ``joint_names``."""

    joint_names: tuple[str, ...] = ()
    torques_nm: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        names = tuple(self.joint_names)
        if not names:
            raise ValueError("joint_names must contain at least one joint")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(
            self,
            "torques_nm",
            _finite_vector(self.torques_nm, size=len(names), field="torques_nm"),
        )


@dataclass(frozen=True)
class FrameTransformReading(SensorReading):
    """Pose of the target frame expressed in the mount frame."""

    position: Vec3 = (0.0, 0.0, 0.0)
    quaternion: Quaternion = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self, "position", _finite_vector(self.position, size=3, field="position")
        )
        object.__setattr__(
            self, "quaternion", _unit_quaternion(self.quaternion, field="quaternion")
        )


class SensorReadings(Mapping[str, SensorReading]):
    """The readings a suite produced, addressed by sensor name.

    Typed accessors exist so a perception pipeline fails loudly on the wrong
    sensor kind instead of unpacking a differently shaped tuple.
    """

    def __init__(self, readings: Iterable[SensorReading] = ()) -> None:
        items: dict[str, SensorReading] = {}
        for reading in readings:
            if not isinstance(reading, SensorReading):
                raise ValueError("readings must be SensorReading instances")
            if reading.name in items:
                raise ValueError(f"duplicate reading for sensor {reading.name!r}")
            items[reading.name] = reading
        self._items = items

    def __getitem__(self, name: str) -> SensorReading:
        try:
            return self._items[name]
        except KeyError:
            raise SensorUnavailable(name, "no reading was produced this step") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def _typed(self, name: str, expected: type) -> Any:
        reading = self[name]
        if not isinstance(reading, expected):
            raise SensorUnavailable(
                name, f"expected {expected.__name__}, got {type(reading).__name__}"
            )
        return reading

    def camera(self, name: str) -> CameraFrame:
        return self._typed(name, CameraFrame)

    def imu(self, name: str) -> ImuReading:
        return self._typed(name, ImuReading)

    def contact(self, name: str) -> ContactReading:
        return self._typed(name, ContactReading)

    def joint_torque(self, name: str) -> JointTorqueReading:
        return self._typed(name, JointTorqueReading)

    def frame_transform(self, name: str) -> FrameTransformReading:
        return self._typed(name, FrameTransformReading)


class SensorSchedule:
    """Decides which sensors are due, so rate is enforced in one place.

    Every backend would otherwise reimplement this, and any two that rounded
    differently would hand policies different frame counts for the same task.
    The first sample of an episode is always due: a policy must not have to
    step blind before its camera turns on.
    """

    def __init__(self, specs: Iterable[SensorSpec], *, tolerance_s: float = 1e-9) -> None:
        self._specs = tuple(specs)
        names = [spec.name for spec in self._specs]
        if len(set(names)) != len(names):
            raise ValueError("sensor names must be unique within a suite")
        self._tolerance_s = _positive(tolerance_s, field="tolerance_s")
        self._last: dict[str, float] = {}

    @property
    def specs(self) -> tuple[SensorSpec, ...]:
        return self._specs

    def reset(self) -> None:
        self._last.clear()

    def due(self, time_s: float) -> tuple[SensorSpec, ...]:
        """Return the sensors whose period has elapsed by ``time_s``."""
        now = _finite(time_s, field="time_s")
        ready: list[SensorSpec] = []
        for spec in self._specs:
            last = self._last.get(spec.name)
            if last is None or now - last >= spec.period_s - self._tolerance_s:
                ready.append(spec)
        return tuple(ready)

    def mark(self, specs: Iterable[SensorSpec], time_s: float) -> None:
        now = _finite(time_s, field="time_s")
        for spec in specs:
            self._last[spec.name] = now


@runtime_checkable
class SensorSuite(Protocol):
    """The only sensing interface a task or a policy uses.

    An implementation owns its simulator entirely.  It must honour the declared
    rates, return readings whose ``time_s`` is the simulation time the
    measurement was taken at, and raise :class:`SensorUnavailable` rather than
    fabricate a reading it cannot produce.
    """

    @property
    def specs(self) -> tuple[SensorSpec, ...]:
        ...

    def reset(self) -> None:
        ...

    def sample(self, time_s: float) -> SensorReadings:
        ...

    def describe(self) -> dict[str, Any]:
        ...


def look_at_quaternion(
    eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)
) -> Quaternion:
    """Orientation of a camera at ``eye`` pointing at ``target``.

    The convention is the one every renderer in this project uses: the camera
    looks along its own -z axis with +y up.  Declaring a camera by where it
    looks -- rather than by four hand-tuned numbers -- is what lets a sensor
    suite be reviewed, and lets a backend place the camera from the spec
    instead of the spec being copied from the scene file.
    """
    eye_v = np.asarray(_finite_vector(eye, size=3, field="eye"), dtype=np.float64)
    target_v = np.asarray(_finite_vector(target, size=3, field="target"), dtype=np.float64)
    up_v = np.asarray(_finite_vector(up, size=3, field="up"), dtype=np.float64)

    backward = eye_v - target_v
    norm = float(np.linalg.norm(backward))
    if norm <= 1e-12:
        raise ValueError("eye and target must differ")
    z_axis = backward / norm
    right = np.cross(up_v, z_axis)
    if float(np.linalg.norm(right)) <= 1e-9:
        raise ValueError("up must not be parallel to the viewing direction")
    x_axis = right / float(np.linalg.norm(right))
    y_axis = np.cross(z_axis, x_axis)

    matrix = np.column_stack((x_axis, y_axis, z_axis))
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diagonal(matrix)))
        if index == 0:
            scale = sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    return _unit_quaternion((w, x, y, z), field="quaternion")


def quaternion_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a ``wxyz`` quaternion, as a 3x3 float array."""
    w, x, y, z = _unit_quaternion(quaternion, field="quaternion")
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def describe_specs(specs: Sequence[SensorSpec]) -> dict[str, Any]:
    """Render a suite's declaration for the report envelope."""
    return {
        "count": len(specs),
        "sensors": [spec.to_dict() for spec in specs],
    }


__all__ = [
    "WORLD_FRAME",
    "CameraFrame",
    "CameraSpec",
    "ContactReading",
    "ContactSensorSpec",
    "FrameTransformReading",
    "FrameTransformSpec",
    "ImuReading",
    "ImuSpec",
    "JointTorqueReading",
    "JointTorqueSpec",
    "SensorKind",
    "SensorReading",
    "SensorReadings",
    "SensorSchedule",
    "SensorSpec",
    "SensorSuite",
    "SensorUnavailable",
    "describe_specs",
    "look_at_quaternion",
    "quaternion_matrix",
]
