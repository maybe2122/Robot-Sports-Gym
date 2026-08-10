"""MuJoCo reference backend for single-shot sports benchmarks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite, sqrt

import mujoco

from ...physics import Aerodynamics, Atmosphere
from ...scene import build_model
from ...specs import Sport
from ..controllers import ControllerObservation, PaddleCommand
from ..task_config import TaskFrame
from ..types import BallState, SemanticContact, ShotSpec, Vec3


@dataclass(frozen=True)
class MujocoSportProfile:
    """Which geoms of one sport's scene carry which benchmark meaning.

    Scene geometry is named by convention (``{sport}_ball``, ``{sport}_surface``
    and so on), so a profile only has to name the pieces that differ: the
    benchmark effector fixture and the sport's own playing surface and net.
    """

    sport: Sport
    effector_body: str
    effector_geom: str
    surface_geoms: tuple[str, ...] = ()
    net_geoms: tuple[str, ...] = ()
    floor_geoms: tuple[str, ...] = ()
    extra_geoms: Mapping[str, str] = field(default_factory=dict)

    @property
    def ball_joint(self) -> str:
        return f"{self.sport.value}_ball_free"

    @property
    def ball_body(self) -> str:
        return f"{self.sport.value}_ball"

    @property
    def ball_geom(self) -> str:
        return f"{self.sport.value}_ball_geom"

    def semantic_geom_names(self) -> dict[str, str]:
        """Map geom name to semantic category, effector included."""
        mapping: dict[str, str] = {self.effector_geom: "robot_racket"}
        for name in self.surface_geoms:
            mapping[name] = "table"
        for name in self.net_geoms:
            mapping[name] = "net"
        for name in (*self.floor_geoms, f"{self.sport.value}_surface", "campus_ground"):
            mapping.setdefault(name, "floor")
        mapping.update(self.extra_geoms)
        return mapping


TABLE_TENNIS_PROFILE = MujocoSportProfile(
    sport=Sport.TABLE_TENNIS,
    effector_body="table_tennis_benchmark_paddle",
    effector_geom="table_tennis_benchmark_paddle_blade",
    surface_geoms=("table_tennis_table",),
    net_geoms=("table_tennis_net", "table_tennis_post_a", "table_tennis_post_b"),
)

PROFILES: dict[Sport, MujocoSportProfile] = {Sport.TABLE_TENNIS: TABLE_TENNIS_PROFILE}


class MujocoShotBackend:
    """One deterministic single-sport scene with a mocap test effector.

    The mocap effector is a diagnostic fixture rather than a robot.  Only
    contact involving its geom is exposed as ``ball``/``robot_racket``; contacts
    with decorative rackets or handles can therefore never create a benchmark
    hit event.
    """

    def __init__(
        self,
        *,
        sport: Sport | str = Sport.TABLE_TENNIS,
        wind: Vec3 = (0.0, 0.0, 0.0),
    ) -> None:
        selected = Sport(sport)
        try:
            self.profile = PROFILES[selected]
        except KeyError:
            raise ValueError(
                f"no MuJoCo benchmark profile is defined for sport {selected.value!r}"
            ) from None
        self.sport = selected
        self.model = build_model(selected.value, benchmark_paddle=True)
        self.data = mujoco.MjData(self.model)
        validated_wind = self._finite_values(wind, size=3, field="wind")
        self.aerodynamics = Aerodynamics(self.model, Atmosphere(wind=validated_wind))

        joint_id = self._require_id(mujoco.mjtObj.mjOBJ_JOINT, self.profile.ball_joint)
        self._ball_qpos_address = int(self.model.jnt_qposadr[joint_id])
        self._ball_dof_address = int(self.model.jnt_dofadr[joint_id])
        self._ball_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, self.profile.ball_body)
        self._ball_geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, self.profile.ball_geom)

        effector_body = self.profile.effector_body
        paddle_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, effector_body)
        self._paddle_mocap_id = int(self.model.body_mocapid[paddle_body_id])
        if self._paddle_mocap_id < 0:  # pragma: no cover - protects scene/backend drift
            raise RuntimeError(f"{effector_body} is not a mocap body")
        self._paddle_geom_id = self._require_id(
            mujoco.mjtObj.mjOBJ_GEOM, self.profile.effector_geom
        )

        self._semantic_geoms: dict[int, str] = {
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, name): category
            for name, category in self.profile.semantic_geom_names().items()
        }
        self.reset()

    @property
    def task_frame(self) -> TaskFrame:
        """Identity: the single-sport MuJoCo scene builds the table at the origin."""
        return TaskFrame()

    def _require_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = int(mujoco.mj_name2id(self.model, object_type, name))
        if object_id < 0:
            raise RuntimeError(f"MuJoCo benchmark model is missing {name!r}")
        return object_id

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self) -> None:
        """Restore the complete scene, including the blade's parked mocap pose."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def launch_ball(self, shot: ShotSpec) -> None:
        """Apply a versioned world-frame pose, velocity, and spin to the ball."""
        if shot.sport != self.sport.value:
            raise ValueError(
                f"MuJoCo {self.sport.value} backend cannot launch sport {shot.sport!r}"
            )

        qpos = self._ball_qpos_address
        dof = self._ball_dof_address
        self.data.qpos[qpos : qpos + 3] = shot.position
        # MuJoCo free-joint quaternion order is (w, x, y, z).
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
            position=tuple(float(value) for value in self.data.qpos[qpos : qpos + 3]),
            linear_velocity=tuple(
                float(value) for value in self.data.qvel[dof : dof + 3]
            ),
            angular_velocity=tuple(
                float(value) for value in self.data.qvel[dof + 3 : dof + 6]
            ),
        )

    def semantic_contacts(self) -> tuple[SemanticContact, ...]:
        """Return actual ball contacts normalized to benchmark categories.

        MuJoCo contact normals are converted to point from the ball toward the
        other semantic participant, independent of geom ordering.
        """
        contacts: list[SemanticContact] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if self._ball_geom_id == geom1:
                other_geom = geom2
                normal_sign = 1.0
            elif self._ball_geom_id == geom2:
                other_geom = geom1
                normal_sign = -1.0
            else:
                continue

            category = self._semantic_geoms.get(other_geom)
            # Geom margins are zero in this scene.  Checking signed distance
            # nevertheless prevents a future positive-margin proximity pair
            # from being interpreted as a physical racket strike.
            if category is None or float(contact.dist) > 0.0:
                continue
            position = tuple(float(value) for value in contact.pos)
            normal = tuple(normal_sign * float(value) for value in contact.frame[:3])
            contacts.append(
                SemanticContact.between(
                    "ball",
                    category,
                    position=position,
                    normal=normal,
                )
            )
        return tuple(contacts)

    def observe(self) -> ControllerObservation:
        return ControllerObservation(
            time_s=self.time,
            ball=self.get_ball_state(),
            paddle_position=tuple(
                float(value) for value in self.data.mocap_pos[self._paddle_mocap_id]
            ),
            paddle_quaternion=tuple(
                float(value) for value in self.data.mocap_quat[self._paddle_mocap_id]
            ),
        )

    @staticmethod
    def _finite_values(values: object, *, size: int, field: str) -> tuple[float, ...]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise ValueError(f"{field} must contain {size} finite numbers")
        try:
            result = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain {size} finite numbers") from exc
        if len(result) != size or not all(isfinite(value) for value in result):
            raise ValueError(f"{field} must contain {size} finite numbers")
        return result

    def apply_action(self, action: object | None) -> None:
        if action is None:
            return
        if not isinstance(action, PaddleCommand):
            raise TypeError(
                "MujocoShotBackend expects PaddleCommand or None; "
                f"received {type(action).__name__}"
            )
        position = self._finite_values(action.position, size=3, field="paddle position")
        quaternion = self._finite_values(
            action.quaternion, size=4, field="paddle quaternion"
        )
        quaternion_norm = sqrt(sum(value * value for value in quaternion))
        if quaternion_norm <= 1e-12:
            raise ValueError("paddle quaternion must have non-zero norm")

        self.data.mocap_pos[self._paddle_mocap_id] = position
        self.data.mocap_quat[self._paddle_mocap_id] = tuple(
            value / quaternion_norm for value in quaternion
        )

    def step(self, action: object | None = None) -> None:
        """Apply passive aerodynamics and advance exactly one physics tick."""
        if action is not None:
            self.apply_action(action)
        self.data.xfrc_applied[:] = 0.0
        self.aerodynamics.apply(self.data)
        mujoco.mj_step(self.model, self.data)


# More explicit alias for code that supports multiple sports in one process.
MujocoTableTennisBackend = MujocoShotBackend
