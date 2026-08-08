"""MuJoCo reference backend for the table-tennis single-shot benchmark."""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite, sqrt

import mujoco

from ...physics import Aerodynamics, Atmosphere
from ...scene import build_model
from ...specs import Sport
from ..controllers import ControllerObservation, PaddleCommand
from ..types import BallState, SemanticContact, ShotSpec, Vec3


class MujocoShotBackend:
    """One deterministic table-tennis scene with an optional mocap test blade.

    The mocap blade is a diagnostic fixture rather than a robot.  Only contact
    involving its blade geom is exposed as ``ball``/``robot_racket``; contacts
    with decorative paddles or handles can therefore never create a benchmark
    hit event.
    """

    _BALL_JOINT = "table_tennis_ball_free"
    _BALL_BODY = "table_tennis_ball"
    _BALL_GEOM = "table_tennis_ball_geom"
    _PADDLE_BODY = "table_tennis_benchmark_paddle"
    _PADDLE_GEOM = "table_tennis_benchmark_paddle_blade"

    def __init__(
        self,
        *,
        wind: Vec3 = (0.0, 0.0, 0.0),
    ) -> None:
        self.model = build_model(Sport.TABLE_TENNIS.value, benchmark_paddle=True)
        self.data = mujoco.MjData(self.model)
        validated_wind = self._finite_values(wind, size=3, field="wind")
        self.aerodynamics = Aerodynamics(self.model, Atmosphere(wind=validated_wind))

        joint_id = self._require_id(mujoco.mjtObj.mjOBJ_JOINT, self._BALL_JOINT)
        self._ball_qpos_address = int(self.model.jnt_qposadr[joint_id])
        self._ball_dof_address = int(self.model.jnt_dofadr[joint_id])
        self._ball_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, self._BALL_BODY)
        self._ball_geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, self._BALL_GEOM)

        paddle_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, self._PADDLE_BODY)
        self._paddle_mocap_id = int(self.model.body_mocapid[paddle_body_id])
        if self._paddle_mocap_id < 0:  # pragma: no cover - protects scene/backend drift
            raise RuntimeError(f"{self._PADDLE_BODY} is not a mocap body")
        self._paddle_geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, self._PADDLE_GEOM)

        self._semantic_geoms: dict[int, str] = {
            self._paddle_geom_id: "robot_racket",
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "table_tennis_table"): "table",
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "table_tennis_net"): "net",
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "table_tennis_post_a"): "net",
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "table_tennis_post_b"): "net",
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "table_tennis_surface"): "floor",
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "campus_ground"): "floor",
        }
        self.reset()

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
        if shot.sport != Sport.TABLE_TENNIS.value:
            raise ValueError(
                f"MuJoCo table-tennis backend cannot launch sport {shot.sport!r}"
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
