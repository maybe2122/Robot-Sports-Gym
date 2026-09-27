"""MuJoCo reference backend for single-shot sports benchmarks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite, sqrt

import mujoco
import numpy as np

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
    # Spread each commanded effector pose over this many seconds of physics
    # steps instead of teleporting there at once.  Zero keeps the teleport the
    # frozen table-tennis and tennis scores were measured with; a sport whose
    # ball is smaller than the distance the face covers in one control period
    # needs it, or the face jumps straight past the ball.
    effector_interpolation_s: float = 0.0
    # Where the task frame's origin sits in the scene.  Net sports put it at
    # the centre of the court; a launch task at a goal puts it under the goal,
    # so the robot is at x < 0 and shoots toward +x like every other task.
    task_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

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

TENNIS_PROFILE = MujocoSportProfile(
    sport=Sport.TENNIS,
    effector_body="tennis_benchmark_paddle",
    effector_geom="tennis_benchmark_paddle_blade",
    # In tennis the court *is* the playing surface: a ball that reaches the
    # ground has landed, in or out, and there is no floor underneath it to be a
    # separate failure.  The line markings are part of that same surface.
    surface_geoms=(
        "tennis_surface",
        "tennis_singles_line_n",
        "tennis_singles_line_s",
        "tennis_singles_line_e",
        "tennis_singles_line_w",
        "tennis_doubles_line_n",
        "tennis_doubles_line_s",
        "tennis_doubles_line_e",
        "tennis_doubles_line_w",
        "tennis_service_e",
        "tennis_service_w",
        "tennis_service_center",
    ),
    net_geoms=("tennis_net", "tennis_post_a", "tennis_post_b"),
)

BADMINTON_PROFILE = MujocoSportProfile(
    sport=Sport.BADMINTON,
    effector_body="badminton_benchmark_paddle",
    effector_geom="badminton_benchmark_paddle_blade",
    # The court is the ground, as in tennis; its painted lines have no
    # collision geometry, so the surface geom alone is the landing surface.
    surface_geoms=("badminton_surface",),
    net_geoms=("badminton_net", "badminton_post_a", "badminton_post_b"),
    # One 200 Hz control period.  A 30 m/s face covers 150 mm in it; the
    # shuttle's cork is 27 mm across.
    effector_interpolation_s=0.005,
)

FOOTBALL_PROFILE = MujocoSportProfile(
    sport=Sport.FOOTBALL,
    effector_body="football_benchmark_paddle",
    effector_geom="football_benchmark_paddle_blade",
    extra_geoms={
        "football_goal_e_left_post": "post",
        "football_goal_e_right_post": "post",
        "football_goal_e_crossbar": "post",
        "football_goal_e_goal_net": "goal_net",
    },
    effector_interpolation_s=0.005,
    # The east goal line, so the kicker is at x < 0 shooting toward +x.
    task_origin=(52.5, 0.0, 0.0),
)

BASKETBALL_PROFILE = MujocoSportProfile(
    sport=Sport.BASKETBALL,
    effector_body="basketball_benchmark_paddle",
    effector_geom="basketball_benchmark_paddle_blade",
    extra_geoms={
        **{f"basketball_e_rim_{index}": "rim" for index in range(32)},
        "basketball_e_backboard": "backboard",
    },
    effector_interpolation_s=0.005,
    # The floor below the east rim's centre: the rim is at (0, 0, 3.05).
    task_origin=(12.425, 0.0, 0.0),
)

PROFILES: dict[Sport, MujocoSportProfile] = {
    Sport.TABLE_TENNIS: TABLE_TENNIS_PROFILE,
    Sport.TENNIS: TENNIS_PROFILE,
    Sport.BADMINTON: BADMINTON_PROFILE,
    Sport.FOOTBALL: FOOTBALL_PROFILE,
    Sport.BASKETBALL: BASKETBALL_PROFILE,
}


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
        self._frame = TaskFrame(origin_xyz=self.profile.task_origin)
        self._origin = np.asarray(self.profile.task_origin, dtype=float)
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

        self._interpolation_steps = max(
            0, round(self.profile.effector_interpolation_s / self.timestep)
        )
        self._effector_from: tuple[np.ndarray, np.ndarray] | None = None
        self._effector_to: tuple[np.ndarray, np.ndarray] | None = None
        self._effector_step = 0

        self._semantic_geoms: dict[int, str] = {
            self._require_id(mujoco.mjtObj.mjOBJ_GEOM, name): category
            for name, category in self.profile.semantic_geom_names().items()
        }
        self.reset()

    @property
    def task_frame(self) -> TaskFrame:
        """The sport's task frame; identity for the net sports.

        Everything this backend hands out -- ball state, contact points,
        observations -- is in this frame, and everything it takes in -- shot
        states, effector poses -- is read in it, so a judge never sees a
        scene coordinate.
        """
        return self._frame

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
        self._effector_from = self._effector_to = None
        self._effector_step = 0
        mujoco.mj_forward(self.model, self.data)

    def launch_ball(self, shot: ShotSpec) -> None:
        """Apply a versioned world-frame pose, velocity, and spin to the ball."""
        if shot.sport != self.sport.value:
            raise ValueError(
                f"MuJoCo {self.sport.value} backend cannot launch sport {shot.sport!r}"
            )

        qpos = self._ball_qpos_address
        dof = self._ball_dof_address
        self.data.qpos[qpos : qpos + 3] = np.asarray(shot.position) + self._origin
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
            position=tuple(
                float(value) for value in self.data.qpos[qpos : qpos + 3] - self._origin
            ),
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
            position = tuple(float(value) for value in contact.pos - self._origin)
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
                float(value)
                for value in self.data.mocap_pos[self._paddle_mocap_id] - self._origin
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

        target = (
            np.asarray(position, dtype=float) + self._origin,
            np.asarray(quaternion, dtype=float) / quaternion_norm,
        )
        if self._interpolation_steps == 0:
            self.data.mocap_pos[self._paddle_mocap_id] = target[0]
            self.data.mocap_quat[self._paddle_mocap_id] = target[1]
            return
        self._effector_from = (
            self.data.mocap_pos[self._paddle_mocap_id].copy(),
            self.data.mocap_quat[self._paddle_mocap_id].copy(),
        )
        self._effector_to = target
        self._effector_step = 0

    def _advance_effector(self) -> None:
        """Move the face one physics step along its commanded segment."""
        if self._effector_to is None or self._effector_from is None:
            return
        if self._effector_step >= self._interpolation_steps:
            return
        self._effector_step += 1
        fraction = self._effector_step / self._interpolation_steps
        start_position, start_quaternion = self._effector_from
        end_position, end_quaternion = self._effector_to
        # Normalized linear interpolation, taking the short way round.
        if float(start_quaternion @ end_quaternion) < 0.0:
            end_quaternion = -end_quaternion
        quaternion = (1.0 - fraction) * start_quaternion + fraction * end_quaternion
        self.data.mocap_pos[self._paddle_mocap_id] = (
            (1.0 - fraction) * start_position + fraction * end_position
        )
        self.data.mocap_quat[self._paddle_mocap_id] = quaternion / np.linalg.norm(quaternion)

    def step(self, action: object | None = None) -> None:
        """Apply passive aerodynamics and advance exactly one physics tick."""
        if action is not None:
            self.apply_action(action)
        self._advance_effector()
        self.data.xfrc_applied[:] = 0.0
        self.aerodynamics.apply(self.data)
        mujoco.mj_step(self.model, self.data)


# More explicit alias for code that supports multiple sports in one process.
MujocoTableTennisBackend = MujocoShotBackend
