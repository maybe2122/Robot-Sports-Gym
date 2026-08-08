"""Simulation lifecycle, launch controls and viewer integration."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco

from .physics import Aerodynamics, Atmosphere
from .scene import SCENES, build_model
from .specs import BALLS, Sport


@dataclass
class RunStats:
    simulated_seconds: float
    steps: int
    max_ball_height: float


class Simulation:
    """One deterministic scene with passive aerodynamic loads."""

    def __init__(
        self,
        scene: str = "campus",
        wind: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        if scene not in SCENES:
            raise ValueError(f"unknown scene {scene!r}")
        self.scene = scene
        self.model = build_model(scene)
        self.data = mujoco.MjData(self.model)
        self.aerodynamics = Aerodynamics(self.model, Atmosphere(wind=wind))
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()
        self._ball_sports = list(self.aerodynamics._body_ids)
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = self._initial_qvel
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def launch(self) -> None:
        """Give every scene ball a sport-appropriate velocity and visible spin."""
        for sport in self._ball_sports:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{sport.value}_ball_free"
            )
            dof_address = self.model.jnt_dofadr[joint_id]
            self.data.qvel[dof_address : dof_address + 3] = BALLS[sport].launch_velocity
            if sport in (Sport.TENNIS, Sport.TABLE_TENNIS):
                self.data.qvel[dof_address + 3 : dof_address + 6] = (0.0, 28.0, 6.0)
            elif sport is Sport.FOOTBALL:
                self.data.qvel[dof_address + 3 : dof_address + 6] = (0.0, 8.0, 0.0)
            elif sport is Sport.BASKETBALL:
                self.data.qvel[dof_address + 3 : dof_address + 6] = (0.0, 3.0, 0.0)
            else:
                self.data.qvel[dof_address + 3 : dof_address + 6] = (0.0, 0.0, 12.0)

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            self.data.xfrc_applied[:] = 0.0
            self.aerodynamics.apply(self.data)
            mujoco.mj_step(self.model, self.data)

    def ball_height(self, sport: Sport) -> float:
        body_id = self.aerodynamics._body_ids[sport]
        return float(self.data.xpos[body_id, 2])

    def run_headless(self, duration: float, launch: bool = True) -> RunStats:
        if launch:
            self.launch()
        steps = max(0, round(duration / self.model.opt.timestep))
        max_height = max((self.ball_height(s) for s in self._ball_sports), default=0.0)
        for _ in range(steps):
            self.step()
            max_height = max(
                max_height,
                *(self.ball_height(sport) for sport in self._ball_sports),
            )
        return RunStats(float(self.data.time), steps, max_height)

    def run_viewer(self, duration: float | None = None, launch: bool = True) -> None:
        """Run the native viewer. Space launches and R restores the complete scene."""
        try:
            import mujoco.viewer
        except ImportError as exc:  # pragma: no cover - depends on desktop extras
            raise RuntimeError("MuJoCo viewer support is unavailable") from exc

        controls = {"reset": False, "launch": False}

        def on_key(keycode: int) -> None:
            if keycode in (ord("R"), ord("r")):
                controls["reset"] = True
            elif keycode == ord(" "):
                controls["launch"] = True

        if launch:
            self.launch()
        start = time.monotonic()
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=on_key, show_left_ui=False, show_right_ui=False
        ) as viewer:
            camera_name = f"{self.scene}_camera"
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id >= 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera_id
            while viewer.is_running() and (duration is None or time.monotonic() - start < duration):
                frame_start = time.monotonic()
                if controls["reset"]:
                    self.reset()
                    controls["reset"] = False
                if controls["launch"]:
                    self.launch()
                    controls["launch"] = False
                self.step()
                viewer.sync()
                delay = self.model.opt.timestep - (time.monotonic() - frame_start)
                if delay > 0:
                    time.sleep(delay)

