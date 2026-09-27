"""Display or record the actual scored rollout, without changing its controls."""

from pathlib import Path
from time import perf_counter, sleep

import mujoco
import numpy as np


class RolloutDisplay:
    def __init__(self, backend, *, viewer: bool = False, video: Path | None = None):
        self.backend = backend
        self.viewer_enabled = viewer
        self.video = video
        self.viewer = None
        self.renderer = None
        self.frames = []
        self.last_shot = None
        self.last_frame = -1.0
        self.wall_start = 0.0

    def __enter__(self):
        try:
            if self.viewer_enabled:
                from mujoco import viewer as mj_viewer

                self.viewer = mj_viewer.launch_passive(self.backend.model, self.backend.data)
                self._camera(self.viewer.cam)
            if self.video is not None:
                if self.video.suffix.lower() != ".gif":
                    raise ValueError("--video must end in .gif")
                from PIL import Image  # noqa: F401

                self.renderer = mujoco.Renderer(self.backend.model, height=400, width=640)
                self.camera = mujoco.MjvCamera()
                self._camera(self.camera)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    @staticmethod
    def _camera(camera):
        camera.lookat[:] = (-0.6, 0.0, 1.0)
        camera.distance = 4.5
        camera.azimuth = 135.0
        camera.elevation = -18.0

    def __call__(self, backend, shot, judge):
        if shot.shot_id != self.last_shot:
            self.last_shot = shot.shot_id
            self.last_frame = -1.0
            self.wall_start = perf_counter()
        if backend.time - self.last_frame < 1 / 25 and not judge.done:
            return
        self.last_frame = backend.time
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise RuntimeError("viewer closed; rollout interrupted")
            self.viewer.sync()
            sleep(max(0.0, backend.time - (perf_counter() - self.wall_start)))
        if self.renderer is not None:
            from PIL import Image, ImageDraw

            self.renderer.update_scene(backend.data, camera=self.camera)
            frame = Image.fromarray(np.array(self.renderer.render()))
            draw = ImageDraw.Draw(frame)
            result = judge.result
            label = (
                f"Embodied table tennis | {shot.shot_id} | t={backend.time:.2f}s\n"
                f"hit={result.hit} valid_return={result.valid_return} "
                f"failure={result.failure_reason or '-'}"
            )
            draw.rectangle((0, 0, 640, 34), fill=(20, 20, 20))
            draw.text((5, 3), label, fill=(255, 255, 255))
            self.frames.append(frame)

    def __exit__(self, exc_type, exc, traceback):
        if self.renderer is not None:
            self.renderer.close()
        if self.viewer is not None:
            self.viewer.close()
        if exc_type is None and self.video is not None and self.frames:
            self.video.parent.mkdir(parents=True, exist_ok=True)
            self.frames[0].save(
                self.video, save_all=True, append_images=self.frames[1:], duration=40, loop=0
            )
        self.frames.clear()
