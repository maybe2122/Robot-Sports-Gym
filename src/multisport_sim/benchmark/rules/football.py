"""Football kick-to-target rules: IFAB goal geometry plus the launch engine.

The task frame's origin is the centre of the goal line on the ground, the
kicker is at x < 0 and kicks toward +x, z is up.  A goal is scored, per IFAB
Law 10, when the *whole* ball passes over the goal line between the posts and
under the crossbar -- so the judge waits for the ball's centre to be one
radius beyond the line, and checks where it is at that instant.

Placement is measured where a goalkeeper would measure it: in the goal mouth,
as ``(y, z)`` of the ball's centre when it crosses.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...specs import BALLS, COURTS, Sport
from ..events import POST
from ..types import BallState, EpisodeResult
from .launch import LaunchJudge

GOAL_HALF_WIDTH_M = 3.66
"""Half of the 7.32 m between the posts' inner edges."""

CROSSBAR_HEIGHT_M = 2.44
"""Height of the crossbar's lower edge above the ground."""


@dataclass(frozen=True)
class FootballGoalSpec:
    """Pitch and goal geometry in the task frame (origin at the goal line)."""

    pitch_length_m: float = COURTS[Sport.FOOTBALL].length
    pitch_width_m: float = COURTS[Sport.FOOTBALL].width
    goal_half_width_m: float = GOAL_HALF_WIDTH_M
    crossbar_height_m: float = CROSSBAR_HEIGHT_M
    ball_radius_m: float = BALLS[Sport.FOOTBALL].radius
    # How long an untouched ball is watched at L0 before it is called settled.
    settle_s: float = 1.0

    def in_goal_mouth(self, y: float, z: float) -> bool:
        return abs(y) < self.goal_half_width_m and z < self.crossbar_height_m

    def on_pitch(self, x: float, y: float) -> bool:
        return -self.pitch_length_m <= x <= 0.0 and abs(y) <= self.pitch_width_m / 2.0


FOOTBALL_GOAL = FootballGoalSpec()


class FootballKickJudge(LaunchJudge):
    """Judge one kick at the goal: a goal is the whole ball over the line."""

    def __init__(
        self,
        *,
        timeout_s: float = 3.0,
        goal_spec: FootballGoalSpec = FOOTBALL_GOAL,
    ) -> None:
        super().__init__(sport=Sport.FOOTBALL.value, timeout_s=timeout_s)
        self.goal_spec = goal_spec

    def _origin_valid(self, ball: BallState) -> bool:
        x, y, z = ball.position
        spec = self.goal_spec
        return (
            spec.on_pitch(x, y)
            and x < -spec.ball_radius_m
            and z <= spec.ball_radius_m + 0.02
            and ball.linear_velocity[0] <= 0.0
        )

    def _before_strike(self, ball, rising, time_s) -> EpisodeResult | None:
        x, y, _ = ball.position
        spec = self.goal_spec
        if self.result.level == "L0":
            if time_s - self._start_time_s >= spec.settle_s:
                origin = self.origin
                valid = (
                    origin is not None
                    and self._origin_valid(origin)
                    and spec.on_pitch(x, y)
                    and ball.position[2] <= spec.ball_radius_m + 0.05
                )
                self.result.incoming_valid = valid
                return self._finish(time_s, None if valid else "out")
            return None
        if not spec.on_pitch(x, y):
            return self._finish(time_s, "miss")
        return None

    def _after_strike(self, ball, previous, rising, time_s) -> EpisodeResult | None:
        spec = self.goal_spec
        if any(contact.involves("ball", POST) for contact in rising):
            # The woodwork: ``net_touch`` is the shared field for "touched the
            # structure it is aimed at".
            self.result.net_touch = True
        x, y, z = ball.position
        line = spec.ball_radius_m
        if previous is not None and previous.position[0] < line <= x:
            self.result.crossed_net = True
            self.result.landing_xy = (y, z)
            if spec.in_goal_mouth(y, z):
                return self._succeed(time_s, (y, z))
            return self._finish(time_s, "out")
        if abs(y) > spec.pitch_width_m / 2.0 or x < -spec.pitch_length_m:
            return self._finish(time_s, "out")
        return None


__all__ = [
    "CROSSBAR_HEIGHT_M",
    "FOOTBALL_GOAL",
    "GOAL_HALF_WIDTH_M",
    "FootballGoalSpec",
    "FootballKickJudge",
]
