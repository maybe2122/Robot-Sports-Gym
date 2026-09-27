"""Basketball shooting rules: FIBA basket geometry plus the launch engine.

The task frame's origin is the floor directly below the centre of the rim; the
shooter is at x < 0 and shoots toward +x, the backboard's face is 0.34 m
beyond the rim centre, and z is up.  A basket is scored, per FIBA Article 16,
when the ball enters the basket from above and stays in or passes through it:
the judge scores the ball's centre coming *down* through the rim's plane
inside the ring.  Rim and backboard contacts on the way are allowed -- a bank
shot counts -- and a ball that enters from below is a violation.

Placement is measured in the rim plane, as ``(x, y)`` of the ball's centre
when it passes down through it; a centred entry is the clean shot.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from ...specs import BALLS, Sport
from ..events import FLOOR, RIM, TABLE
from ..types import BallState, EpisodeResult
from .launch import LaunchJudge

RIM_HEIGHT_M = 3.05
"""Height of the top of the ring above the floor."""

RIM_INNER_RADIUS_M = 0.225
"""Half of the ring's 0.45 m inner diameter."""


@dataclass(frozen=True)
class BasketSpec:
    """Basket geometry in the task frame (origin on the floor below the rim)."""

    rim_height_m: float = RIM_HEIGHT_M
    rim_inner_radius_m: float = RIM_INNER_RADIUS_M
    ball_radius_m: float = BALLS[Sport.BASKETBALL].radius
    court_half_width_m: float = 7.5

    def inside_ring(self, x: float, y: float) -> bool:
        return hypot(x, y) < self.rim_inner_radius_m


BASKET = BasketSpec()


class BasketballShootJudge(LaunchJudge):
    """Judge one shot at the basket."""

    def __init__(self, *, timeout_s: float = 4.0, basket_spec: BasketSpec = BASKET) -> None:
        super().__init__(sport=Sport.BASKETBALL.value, timeout_s=timeout_s)
        self.basket_spec = basket_spec

    def _origin_valid(self, ball: BallState) -> bool:
        x, y, z = ball.position
        spec = self.basket_spec
        return (
            x < -spec.rim_inner_radius_m - spec.ball_radius_m
            and abs(y) <= spec.court_half_width_m
            and spec.ball_radius_m < z < spec.rim_height_m
            and ball.linear_velocity[0] >= 0.0
        )

    @staticmethod
    def _grounded(rising) -> bool:
        return any(
            contact.involves("ball", FLOOR) or contact.involves("ball", TABLE)
            for contact in rising
        )

    def _before_strike(self, ball, rising, time_s) -> EpisodeResult | None:
        if not self._grounded(rising):
            return None
        if self.result.level == "L0":
            origin = self.origin
            valid = (
                origin is not None
                and self._origin_valid(origin)
                and ball.position[0] < 0.0
            )
            self.result.incoming_valid = valid
            return self._finish(time_s, None if valid else "out")
        return self._finish(time_s, "miss")

    def _after_strike(self, ball, previous, rising, time_s) -> EpisodeResult | None:
        spec = self.basket_spec
        if any(contact.involves("ball", RIM) for contact in rising):
            # ``net_touch`` is the shared field for "touched the structure it
            # is aimed at"; here, the ring.
            self.result.net_touch = True
        if previous is not None:
            z_before, z_now = previous.position[2], ball.position[2]
            x, y = ball.position[0], ball.position[1]
            if z_before > spec.rim_height_m >= z_now and spec.inside_ring(x, y):
                self.result.landing_xy = (x, y)
                return self._succeed(time_s, (x, y))
            if z_before < spec.rim_height_m <= z_now and spec.inside_ring(x, y):
                # Up through the ring from below: a violation, not a basket.
                return self._finish(time_s, "fault")
        if self._grounded(rising):
            return self._finish(time_s, "out")
        return None


__all__ = [
    "BASKET",
    "RIM_HEIGHT_M",
    "RIM_INNER_RADIUS_M",
    "BasketSpec",
    "BasketballShootJudge",
]
