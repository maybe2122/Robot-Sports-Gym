"""Tennis single-shot rules: regulation court geometry plus the shared engine.

The coordinate convention is the one every task in this project uses: the
playing surface is centred at the origin, its length is the x axis, the robot
occupies x < 0, the opponent occupies x > 0, and z points up.  Only two things
differ from table tennis, and both are geometry rather than logic:

* the surface is the court itself, at ground level, so a ball that lands on it
  has landed *in* -- there is no separate floor beneath it to fall to;
* the singles court is used.  The doubles alleys belong to a different game,
  and scoring a return that lands in them as legal would measure the wrong task.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...specs import COURTS, Sport
from .base import RectangularSurface
from .net_return import NetReturnJudge

SINGLES_WIDTH_M = 8.23
"""ITF singles width; the doubles court is 10.97 m and is not this task."""

NET_HEIGHT_M = 0.914
"""Height at the centre of the net, which is what a return has to clear."""


@dataclass(frozen=True)
class TennisCourtSpec(RectangularSurface):
    """Court geometry the rule engine needs, independent of render geometry."""

    length_m: float = COURTS[Sport.TENNIS].length
    width_m: float = SINGLES_WIDTH_M
    top_height_m: float = 0.0
    net_plane_x_m: float = 0.0
    # The court is the ground, so a bounce is flat and right at z = 0; the
    # tolerance only has to absorb the ball's own contact depth.
    height_tolerance_m: float = 0.08

    @property
    def net_height_m(self) -> float:
        return NET_HEIGHT_M


TENNIS = TennisCourtSpec()


class TennisReturnJudge(NetReturnJudge):
    """Judge one tennis return against the regulation singles court."""

    def __init__(
        self,
        *,
        timeout_s: float = 3.0,
        court_spec: TennisCourtSpec = TENNIS,
    ) -> None:
        super().__init__(
            sport=Sport.TENNIS.value,
            surface=court_spec,
            timeout_s=timeout_s,
        )

    @property
    def court_spec(self) -> TennisCourtSpec:
        """The surface, named the way the sport names it."""
        return self.surface  # type: ignore[return-value]


__all__ = ["NET_HEIGHT_M", "SINGLES_WIDTH_M", "TENNIS", "TennisCourtSpec", "TennisReturnJudge"]
