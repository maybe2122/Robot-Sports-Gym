"""Badminton serve rules: BWF singles service geometry plus the launch engine.

The coordinate convention is every task's: the net is the plane x = 0, the
robot occupies x < 0 and faces +x, z is up.  Facing +x the server's right hand
is -y; the receiver faces -x, so the receiver's right hand is +y.  A serve from
the server's right service court (y < 0) must land in the receiver's right
service court (y > 0) -- diagonally opposite, across the centre line.

What the judge enforces, from the BWF Laws of Badminton (section 9, Service):

* the whole shuttle is below 1.15 m from the court surface at the instant it is
  struck -- checked at the contact point, which is on the cork because the
  cork is the shuttle's only collision geometry;
* the shuttle crosses the net and first lands inside the receiver's singles
  service court: beyond the short service line, not beyond the back boundary
  line, between the centre line and the singles sideline, lines included;
* a serve that touches the net and still lands in the correct court is good.

Not enforced: the server's feet, the continuous-forward-racket rule and delays,
which constrain a player's body rather than the shuttle and have no meaning for
a benchmark fixture.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...specs import COURTS, Sport
from ..events import FLOOR, NET, TABLE
from ..types import BallState, EpisodeResult, SemanticContact
from .base import RectangularSurface
from .launch import LaunchJudge

SINGLES_WIDTH_M = 5.18
"""BWF singles court width; the doubles court is 6.10 m and is not this task."""

NET_HEIGHT_M = 1.524
"""Net height at the centre of the court (1.55 m at the posts)."""

SHORT_SERVICE_LINE_M = 1.98
"""Distance of the short service line from the net."""

SERVICE_HEIGHT_LIMIT_M = 1.15
"""The whole shuttle must be below this height when struck (Law 9.1.6)."""


@dataclass(frozen=True)
class BadmintonCourtSpec(RectangularSurface):
    """Court geometry the serve rules need, independent of render geometry."""

    length_m: float = COURTS[Sport.BADMINTON].length
    width_m: float = SINGLES_WIDTH_M
    top_height_m: float = 0.0
    net_plane_x_m: float = 0.0
    height_tolerance_m: float = 0.05
    short_service_line_m: float = SHORT_SERVICE_LINE_M
    service_height_limit_m: float = SERVICE_HEIGHT_LIMIT_M

    @property
    def net_height_m(self) -> float:
        return NET_HEIGHT_M

    def in_receiving_service_court(self, x: float, y: float, *, server_y: float) -> bool:
        """Whether a landing point is inside the court a serve from ``server_y`` must reach.

        In singles the service court runs from the short service line to the
        back boundary line; the long service line for doubles does not apply.
        """
        if server_y == 0.0:
            return False
        depth = x - self.net_plane_x_m
        if not self.short_service_line_m <= depth <= self.length_m / 2.0:
            return False
        # Diagonal: the landing must be on the opposite side of the centre line
        # from the server.  The centre line itself belongs to both courts.
        lateral = (y - self.center_y_m) * (-1.0 if server_y > 0.0 else 1.0)
        return 0.0 <= lateral <= self.width_m / 2.0


BADMINTON = BadmintonCourtSpec()


class BadmintonServeJudge(LaunchJudge):
    """Judge one badminton serve against the BWF singles service rules."""

    def __init__(
        self,
        *,
        timeout_s: float = 4.0,
        court_spec: BadmintonCourtSpec = BADMINTON,
    ) -> None:
        super().__init__(sport=Sport.BADMINTON.value, timeout_s=timeout_s)
        self.court_spec = court_spec
        self._crossed = False
        self._net_touch = False

    @property
    def surface(self) -> BadmintonCourtSpec:
        """The court, under the name the net-return judges expose."""
        return self.court_spec

    def _reset_sport(self) -> None:
        self._crossed = False
        self._net_touch = False

    def _origin_valid(self, ball: BallState) -> bool:
        x, y, _ = ball.position
        court = self.court_spec
        return (
            x < court.net_plane_x_m
            and y != court.center_y_m
            and abs(y - court.center_y_m) <= court.width_m / 2.0
            and ball.linear_velocity[0] >= 0.0
        )

    def _landing(self, rising: tuple[SemanticContact, ...], ball: BallState):
        """The first contact with the ground, court or surround, if any."""
        for contact in rising:
            if contact.involves("ball", TABLE) or contact.involves("ball", FLOOR):
                if contact.position is not None:
                    return contact, (contact.position[0], contact.position[1])
                return contact, (ball.position[0], ball.position[1])
        return None

    def _strike_fault(self, contact: SemanticContact, ball: BallState) -> bool:
        height = contact.position[2] if contact.position is not None else ball.position[2]
        return height > self.court_spec.service_height_limit_m

    def _before_strike(self, ball, rising, time_s) -> EpisodeResult | None:
        landing = self._landing(rising, ball)
        if landing is None:
            return None
        _, (x, _) = landing
        if self.result.level == "L0":
            origin = self.origin
            valid = (
                origin is not None
                and self._origin_valid(origin)
                and x < self.court_spec.net_plane_x_m
            )
            self.result.incoming_valid = valid
            return self._finish(time_s, None if valid else "out")
        return self._finish(time_s, "miss")

    def _after_strike(self, ball, previous, rising, time_s) -> EpisodeResult | None:
        court = self.court_spec
        net_x = court.net_plane_x_m
        if any(contact.involves("ball", NET) for contact in rising):
            self._net_touch = True
            self.result.net_touch = True
        if previous is not None and previous.position[0] <= net_x < ball.position[0]:
            self._crossed = True
            self.result.crossed_net = True

        landing = self._landing(rising, ball)
        if landing is None:
            return None
        _, (x, y) = landing
        self.result.landing_xy = (x, y)
        if x < net_x:
            return self._finish(time_s, "net" if self._net_touch else "own_side")
        origin = self.origin
        server_y = origin.position[1] if origin is not None else 0.0
        if self._crossed and court.in_receiving_service_court(x, y, server_y=server_y):
            return self._succeed(time_s, (x, y))
        return self._finish(time_s, "out")


__all__ = [
    "BADMINTON",
    "NET_HEIGHT_M",
    "SERVICE_HEIGHT_LIMIT_M",
    "SHORT_SERVICE_LINE_M",
    "SINGLES_WIDTH_M",
    "BadmintonCourtSpec",
    "BadmintonServeJudge",
]
