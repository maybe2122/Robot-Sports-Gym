"""Backend-neutral single-return judging for net sports.

Table tennis and tennis score a return the same way: an incoming ball arrives
from the opponent half, the robot strikes it on its own half, the ball crosses
the net, and its first landing must be inside the opponent half.  Only the
surface geometry and the sport name differ, so both share this engine; the
sport-specific modules supply the surface and the defaults.
"""

from __future__ import annotations

from math import isclose, sqrt
from typing import Iterable

from ..events import BALL, FLOOR, NET, ROBOT_RACKET, TABLE, ContactEdgeDetector
from ..types import (
    BallState,
    EpisodeResult,
    FailureReason,
    SemanticContact,
    ShotSpec,
    TargetSpec,
)
from .base import PlayingSurface, finite


class NetReturnJudge:
    """Judge L0--L5 incoming shots and single returns for one net sport.

    ``contacts`` passed to :meth:`update` are active semantic contacts for the
    current physics step. Rising-edge detection is internal and reset per shot.
    The playing surface is reported as the ``table`` semantic category in every
    sport, so one contact vocabulary covers the whole benchmark.
    """

    def __init__(
        self,
        *,
        sport: str,
        surface: PlayingSurface,
        timeout_s: float = 2.0,
        require_incoming_bounce: bool = True,
    ) -> None:
        timeout = finite(timeout_s, "timeout_s")
        if timeout <= 0.0:
            raise ValueError("timeout_s must be greater than zero")
        if not isinstance(sport, str) or not sport.strip():
            raise ValueError("sport must be a non-empty string")
        self.sport = sport
        self.timeout_s = timeout
        self.surface = surface
        self.require_incoming_bounce = bool(require_incoming_bounce)
        self._edges = ContactEdgeDetector()
        self._result: EpisodeResult | None = None
        self._target: TargetSpec | None = None
        self._done = False
        self._start_time_s = 0.0
        self._last_time_s = 0.0
        self._previous_x: float | None = None
        self._previous_ball: BallState | None = None
        self._incoming_origin_ok = False
        self._incoming_crossed_net = False
        self._incoming_first_table_seen = False
        self._hit = False
        self._hit_on_robot_side = False
        self._outgoing_net_touch = False
        self._waiting_for_racket_release = False

    @property
    def done(self) -> bool:
        return self._done

    @property
    def result(self) -> EpisodeResult:
        if self._result is None:
            raise RuntimeError("judge must be reset before reading its result")
        return self._result

    def reset(
        self,
        shot: ShotSpec | str,
        *,
        target: TargetSpec | None = None,
        start_time_s: float = 0.0,
    ) -> EpisodeResult:
        start = finite(start_time_s, "start_time_s")
        if isinstance(shot, ShotSpec):
            if shot.sport != self.sport:
                raise ValueError(f"{type(self).__name__} requires sport={self.sport!r}")
            shot_id = shot.shot_id
            level = shot.level
            tags = shot.tags
            selected_target = shot.target if target is None else target
        elif isinstance(shot, str) and shot.strip():
            shot_id = shot
            level = None
            tags = ()
            selected_target = target
        else:
            raise ValueError("shot must be a ShotSpec or non-empty shot_id")
        if selected_target is not None and not isinstance(selected_target, TargetSpec):
            raise TypeError("target must be a TargetSpec or None")

        self._result = EpisodeResult(shot_id=shot_id, level=level, tags=tags)
        self._target = selected_target
        self._done = False
        self._start_time_s = start
        self._last_time_s = start
        self._previous_x = None
        self._previous_ball = None
        self._incoming_origin_ok = False
        self._incoming_crossed_net = False
        self._incoming_first_table_seen = False
        self._hit = False
        self._hit_on_robot_side = False
        self._outgoing_net_touch = False
        self._waiting_for_racket_release = False
        self._edges.reset()
        return self._result

    def _finish(
        self, time_s: float, failure_reason: FailureReason | None = None
    ) -> EpisodeResult:
        result = self.result
        if failure_reason is not None:
            result.failure_reason = failure_reason
        result.episode_time_s = time_s - self._start_time_s
        result.validate()
        self._done = True
        return result

    def abort(
        self,
        *,
        failure_reason: FailureReason,
        time_s: float | None = None,
    ) -> EpisodeResult:
        """Terminate an adapter-owned numerical or safety failure explicitly."""
        if self._done:
            return self.result
        if self._result is None:
            raise RuntimeError("judge must be reset before abort")
        if failure_reason not in {"numerical", "safety"}:
            raise ValueError("abort failure_reason must be 'numerical' or 'safety'")
        current_time = self._last_time_s if time_s is None else finite(time_s, "time_s")
        if current_time < self._last_time_s:
            raise ValueError("time_s must be monotonic")
        self._last_time_s = current_time
        return self._finish(current_time, failure_reason)

    def _is_surface_contact(self, contact: SemanticContact) -> bool:
        surface = self.surface
        if contact.normal is not None:
            if abs(contact.normal[2]) < surface.minimum_normal_z:
                return False
        if contact.position is not None:
            if abs(contact.position[2] - surface.top_height_m) > surface.height_tolerance_m:
                return False
        return True

    def _landing_point(
        self, contact: SemanticContact, ball: BallState
    ) -> tuple[float, float]:
        if contact.position is not None:
            return contact.position[0], contact.position[1]
        return ball.position[0], ball.position[1]

    def _record_target(self, landing_xy: tuple[float, float]) -> None:
        if self._target is None:
            return
        dx = landing_xy[0] - self._target.center_xy[0]
        dy = landing_xy[1] - self._target.center_xy[1]
        error = sqrt(dx * dx + dy * dy)
        self.result.target_error_m = error
        self.result.target_hit = error <= self._target.radius_m or isclose(
            error, self._target.radius_m, rel_tol=0.0, abs_tol=1e-12
        )

    def update(
        self,
        *,
        time_s: float,
        ball: BallState,
        contacts: Iterable[SemanticContact | frozenset[str]],
    ) -> EpisodeResult:
        if self._done:
            return self.result
        if self._result is None:
            raise RuntimeError("judge must be reset before update")
        if not isinstance(ball, BallState):
            raise TypeError("ball must be a BallState")
        current_time = finite(time_s, "time_s")
        if current_time < self._last_time_s:
            raise ValueError("time_s must be monotonic")

        previous_ball = self._previous_ball
        self._previous_ball = ball

        active_contacts = tuple(contacts)
        rising = self._edges.update(active_contacts)
        racket_active = any(
            contact.involves(BALL, ROBOT_RACKET)
            if isinstance(contact, SemanticContact)
            else BALL in contact and ROBOT_RACKET in contact
            for contact in active_contacts
        )
        racket_contact = next(
            (item for item in rising if item.involves(BALL, ROBOT_RACKET)), None
        )
        table_contact = next((item for item in rising if item.involves(BALL, TABLE)), None)
        net_contact = next((item for item in rising if item.involves(BALL, NET)), None)
        floor_contact = next((item for item in rising if item.involves(BALL, FLOOR)), None)

        x = ball.position[0]
        net_x = self.surface.net_plane_x_m
        if self._previous_x is None:
            self._incoming_origin_ok = x > net_x and ball.linear_velocity[0] < 0.0
        else:
            if not self._hit and self._previous_x > net_x >= x:
                self._incoming_crossed_net = True
            if self._hit and self._previous_x <= net_x < x:
                self.result.crossed_net = True

        if net_contact is not None:
            self.result.net_touch = True
            if self._hit:
                self._outgoing_net_touch = True

        # Before a racket strike, the first surface bounce determines L0 validity.
        if not self._hit and table_contact is not None and self._is_surface_contact(table_contact):
            landing_xy = self._landing_point(table_contact, ball)
            side = self.surface.side_at(*landing_xy)
            if not self._incoming_first_table_seen:
                self._incoming_first_table_seen = True
                self.result.incoming_valid = bool(
                    self._incoming_origin_ok and self._incoming_crossed_net and side == "robot"
                )
                if self.result.level == "L0":
                    self._previous_x = x
                    self._last_time_s = current_time
                    if self.result.incoming_valid:
                        return self._finish(current_time)
                    return self._finish(current_time, "out")
                if not self.result.incoming_valid:
                    self._previous_x = x
                    self._last_time_s = current_time
                    return self._finish(current_time, "out")
            elif racket_contact is None:
                self._previous_x = x
                self._last_time_s = current_time
                return self._finish(current_time, "miss")
            table_contact = None

        if racket_contact is not None and not self._hit:
            self._hit = True
            self._hit_on_robot_side = x <= net_x
            self.result.hit = True
            self.result.contact_time_s = current_time - self._start_time_s
            incoming_ball = previous_ball if previous_ball is not None else ball
            self.result.incoming_speed_mps = sqrt(
                sum(component * component for component in incoming_ball.linear_velocity)
            )
            self.result.incoming_spin_radps = incoming_ball.angular_velocity
            self._waiting_for_racket_release = True
            if self.result.level == "L1":
                self._previous_x = x
                self._last_time_s = current_time
                return self._finish(current_time)

        if self._waiting_for_racket_release and not racket_active:
            self.result.outgoing_speed_mps = sqrt(
                sum(component * component for component in ball.linear_velocity)
            )
            self._waiting_for_racket_release = False

        if self._hit and table_contact is not None and self._is_surface_contact(table_contact):
            landing_xy = self._landing_point(table_contact, ball)
            self.result.landing_xy = landing_xy
            side = self.surface.side_at(*landing_xy)
            if side == "robot":
                self._previous_x = x
                self._last_time_s = current_time
                return self._finish(current_time, "own_side")
            if side == "opponent":
                if self._hit_on_robot_side and self.result.crossed_net:
                    self.result.valid_return = True
                    self._record_target(landing_xy)
                    self._previous_x = x
                    self._last_time_s = current_time
                    return self._finish(current_time)
                self._previous_x = x
                self._last_time_s = current_time
                return self._finish(current_time, "out")
            self._previous_x = x
            self._last_time_s = current_time
            return self._finish(current_time, "out")

        if floor_contact is not None:
            self._previous_x = x
            self._last_time_s = current_time
            if not self._hit:
                return self._finish(current_time, "miss")
            if self._outgoing_net_touch and not self.result.crossed_net:
                return self._finish(current_time, "net")
            return self._finish(current_time, "out")

        if current_time - self._start_time_s >= self.timeout_s:
            self._previous_x = x
            self._last_time_s = current_time
            return self._finish(current_time, "timeout")

        self._previous_x = x
        self._last_time_s = current_time
        return self.result
