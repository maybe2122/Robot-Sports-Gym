"""Backend-neutral table-tennis single-shot judging rules.

Task-local coordinates follow the existing simulator scene: the table is centred
at the origin, its length is the x axis, the robot occupies x < 0, the opponent
occupies x > 0, and z points upward.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, sqrt
from numbers import Real
from typing import Iterable

from ...specs import TABLE_TENNIS as PHYSICS_TABLE_TENNIS
from ..events import BALL, FLOOR, NET, ROBOT_RACKET, TABLE, ContactEdgeDetector
from ..types import (
    BallState,
    EpisodeResult,
    FailureReason,
    SemanticContact,
    ShotSpec,
    TargetSpec,
)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class TableTennisTableSpec:
    """Geometry needed by the rule engine, independent of render geometry."""

    length_m: float = PHYSICS_TABLE_TENNIS.table_length
    width_m: float = PHYSICS_TABLE_TENNIS.table_width
    top_height_m: float = PHYSICS_TABLE_TENNIS.table_height
    net_plane_x_m: float = PHYSICS_TABLE_TENNIS.net_plane_x
    center_y_m: float = 0.0
    tabletop_height_tolerance_m: float = 0.05
    minimum_tabletop_normal_z: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "length_m",
            "width_m",
            "top_height_m",
            "net_plane_x_m",
            "center_y_m",
            "tabletop_height_tolerance_m",
            "minimum_tabletop_normal_z",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.length_m <= 0.0 or self.width_m <= 0.0:
            raise ValueError("table length and width must be greater than zero")
        if self.tabletop_height_tolerance_m < 0.0:
            raise ValueError("tabletop_height_tolerance_m must be non-negative")
        if not 0.0 <= self.minimum_tabletop_normal_z <= 1.0:
            raise ValueError("minimum_tabletop_normal_z must be between zero and one")

    def side_at(self, x: float, y: float) -> str | None:
        """Classify a top-surface point as ``robot``, ``opponent`` or outside."""
        relative_x = x - self.net_plane_x_m
        relative_y = y - self.center_y_m
        if abs(relative_y) > self.width_m / 2.0:
            return None
        if 0.0 < relative_x <= self.length_m / 2.0:
            return "opponent"
        if -self.length_m / 2.0 <= relative_x < 0.0:
            return "robot"
        return None


TABLE_TENNIS = TableTennisTableSpec()


class TableTennisReturnJudge:
    """Judge L0--L5 table-tennis incoming shots and single returns.

    ``contacts`` passed to :meth:`update` are active semantic contacts for the
    current physics step. Rising-edge detection is internal and reset per shot.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 2.0,
        table_spec: TableTennisTableSpec = TABLE_TENNIS,
    ) -> None:
        timeout = _finite(timeout_s, "timeout_s")
        if timeout <= 0.0:
            raise ValueError("timeout_s must be greater than zero")
        self.timeout_s = timeout
        self.table_spec = table_spec
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
        start = _finite(start_time_s, "start_time_s")
        if isinstance(shot, ShotSpec):
            if shot.sport != "table_tennis":
                raise ValueError("TableTennisReturnJudge requires sport='table_tennis'")
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
        current_time = self._last_time_s if time_s is None else _finite(time_s, "time_s")
        if current_time < self._last_time_s:
            raise ValueError("time_s must be monotonic")
        self._last_time_s = current_time
        return self._finish(current_time, failure_reason)

    def _is_tabletop_contact(self, contact: SemanticContact) -> bool:
        if contact.normal is not None:
            if abs(contact.normal[2]) < self.table_spec.minimum_tabletop_normal_z:
                return False
        if contact.position is not None:
            if (
                abs(contact.position[2] - self.table_spec.top_height_m)
                > self.table_spec.tabletop_height_tolerance_m
            ):
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
        current_time = _finite(time_s, "time_s")
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
        net_x = self.table_spec.net_plane_x_m
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

        # Before a racket strike, the first tabletop bounce determines L0 validity.
        if not self._hit and table_contact is not None and self._is_tabletop_contact(table_contact):
            landing_xy = self._landing_point(table_contact, ball)
            side = self.table_spec.side_at(*landing_xy)
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

        if self._hit and table_contact is not None and self._is_tabletop_contact(table_contact):
            landing_xy = self._landing_point(table_contact, ball)
            self.result.landing_xy = landing_xy
            side = self.table_spec.side_at(*landing_xy)
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


# Short name retained for the terminology used in docs/Benchmark.md.
TableTennisJudge = TableTennisReturnJudge
