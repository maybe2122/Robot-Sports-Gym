"""Backend-neutral judging for robot-side launch tasks.

Serving a shuttle, kicking a ball at a goal and shooting at a basket share one
episode shape that the net-return tasks do not: the object starts on the
robot's side -- held, resting or released -- and the robot strikes it exactly
once toward a goal.  There is no incoming flight to validate.

The shared result schema is kept, and each field keeps a meaning a reader of a
return report would recognise:

* ``incoming_valid`` -- the starting state is one the task allows (at L0: the
  object, left alone, behaves as the bank says it does);
* ``hit`` -- the implement touched the object;
* ``valid_return`` -- the launch achieved the task's goal (a legal serve, a
  goal, a made basket);
* ``target_hit`` / ``target_error_m`` -- how close to the declared target, in
  whatever plane the sport measures placement.

A sport subclass supplies the geometry and three hooks; the lifecycle, the
rising-edge contact detection, the L0/L1 early exits and the timeout live here.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import isclose, sqrt

from ..events import BALL, ROBOT_RACKET, ContactEdgeDetector
from ..types import (
    BallState,
    EpisodeResult,
    FailureReason,
    SemanticContact,
    ShotSpec,
    TargetSpec,
)
from .base import finite


class LaunchJudge:
    """Score one robot-side launch; subclasses define what success means."""

    def __init__(self, *, sport: str, timeout_s: float) -> None:
        timeout = finite(timeout_s, "timeout_s")
        if timeout <= 0.0:
            raise ValueError("timeout_s must be greater than zero")
        if not isinstance(sport, str) or not sport.strip():
            raise ValueError("sport must be a non-empty string")
        self.sport = sport
        self.timeout_s = timeout
        self._edges = ContactEdgeDetector()
        self._result: EpisodeResult | None = None
        self._target: TargetSpec | None = None
        self._done = False
        self._start_time_s = 0.0
        self._last_time_s = 0.0
        self._first_update = True
        self._origin: BallState | None = None
        self._previous_ball: BallState | None = None
        self._hit = False
        self._waiting_for_release = False

    # -- protocol ---------------------------------------------------------

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
            shot_id, level, tags = shot.shot_id, shot.level, shot.tags
            selected_target = shot.target if target is None else target
        elif isinstance(shot, str) and shot.strip():
            shot_id, level, tags = shot, None, ()
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
        self._first_update = True
        self._origin = None
        self._previous_ball = None
        self._hit = False
        self._waiting_for_release = False
        self._edges.reset()
        self._reset_sport()
        return self._result

    def abort(
        self, *, failure_reason: FailureReason, time_s: float | None = None
    ) -> EpisodeResult:
        """Terminate an adapter-owned numerical or safety failure explicitly."""
        if self._done:
            return self.result
        if self._result is None:
            raise RuntimeError("judge must be reset before abort")
        if failure_reason not in {"numerical", "safety"}:
            raise ValueError("abort failure_reason must be 'numerical' or 'safety'")
        current = self._last_time_s if time_s is None else finite(time_s, "time_s")
        if current < self._last_time_s:
            raise ValueError("time_s must be monotonic")
        self._last_time_s = current
        return self._finish(current, failure_reason)

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
        current = finite(time_s, "time_s")
        if current < self._last_time_s:
            raise ValueError("time_s must be monotonic")
        self._last_time_s = current
        previous = self._previous_ball
        self._previous_ball = ball

        active = tuple(contacts)
        rising = self._edges.update(active)
        racket_active = any(
            contact.involves(BALL, ROBOT_RACKET)
            if isinstance(contact, SemanticContact)
            else BALL in contact and ROBOT_RACKET in contact
            for contact in active
        )
        racket_contact = next(
            (item for item in rising if item.involves(BALL, ROBOT_RACKET)), None
        )

        if self._first_update:
            self._first_update = False
            self._origin = ball
            origin_ok = self._origin_valid(ball)
            # At L0 validity is decided by what the object then does; at every
            # other level the starting state is all there is to validate.
            if self.result.level != "L0":
                self.result.incoming_valid = origin_ok

        if not self._hit:
            if racket_contact is not None:
                self._hit = True
                self.result.hit = True
                self.result.contact_time_s = current - self._start_time_s
                incoming = previous if previous is not None else ball
                self.result.incoming_speed_mps = _speed(incoming)
                self.result.incoming_spin_radps = incoming.angular_velocity
                self._waiting_for_release = True
                fault = self._strike_fault(racket_contact, ball)
                if fault:
                    return self._finish(current, "fault")
                if self.result.level == "L1":
                    return self._finish(current)
            else:
                verdict = self._before_strike(ball, rising, current)
                if verdict is not None:
                    return verdict
        else:
            if self._waiting_for_release and not racket_active:
                self.result.outgoing_speed_mps = _speed(ball)
                self._waiting_for_release = False
            verdict = self._after_strike(ball, previous, rising, current)
            if verdict is not None:
                return verdict

        if current - self._start_time_s >= self.timeout_s:
            return self._finish(current, "timeout")
        return self.result

    # -- sport hooks ------------------------------------------------------

    def _reset_sport(self) -> None:
        """Clear per-episode sport state."""

    def _origin_valid(self, ball: BallState) -> bool:
        raise NotImplementedError

    def _strike_fault(self, contact: SemanticContact, ball: BallState) -> bool:
        """Whether the strike itself breaks a rule; most sports have none."""
        del contact, ball
        return False

    def _before_strike(
        self, ball: BallState, rising: tuple[SemanticContact, ...], time_s: float
    ) -> EpisodeResult | None:
        raise NotImplementedError

    def _after_strike(
        self,
        ball: BallState,
        previous: BallState | None,
        rising: tuple[SemanticContact, ...],
        time_s: float,
    ) -> EpisodeResult | None:
        raise NotImplementedError

    # -- helpers for subclasses -------------------------------------------

    @property
    def origin(self) -> BallState | None:
        """The object's state at the first update of the episode."""
        return self._origin

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

    def _succeed(self, time_s: float, placement: tuple[float, float]) -> EpisodeResult:
        """Record a successful launch and score its placement against the target."""
        self.result.valid_return = True
        if self._target is not None:
            dx = placement[0] - self._target.center_xy[0]
            dy = placement[1] - self._target.center_xy[1]
            error = sqrt(dx * dx + dy * dy)
            self.result.target_error_m = error
            self.result.target_hit = error <= self._target.radius_m or isclose(
                error, self._target.radius_m, rel_tol=0.0, abs_tol=1e-12
            )
        return self._finish(time_s)


def _speed(ball: BallState) -> float:
    return sqrt(sum(component * component for component in ball.linear_velocity))


__all__ = ["LaunchJudge"]
