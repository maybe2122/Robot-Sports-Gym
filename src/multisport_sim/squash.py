"""Rules, event judging, and statistics for the squash scoring demonstration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Player(str, Enum):
    A = "A"
    B = "B"

    @property
    def opponent(self) -> Player:
        return Player.B if self is Player.A else Player.A


class RallyEventKind(str, Enum):
    SERVE = "serve"
    RACKET_HIT = "racket_hit"
    FRONT_WALL = "front_wall"
    FLOOR_BOUNCE = "floor_bounce"
    TIN = "tin"
    OUT = "out"
    POINT = "point"


@dataclass(frozen=True)
class RallyEvent:
    """One ordered, report-friendly observation from a rally."""

    number: int
    time_s: float
    kind: RallyEventKind
    player: Player | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "number": self.number,
            "time_s": round(self.time_s, 3),
            "kind": self.kind.value,
        }
        if self.player is not None:
            result["player"] = self.player.value
        if self.detail is not None:
            result["detail"] = self.detail
        return result


@dataclass(frozen=True)
class RallyResult:
    number: int
    server: Player
    winner: Player
    shots: int
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "server": self.server.value,
            "winner": self.winner.value,
            "shots": self.shots,
            "reason": self.reason,
        }


@dataclass
class SquashMatch:
    """Point-a-rally scoring with the official 11-point, win-by-two finish."""

    server: Player = Player.A
    score_a: int = 0
    score_b: int = 0
    rallies: list[RallyResult] = field(default_factory=list)
    _active_shots: int = 0

    def begin_serve(self) -> Player:
        if self._active_shots:
            raise RuntimeError("a rally is already active")
        self._active_shots = 1
        return self.server

    def record_shot(self) -> None:
        if not self._active_shots:
            raise RuntimeError("begin_serve must be called before recording shots")
        self._active_shots += 1

    def award_point(self, winner: Player, *, reason: str) -> RallyResult:
        if not self._active_shots:
            raise RuntimeError("begin_serve must be called before awarding a point")
        if winner is Player.A:
            self.score_a += 1
        else:
            self.score_b += 1
        result = RallyResult(
            number=len(self.rallies) + 1,
            server=self.server,
            winner=winner,
            shots=self._active_shots,
            reason=reason,
        )
        self.rallies.append(result)
        self.server = winner
        self._active_shots = 0
        return result

    @property
    def game_winner(self) -> Player | None:
        if max(self.score_a, self.score_b) < 11 or abs(self.score_a - self.score_b) < 2:
            return None
        return Player.A if self.score_a > self.score_b else Player.B

    def statistics(self) -> dict[str, object]:
        completed = len(self.rallies)
        lengths = [rally.shots for rally in self.rallies]
        winners = {player.value: sum(rally.winner is player for rally in self.rallies) for player in Player}
        return {
            "score": {Player.A.value: self.score_a, Player.B.value: self.score_b},
            "server": self.server.value,
            "rallies": completed,
            "longest_rally_shots": max(lengths, default=0),
            "average_rally_shots": sum(lengths) / completed if completed else 0.0,
            "points_won": winners,
            "game_winner": self.game_winner.value if self.game_winner else None,
        }


@dataclass
class SquashRallyJudge:
    """Resolve a rally from semantic contacts observed by a simulator.

    A stroke must reach the front wall before touching the floor.  After a
    legal front-wall contact, the opponent may return the ball before its
    second floor bounce.  The demonstration controller supplies racket-hit
    events, while wall and floor events come from the simulated trajectory.
    """

    match: SquashMatch
    events: list[RallyEvent] = field(default_factory=list)
    last_striker: Player | None = None
    front_wall_reached: bool = False
    floor_bounces: int = 0
    result: RallyResult | None = None

    def _record(
        self,
        kind: RallyEventKind,
        at: float,
        *,
        player: Player | None = None,
        detail: str | None = None,
    ) -> None:
        if self.events and at < self.events[-1].time_s:
            raise ValueError("rally event times must be monotonic")
        self.events.append(RallyEvent(len(self.events) + 1, at, kind, player, detail))

    def _require_active(self) -> None:
        if self.last_striker is None:
            raise RuntimeError("begin must be called before recording rally events")
        if self.result is not None:
            raise RuntimeError("the rally is already complete")

    def _award(self, winner: Player, reason: str, at: float) -> RallyResult:
        self.result = self.match.award_point(winner, reason=reason)
        self._record(RallyEventKind.POINT, at, player=winner, detail=reason)
        return self.result

    def begin(self, *, at: float = 0.0) -> Player:
        if self.last_striker is not None:
            raise RuntimeError("the rally has already started")
        server = self.match.begin_serve()
        self.last_striker = server
        self._record(RallyEventKind.SERVE, at, player=server)
        return server

    def racket_hit(self, player: Player, *, at: float) -> RallyResult | None:
        self._require_active()
        assert self.last_striker is not None
        self._record(RallyEventKind.RACKET_HIT, at, player=player)
        if player is self.last_striker:
            return self._award(player.opponent, "double_hit", at)
        if not self.front_wall_reached:
            return self._award(player, "front_wall_not_reached", at)
        self.match.record_shot()
        self.last_striker = player
        self.front_wall_reached = False
        self.floor_bounces = 0
        return None

    def front_wall(self, *, at: float) -> None:
        self._require_active()
        self.front_wall_reached = True
        self.floor_bounces = 0
        self._record(RallyEventKind.FRONT_WALL, at, player=self.last_striker)

    def floor_bounce(self, *, at: float) -> RallyResult | None:
        self._require_active()
        assert self.last_striker is not None
        self.floor_bounces += 1
        self._record(
            RallyEventKind.FLOOR_BOUNCE,
            at,
            player=self.last_striker,
            detail=str(self.floor_bounces),
        )
        if not self.front_wall_reached:
            return self._award(self.last_striker.opponent, "floor_before_front_wall", at)
        if self.floor_bounces == 2:
            return self._award(self.last_striker, "second_bounce", at)
        return None

    def tin(self, *, at: float) -> RallyResult:
        self._require_active()
        assert self.last_striker is not None
        self._record(RallyEventKind.TIN, at, player=self.last_striker)
        return self._award(self.last_striker.opponent, "tin", at)

    def out(self, *, at: float) -> RallyResult:
        self._require_active()
        assert self.last_striker is not None
        self._record(RallyEventKind.OUT, at, player=self.last_striker)
        return self._award(self.last_striker.opponent, "out", at)
