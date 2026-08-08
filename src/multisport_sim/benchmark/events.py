"""Semantic contact normalization and rising-edge detection."""

from __future__ import annotations

from collections.abc import Iterable

from .types import SemanticContact

BALL = "ball"
ROBOT_RACKET = "robot_racket"
TABLE = "table"
NET = "net"
FLOOR = "floor"


def contact_between(first: str, second: str) -> SemanticContact:
    """Build a semantic contact without backend-specific contact geometry."""
    return SemanticContact.between(first, second)


def _coerce_contact(value: SemanticContact | frozenset[str]) -> SemanticContact:
    if isinstance(value, SemanticContact):
        return value
    return SemanticContact(frozenset(value))


class ContactEdgeDetector:
    """Collapse physical samples by semantic pair and emit contact rising edges.

    ``update`` receives the complete set of contacts active in the current physics
    step. A contact must disappear for at least one update before the same semantic
    pair can produce another rising edge.
    """

    def __init__(self) -> None:
        self._previous_pairs: frozenset[frozenset[str]] = frozenset()

    @property
    def active_pairs(self) -> frozenset[frozenset[str]]:
        return self._previous_pairs

    def reset(self) -> None:
        self._previous_pairs = frozenset()

    def update(
        self,
        current: Iterable[SemanticContact | frozenset[str]],
    ) -> tuple[SemanticContact, ...]:
        by_pair: dict[frozenset[str], SemanticContact] = {}
        for raw_contact in current:
            contact = _coerce_contact(raw_contact)
            by_pair.setdefault(contact.pair, contact)

        current_pairs = frozenset(by_pair)
        rising_pairs = current_pairs - self._previous_pairs
        self._previous_pairs = current_pairs
        return tuple(
            by_pair[pair]
            for pair in sorted(rising_pairs, key=lambda item: tuple(sorted(item)))
        )
