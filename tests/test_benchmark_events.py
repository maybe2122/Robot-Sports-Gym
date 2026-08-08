from __future__ import annotations

import json

import pytest

from multisport_sim.benchmark.events import ContactEdgeDetector
from multisport_sim.benchmark.types import (
    BallState,
    EpisodeResult,
    SemanticContact,
    ShotSpec,
    TargetSpec,
)


def test_semantic_contact_is_order_independent_and_json_friendly() -> None:
    contact = SemanticContact.between(
        "table",
        "ball",
        position=(0.5, 0.1, 0.76),
        normal=(0.0, 0.0, 1.0),
    )

    assert contact.pair == frozenset(("ball", "table"))
    assert contact.involves("ball")
    assert contact.involves("table", "ball")
    assert not contact.involves("net")
    assert json.loads(json.dumps(contact.to_dict()))["position"] == [0.5, 0.1, 0.76]


@pytest.mark.parametrize(
    "pair",
    [frozenset(), frozenset(("ball",)), frozenset(("ball", "table", "net"))],
)
def test_semantic_contact_requires_exactly_two_participants(pair: frozenset[str]) -> None:
    with pytest.raises(ValueError, match="exactly two"):
        SemanticContact(pair)


def test_contact_detector_emits_only_real_rising_edges() -> None:
    detector = ContactEdgeDetector()
    first = SemanticContact.between("ball", "robot_racket", position=(0.0, 0.0, 1.0))
    moved = SemanticContact.between("ball", "robot_racket", position=(0.01, 0.0, 1.0))

    assert detector.update([first]) == (first,)
    assert detector.update([moved]) == ()
    assert detector.update([]) == ()
    assert detector.update([moved]) == (moved,)


def test_contact_detector_collapses_duplicate_samples_and_reset_clears_history() -> None:
    detector = ContactEdgeDetector()
    samples = [
        SemanticContact.between("ball", "table", position=(0.2, 0.0, 0.76)),
        SemanticContact.between("table", "ball", position=(0.21, 0.0, 0.76)),
    ]

    rising = detector.update(samples)
    assert len(rising) == 1
    detector.reset()
    assert len(detector.update(samples)) == 1


def test_shot_spec_mapping_round_trip_includes_l3_target() -> None:
    raw = {
        "shot_id": "tt-return-l3-000001",
        "sport": "table_tennis",
        "level": "L3",
        "position": [1.8, 0.1, 1.1],
        "linear_velocity": [-5.6, -0.35, 1.62],
        "angular_velocity": [4.0, -55.0, 2.0],
        "tags": ["topspin", "normal-speed"],
        "target": {"center_xy": [0.7, 0.0], "radius_m": 0.3},
    }

    shot = ShotSpec.from_mapping(raw)

    assert shot.target == TargetSpec((0.7, 0.0), 0.3)
    assert shot.to_dict() == raw
    assert json.loads(json.dumps(shot.to_dict())) == raw


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_public_types_reject_non_finite_values(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        BallState((bad, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        TargetSpec((0.0, 0.0), bad)
    with pytest.raises(ValueError, match="finite"):
        ShotSpec(
            "shot",
            "table_tennis",
            "L2",
            (0.0, 0.0, 1.0),
            (bad, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )


def test_episode_result_enforces_metric_implications_and_serializes_lists() -> None:
    with pytest.raises(ValueError, match="requires hit"):
        EpisodeResult("bad", valid_return=True)
    with pytest.raises(ValueError, match="requires valid_return"):
        EpisodeResult("bad", hit=True, target_hit=True)
    with pytest.raises(ValueError, match="requires valid_return"):
        EpisodeResult("bad", hit=True, target_error_m=0.1)

    result = EpisodeResult(
        "good",
        level="L3",
        tags=("center",),
        incoming_valid=True,
        hit=True,
        valid_return=True,
        target_hit=True,
        crossed_net=True,
        landing_xy=(0.6, 0.0),
        incoming_spin_radps=(1.0, 2.0, 3.0),
    )
    saved = json.loads(json.dumps(result.to_dict()))
    assert saved["landing_xy"] == [0.6, 0.0]
    assert saved["incoming_spin_radps"] == [1.0, 2.0, 3.0]
    assert saved["incoming_valid"]
    assert saved["crossed_net"]
