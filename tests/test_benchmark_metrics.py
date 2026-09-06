from __future__ import annotations

import json

import pytest

from multisport_sim.benchmark.metrics import (
    MetricsError,
    aggregate_results,
    assess_level,
    build_benchmark_report,
    robustness_gap,
    wilson_interval,
)
from multisport_sim.benchmark.shot_bank import ShotBank
from multisport_sim.benchmark.types import EpisodeResult


def test_aggregate_keeps_interpretable_rates_buckets_and_failures() -> None:
    results = [
        EpisodeResult(
            "a",
            level="L2",
            tags=("fast", "topspin"),
            incoming_valid=True,
            hit=True,
            crossed_net=True,
            valid_return=True,
            target_error_m=0.1,
        ),
        EpisodeResult(
            "b",
            level="L2",
            tags=("fast", "underspin"),
            incoming_valid=True,
            hit=True,
            failure_reason="net",
        ),
        EpisodeResult(
            "c",
            level="L2",
            tags=("slow",),
            incoming_valid=False,
            failure_reason="miss",
        ),
    ]

    summary = aggregate_results(results, bucket_groups={"spin": ("topspin", "underspin")})

    assert summary["episodes"] == 3
    assert summary["metrics"]["incoming_valid_rate"] == pytest.approx(2 / 3)
    assert summary["metrics"]["hit_rate"] == pytest.approx(2 / 3)
    assert summary["metrics"]["valid_return_rate"] == pytest.approx(1 / 3)
    assert summary["metrics"]["mean_target_error_m"] == pytest.approx(0.1)
    assert summary["buckets"]["fast"]["metrics"]["valid_return_rate"] == 0.5
    assert summary["bucket_groups"]["spin"]["episodes"] == 2
    assert summary["failures"]["net"] == 1
    assert summary["failures"]["miss"] == 1
    assert summary["outcomes"] == {
        "explicit_failures": 2,
        "without_explicit_failure_reason": 1,
    }


def test_wilson_interval_matches_documented_100_shot_example() -> None:
    interval = wilson_interval(80, 100)

    assert interval is not None
    assert interval[0] == pytest.approx(0.711, abs=0.001)
    assert interval[1] == pytest.approx(0.867, abs=0.001)
    assert wilson_interval(0, 0) is None


def test_assess_level_uses_frozen_primary_and_aggregate_thresholds() -> None:
    manifest = ShotBank.from_resource(split="test").manifest
    tags = (
        ("fast", "topspin", "held-out", "edge", "low"),
        ("fast", "underspin", "held-out", "deep"),
        ("held-out", "sidespin", "short", "low"),
        ("fast", "mixed-spin", "held-out", "edge"),
        ("fast", "topspin", "held-out", "deep"),
        ("held-out", "underspin", "short", "low"),
        ("fast", "sidespin", "held-out", "edge"),
        ("fast", "mixed-spin", "held-out", "deep"),
        ("held-out", "topspin", "short", "low"),
        ("fast", "underspin", "held-out", "edge"),
    )
    results = [
        EpisodeResult(
            f"l5-{index}",
            level="L5",
            tags=tuple(sorted(shot_tags)),
            incoming_valid=True,
            hit=index < 8,
            crossed_net=index < 6,
            valid_return=index < 6,
            failure_reason=None if index < 6 else "out",
        )
        for index, shot_tags in enumerate(tags)
    ]

    assessment = assess_level(results, manifest)

    assert assessment["primary_metric"] == "worst_bucket_valid_return_rate"
    assert assessment["primary_value"] >= 0.5
    assert assessment["worst_bucket"] in assessment["bucket_values"]
    assert assessment["bucket_episodes"][assessment["worst_bucket"]] > 0
    assert assessment["primary_confidence_interval_95"] == assessment[
        "bucket_confidence_intervals_95"
    ][assessment["worst_bucket"]]
    assert assessment["criteria"][1] == {
        "metric": "valid_return_rate",
        "value": 0.6,
        "threshold": 0.6,
        "passed": True,
    }
    assert assessment["passed"]
    assert assessment["reasons"] == ["all frozen pass criteria were met"]


def test_report_records_bank_identity_and_raw_results() -> None:
    bank = ShotBank.from_resource(split="dev").filter(level="L1")
    results = [
        EpisodeResult(
            shot.shot_id,
            level=shot.level,
            tags=shot.tags,
            incoming_valid=True,
            hit=True,
        )
        for shot in bank
    ]

    report = build_benchmark_report(
        results,
        bank=bank,
        backend={"name": "test-backend", "version": "1"},
        robot={"name": "fixture"},
        policy={"name": "scripted"},
        seed=7,
    )

    assert report["level"] == "L1"
    assert report["assessment"]["passed"]
    assert report["assessment"]["primary_confidence_interval_95"] == pytest.approx(
        wilson_interval(len(bank), len(bank))
    )
    assert report["shot_bank"] == {
        "split": "dev",
        "digest": bank.digest,
        "manifest_digest": bank.manifest_digest,
        "source": bank.source,
        "status": "experimental",
        "leaderboard_eligible": False,
    }
    assert report["backend"]["name"] == "test-backend"
    assert report["results"][0]["shot_id"] == bank[0].shot_id
    json.dumps(report, allow_nan=False)


def test_aggregate_rejects_duplicate_episode_ids() -> None:
    result = EpisodeResult("duplicate", level="L1", incoming_valid=True, hit=True)

    with pytest.raises(MetricsError, match="duplicate episode result"):
        aggregate_results([result, result])


def test_robustness_gap_compares_nominal_levels_against_perturbed_ones() -> None:
    """One blended success rate hides exactly the thing this metric is for."""
    results = [
        EpisodeResult(
            shot_id=f"nominal-{index}",
            level="L2",
            incoming_valid=True,
            hit=True,
            valid_return=index < 3,
            failure_reason=None if index < 3 else "net",
        )
        for index in range(4)
    ] + [
        EpisodeResult(
            shot_id=f"hard-{index}",
            level="L4",
            incoming_valid=True,
            hit=True,
            valid_return=index < 1,
            failure_reason=None if index < 1 else "out",
        )
        for index in range(4)
    ]

    gap = robustness_gap(results)

    assert gap is not None
    assert gap["metric"] == "valid_return_rate"
    assert gap["in_distribution"] == pytest.approx(0.75)
    assert gap["perturbed"] == pytest.approx(0.25)
    assert gap["gap"] == pytest.approx(0.5)
    assert gap["in_distribution_episodes"] == 4
    assert gap["perturbed_episodes"] == 4
    assert gap["level_breakdown"] == {
        "L2": {
            "episodes": 4,
            "value": pytest.approx(0.75),
            "confidence_interval_95": pytest.approx(wilson_interval(3, 4)),
        },
        "L4": {
            "episodes": 4,
            "value": pytest.approx(0.25),
            "confidence_interval_95": pytest.approx(wilson_interval(1, 4)),
        },
    }


def test_robustness_gap_is_absent_when_one_group_was_not_run() -> None:
    """A gap computed from one group would be a comparison with nothing."""
    nominal = [
        EpisodeResult(shot_id="a", level="L2", incoming_valid=True, hit=True, valid_return=True)
    ]

    assert robustness_gap(nominal) is None
    assert robustness_gap([]) is None
    # L0 is a physics check with no policy in the loop, so it belongs to neither.
    assert robustness_gap([EpisodeResult(shot_id="b", level="L0")]) is None


def test_mean_episode_time_averages_only_the_episodes_that_reported_one() -> None:
    results = [
        EpisodeResult(shot_id="a", level="L1", episode_time_s=1.0),
        EpisodeResult(shot_id="b", level="L1", episode_time_s=2.0),
        EpisodeResult(shot_id="c", level="L1"),
    ]

    aggregate = aggregate_results(results)

    assert aggregate["metrics"]["mean_episode_time_s"] == pytest.approx(1.5)
    assert aggregate["counts"]["episode_time_samples"] == 2


def test_an_episode_result_survives_a_round_trip_through_its_report_entry() -> None:
    """The submission packager re-reads reports, so this has to be lossless."""
    result = EpisodeResult(
        shot_id="tt-0001",
        level="L3",
        tags=("center", "topspin"),
        incoming_valid=True,
        hit=True,
        valid_return=True,
        target_hit=True,
        crossed_net=True,
        contact_time_s=0.51,
        landing_xy=(0.8, -0.2),
        target_error_m=0.11,
        incoming_speed_mps=5.4,
        outgoing_speed_mps=4.1,
        incoming_spin_radps=(0.0, 20.0, 0.0),
        episode_time_s=1.2,
    )

    assert EpisodeResult.from_dict(result.to_dict()) == result


def test_a_report_entry_with_an_unknown_field_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown result fields"):
        EpisodeResult.from_dict({"shot_id": "x", "success": True})


def test_a_report_entry_that_breaks_an_invariant_is_refused() -> None:
    """`target_hit` without `valid_return` is not a result, it is a corruption."""
    with pytest.raises(ValueError):
        EpisodeResult.from_dict({"shot_id": "x", "target_hit": True, "valid_return": False})
