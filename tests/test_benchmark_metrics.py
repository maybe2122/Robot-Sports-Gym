from __future__ import annotations

import json

import pytest

from multisport_sim.benchmark.metrics import (
    MetricsError,
    aggregate_results,
    assess_level,
    build_benchmark_report,
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
