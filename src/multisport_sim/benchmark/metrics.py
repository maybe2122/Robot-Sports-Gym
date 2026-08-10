"""Interpretable aggregation and level assessment for Shot Skill episodes."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from math import isfinite, sqrt
from typing import Any

from .shot_bank import ShotBank
from .types import EpisodeResult

FAILURE_REASONS = (
    "miss",
    "net",
    "own_side",
    "out",
    "floor",
    "timeout",
    "safety",
    "numerical",
)
_BINARY_FIELDS = (
    "incoming_valid",
    "hit",
    "valid_return",
    "target_hit",
    "crossed_net",
    "net_touch",
)


class MetricsError(ValueError):
    """Raised when finalized episode results cannot be aggregated consistently."""


def wilson_interval(
    successes: int,
    episodes: int,
    *,
    z: float = 1.959963984540054,
) -> list[float] | None:
    """Return a two-sided Wilson interval, or ``None`` for an empty sample."""
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise MetricsError("successes must be an integer")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise MetricsError("episodes must be an integer")
    if episodes < 0 or successes < 0 or successes > episodes:
        raise MetricsError("successes and episodes must satisfy 0 <= successes <= episodes")
    if not isfinite(z) or z <= 0.0:
        raise MetricsError("z must be a positive finite number")
    if episodes == 0:
        return None
    proportion = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (proportion + z * z / (2.0 * episodes)) / denominator
    half_width = (
        z
        * sqrt(proportion * (1.0 - proportion) / episodes + z * z / (4.0 * episodes**2))
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _validated_results(results: Sequence[EpisodeResult]) -> tuple[EpisodeResult, ...]:
    values = tuple(results)
    if not values:
        raise MetricsError("at least one episode result is required")
    ids: set[str] = set()
    for result in values:
        if not isinstance(result, EpisodeResult):
            raise MetricsError("results must contain EpisodeResult instances")
        result.validate()
        if result.shot_id in ids:
            raise MetricsError(f"duplicate episode result for shot_id: {result.shot_id}")
        ids.add(result.shot_id)
        for field in _BINARY_FIELDS:
            if not isinstance(getattr(result, field), bool):
                raise MetricsError(f"{result.shot_id} field {field} must be boolean")
        if result.target_error_m is not None:
            if not isfinite(result.target_error_m) or result.target_error_m < 0.0:
                raise MetricsError(
                    f"{result.shot_id} target_error_m must be finite and non-negative"
                )
        if result.failure_reason is not None and result.failure_reason not in FAILURE_REASONS:
            raise MetricsError(f"{result.shot_id} has unknown failure_reason")
    return values


def _rate(successes: int, episodes: int) -> float:
    return successes / episodes


def _metric_block(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    episodes = len(results)
    counts = {
        field: sum(bool(getattr(result, field)) for result in results)
        for field in _BINARY_FIELDS
    }
    safety_violations = sum(result.failure_reason == "safety" for result in results)
    errors = [result.target_error_m for result in results if result.target_error_m is not None]
    rates = {
        "incoming_valid_rate": _rate(counts["incoming_valid"], episodes),
        "hit_rate": _rate(counts["hit"], episodes),
        "valid_return_rate": _rate(counts["valid_return"], episodes),
        "target_rate": _rate(counts["target_hit"], episodes),
        "crossed_net_rate": _rate(counts["crossed_net"], episodes),
        "net_touch_rate": _rate(counts["net_touch"], episodes),
        "safety_violation_rate": _rate(safety_violations, episodes),
        "mean_target_error_m": sum(errors) / len(errors) if errors else None,
    }
    confidence = {
        "incoming_valid_rate": wilson_interval(counts["incoming_valid"], episodes),
        "hit_rate": wilson_interval(counts["hit"], episodes),
        "valid_return_rate": wilson_interval(counts["valid_return"], episodes),
        "target_rate": wilson_interval(counts["target_hit"], episodes),
        "safety_violation_rate": wilson_interval(safety_violations, episodes),
    }
    return {
        "episodes": episodes,
        "counts": {
            **counts,
            "safety_violation": safety_violations,
            "target_error_samples": len(errors),
        },
        "metrics": rates,
        "confidence_intervals_95": confidence,
    }


def _matches_group(result: EpisodeResult, tags: Sequence[str]) -> bool:
    return bool(set(result.tags).intersection(tags))


def aggregate_results(
    results: Sequence[EpisodeResult],
    *,
    bucket_groups: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Aggregate finalized episodes with total-episode denominators and tag buckets.

    ``hit_rate``, ``valid_return_rate``, ``target_rate`` and the safety rate all use
    the total number of episodes. Mean target error uses only results that recorded
    a target error; its sample count is kept beside the binary counts.
    """
    values = _validated_results(results)
    summary = _metric_block(values)

    tags = sorted({tag for result in values for tag in result.tags})
    buckets = {
        tag: _metric_block(tuple(result for result in values if tag in result.tags))
        for tag in tags
    }

    groups: dict[str, Any] = {}
    for name, group_tags in sorted((bucket_groups or {}).items()):
        if isinstance(group_tags, (str, bytes)) or not group_tags:
            raise MetricsError(f"bucket group {name!r} must contain one or more tags")
        matching = tuple(result for result in values if _matches_group(result, group_tags))
        if matching:
            groups[name] = {"tags": list(group_tags), **_metric_block(matching)}

    failures = Counter(
        result.failure_reason for result in values if result.failure_reason is not None
    )
    explicit_failures = sum(failures.values())
    return {
        **summary,
        "denominators": {
            "binary_rates": "all episodes",
            "mean_target_error_m": "episodes with a recorded target_error_m",
        },
        "buckets": buckets,
        "bucket_groups": groups,
        "failures": {
            **{reason: failures.get(reason, 0) for reason in FAILURE_REASONS},
        },
        "outcomes": {
            "explicit_failures": explicit_failures,
            "without_explicit_failure_reason": len(values) - explicit_failures,
        },
    }


def assess_level(
    results: Sequence[EpisodeResult],
    manifest: Mapping[str, Any],
    *,
    level: str | None = None,
) -> dict[str, Any]:
    """Apply one manifest level's frozen pass criteria to matching results."""
    values = _validated_results(results)
    available = {result.level for result in values}
    if None in available:
        raise MetricsError("level assessment requires every result to declare its level")
    selected_level = level
    if selected_level is None:
        if len(available) != 1:
            raise MetricsError("level must be specified when results contain multiple levels")
        selected_level = next(iter(available))
    level_spec = manifest.get("levels", {}).get(selected_level)
    if not isinstance(level_spec, Mapping):
        raise MetricsError(f"manifest does not define level {selected_level!r}")
    selected = tuple(result for result in values if result.level == selected_level)
    if not selected:
        raise MetricsError(f"no results for level {selected_level!r}")

    aggregate = aggregate_results(selected, bucket_groups=manifest.get("bucket_groups", {}))
    primary_metric = level_spec.get("primary_metric")
    threshold = level_spec.get("pass_threshold")
    if not isinstance(primary_metric, str):
        raise MetricsError(f"level {selected_level} has no primary_metric")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise MetricsError(f"level {selected_level} has no numeric pass_threshold")

    missing_buckets: list[str] = []
    bucket_values: dict[str, float] = {}
    if primary_metric == "worst_bucket_valid_return_rate":
        for bucket in level_spec.get("pass_buckets", ()):
            bucket_summary = aggregate["bucket_groups"].get(bucket) or aggregate[
                "buckets"
            ].get(bucket)
            if bucket_summary is None:
                missing_buckets.append(bucket)
            else:
                bucket_values[bucket] = bucket_summary["metrics"]["valid_return_rate"]
        primary_value = (
            min(bucket_values.values()) if bucket_values and not missing_buckets else None
        )
    else:
        primary_value = aggregate["metrics"].get(primary_metric)
        if primary_value is None:
            raise MetricsError(f"unsupported primary metric for {selected_level}: {primary_metric}")

    criteria = [
        {
            "metric": primary_metric,
            "value": primary_value,
            "threshold": float(threshold),
            "passed": primary_value is not None and primary_value >= float(threshold),
        }
    ]
    aggregate_metric = level_spec.get("aggregate_metric")
    if aggregate_metric is not None:
        aggregate_threshold = level_spec.get("aggregate_threshold")
        aggregate_value = aggregate["metrics"].get(aggregate_metric)
        if aggregate_value is None or not isinstance(aggregate_threshold, (int, float)):
            raise MetricsError(f"invalid aggregate criterion for level {selected_level}")
        criteria.append(
            {
                "metric": aggregate_metric,
                "value": aggregate_value,
                "threshold": float(aggregate_threshold),
                "passed": aggregate_value >= float(aggregate_threshold),
            }
        )

    reasons = [f"missing required result bucket: {bucket}" for bucket in missing_buckets]
    reasons.extend(
        (
            f"{criterion['metric']}={criterion['value']!r} is below "
            f"threshold {criterion['threshold']}"
        )
        for criterion in criteria
        if not criterion["passed"]
    )
    passed = not missing_buckets and all(criterion["passed"] for criterion in criteria)
    if passed:
        reasons.append("all frozen pass criteria were met")
    return {
        "level": selected_level,
        "episodes": len(selected),
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "pass_threshold": float(threshold),
        "bucket_values": bucket_values,
        "missing_buckets": missing_buckets,
        "criteria": criteria,
        "passed": passed,
        "reasons": reasons,
    }


def build_benchmark_report(
    results: Sequence[EpisodeResult],
    *,
    bank: ShotBank,
    backend: Mapping[str, Any],
    robot: Mapping[str, Any],
    policy: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Build the JSON-compatible, reproducible report envelope for one bank run."""
    values = _validated_results(results)
    for result in values:
        try:
            shot = bank.get(result.shot_id)
        except KeyError as exc:
            raise MetricsError(
                f"result references a shot outside the bank: {result.shot_id}"
            ) from exc
        if result.level != shot.level or tuple(result.tags) != tuple(shot.tags):
            raise MetricsError(f"result metadata does not match shot bank for {result.shot_id}")

    manifest = bank.manifest
    aggregate = aggregate_results(values, bucket_groups=manifest.get("bucket_groups", {}))
    levels = sorted({result.level for result in values if result.level is not None})
    assessments = [
        assess_level(values, manifest, level=selected_level) for selected_level in levels
    ]
    single_level = levels[0] if len(levels) == 1 else None
    single_assessment = assessments[0] if len(assessments) == 1 else None
    report = {
        "schema_version": 1,
        "benchmark": manifest["benchmark"],
        "task": manifest["task"],
        "manifest_status": manifest["status"],
        "leaderboard_eligible": manifest["leaderboard_eligible"],
        "backend": dict(backend),
        "robot": dict(robot),
        "policy": dict(policy),
        "seed": seed,
        "level": single_level,
        "assessment": single_assessment,
        "shot_bank": {
            "split": bank.split,
            "digest": bank.digest,
            "manifest_digest": bank.manifest_digest,
            "source": bank.source,
            "status": manifest["status"],
            "leaderboard_eligible": manifest["leaderboard_eligible"],
        },
        "split": bank.split,
        "shot_bank_digest": bank.digest,
        "shot_bank_manifest_digest": bank.manifest_digest,
        "shot_bank_source": bank.source,
        "episodes": aggregate["episodes"],
        "counts": aggregate["counts"],
        "metrics": aggregate["metrics"],
        "confidence_intervals_95": aggregate["confidence_intervals_95"],
        "denominators": aggregate["denominators"],
        "buckets": aggregate["buckets"],
        "bucket_groups": aggregate["bucket_groups"],
        "failures": aggregate["failures"],
        "outcomes": aggregate["outcomes"],
        "level_assessments": assessments,
        "results": [result.to_dict() for result in values],
    }
    try:
        json.dumps(report, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise MetricsError(f"report is not strict JSON-compatible: {exc}") from exc
    return report


__all__ = [
    "FAILURE_REASONS",
    "MetricsError",
    "aggregate_results",
    "assess_level",
    "build_benchmark_report",
    "wilson_interval",
]
