"""Strict, deterministic loading for versioned single-shot benchmark banks."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from importlib import resources
from math import isfinite
from pathlib import Path
from typing import Any, overload

from .types import ShotSpec

VALID_LEVELS = tuple(f"L{index}" for index in range(6))


class ShotBankError(ValueError):
    """Raised when a shot bank or its manifest violates the on-disk contract."""


def _reject_json_constant(value: str) -> None:
    raise ShotBankError(f"non-finite JSON number is not allowed: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ShotBankError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _strict_json_loads(text: str, *, source: str) -> Any:
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except ShotBankError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ShotBankError(f"invalid JSON in {source}: {exc}") from exc
    _require_finite_numbers(value, source=source)
    return value


def _require_finite_numbers(value: Any, *, source: str) -> None:
    if isinstance(value, float) and not isfinite(value):
        raise ShotBankError(f"non-finite number in {source}")
    if isinstance(value, Mapping):
        for child in value.values():
            _require_finite_numbers(child, source=source)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _require_finite_numbers(child, source=source)


def _require_vector(record: Mapping[str, Any], field: str, size: int) -> None:
    value = record.get(field)
    if not isinstance(value, list) or len(value) != size:
        raise ShotBankError(f"{field!r} must be a JSON array of length {size}")
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ShotBankError(f"{field!r} components must be finite numbers")
        if not isfinite(float(component)):
            raise ShotBankError(f"{field!r} components must be finite numbers")


def _validate_target(target: Any, *, shot_id: str) -> None:
    if not isinstance(target, Mapping):
        raise ShotBankError(f"shot {shot_id!r} target must be an object")
    allowed = {"center_xy", "radius_m"}
    unknown = set(target) - allowed
    if unknown:
        raise ShotBankError(f"shot {shot_id!r} target has unknown fields: {sorted(unknown)}")
    _require_vector(target, "center_xy", 2)
    radius = target.get("radius_m")
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise ShotBankError(f"shot {shot_id!r} target.radius_m must be a finite number")
    if not isfinite(float(radius)) or float(radius) <= 0.0:
        raise ShotBankError(f"shot {shot_id!r} target.radius_m must be positive")


def _validate_launch(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Validate which side a shot starts on and its initial x motion.

    The original return banks predate these manifest fields and retain their
    exact opponent-to-robot convention. Serve and shoot tasks start on the
    robot side, so new banks declare both fields instead of bypassing launch
    validation altogether.
    """
    coordinates = manifest.get("coordinate_system", {})
    shot_id = record["shot_id"]
    x = float(record["position"][0])
    vx = float(record["linear_velocity"][0])
    net_x = float(coordinates.get("net_plane_x_m", 0.0))
    origin = coordinates.get("shot_origin")
    motion = coordinates.get("initial_motion")

    if origin is None and motion is None:
        if coordinates.get("opponent_side") == "x > 0" and (x <= net_x or vx >= 0.0):
            raise ShotBankError(
                f"shot {shot_id!r} must launch from opponent x>{net_x:g} "
                "toward robot with vx<0"
            )
        return
    if origin not in {"robot_side", "opponent_side"}:
        raise ShotBankError(
            "coordinate_system.shot_origin must be 'robot_side' or 'opponent_side'"
        )
    allowed_motion = {
        "toward_robot",
        "toward_opponent",
        "stationary_or_toward_robot",
        "stationary_or_toward_opponent",
    }
    if motion not in allowed_motion:
        raise ShotBankError(
            "coordinate_system.initial_motion must name a supported x direction"
        )

    on_robot_side = x < net_x
    if (origin == "robot_side") != on_robot_side or x == net_x:
        side = "robot" if origin == "robot_side" else "opponent"
        raise ShotBankError(f"shot {shot_id!r} must start on the {side} side")
    motion_valid = {
        "toward_robot": vx < 0.0,
        "toward_opponent": vx > 0.0,
        "stationary_or_toward_robot": vx <= 0.0,
        "stationary_or_toward_opponent": vx >= 0.0,
    }[motion]
    if not motion_valid:
        raise ShotBankError(f"shot {shot_id!r} violates initial_motion={motion!r}")


def _validate_record(
    record: Any,
    *,
    line_number: int,
    source: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ShotBankError(f"{source}:{line_number} must contain a JSON object")
    allowed = {
        "shot_id",
        "sport",
        "level",
        "position",
        "linear_velocity",
        "angular_velocity",
        "tags",
        "target",
    }
    missing = allowed - {"target"} - set(record)
    unknown = set(record) - allowed
    if missing:
        raise ShotBankError(f"{source}:{line_number} is missing fields: {sorted(missing)}")
    if unknown:
        raise ShotBankError(f"{source}:{line_number} has unknown fields: {sorted(unknown)}")

    shot_id = record.get("shot_id")
    if not isinstance(shot_id, str) or not shot_id.strip():
        raise ShotBankError(f"{source}:{line_number} shot_id must be a non-empty string")
    if shot_id != shot_id.strip():
        raise ShotBankError(f"shot_id cannot contain leading or trailing whitespace: {shot_id!r}")

    expected_sport = manifest.get("sport")
    if record.get("sport") != expected_sport:
        raise ShotBankError(
            f"shot {shot_id!r} sport {record.get('sport')!r} does not match "
            f"manifest sport {expected_sport!r}"
        )

    level = record.get("level")
    manifest_levels = manifest.get("levels", {})
    if level not in VALID_LEVELS or level not in manifest_levels:
        raise ShotBankError(f"shot {shot_id!r} has invalid level: {level!r}")

    _require_vector(record, "position", 3)
    _require_vector(record, "linear_velocity", 3)
    _require_vector(record, "angular_velocity", 3)

    tags = record.get("tags")
    if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag for tag in tags):
        raise ShotBankError(f"shot {shot_id!r} tags must be non-empty strings")
    if len(tags) != len(set(tags)):
        raise ShotBankError(f"shot {shot_id!r} contains duplicate tags")
    if tags != sorted(tags):
        raise ShotBankError(f"shot {shot_id!r} tags must be lexicographically ordered")

    target = record.get("target")
    if level == "L3" and target is None:
        raise ShotBankError(f"L3 shot {shot_id!r} must define a target")
    if target is not None:
        _validate_target(target, shot_id=shot_id)
        bounds = manifest.get("coordinate_system", {}).get("table_bounds_m", {})
        if isinstance(bounds, Mapping) and "x" in bounds and "y" in bounds:
            center_x, center_y = (float(item) for item in target["center_xy"])
            if not (bounds["x"][0] <= center_x <= bounds["x"][1]):
                raise ShotBankError(f"shot {shot_id!r} target center is outside table x bounds")
            if not (bounds["y"][0] <= center_y <= bounds["y"][1]):
                raise ShotBankError(f"shot {shot_id!r} target center is outside table y bounds")
            if (
                manifest.get("coordinate_system", {}).get("opponent_side") == "x > 0"
                and center_x <= 0.0
            ):
                raise ShotBankError(
                    f"shot {shot_id!r} target center must be on opponent side x>0"
                )

    _validate_launch(record, manifest)

    difficulty = manifest.get("difficulty_ranges", {}).get(level, {})
    for field, actual in (
        ("lateral_y_m", float(record["position"][1])),
        # A bank may declare a launch-height range; v0 does not, and a range
        # the manifest never states is not silently invented here.
        ("height_z_m", float(record["position"][2])),
        ("speed_x_mps", float(record["linear_velocity"][0])),
        ("spin_y_radps", float(record["angular_velocity"][1])),
    ):
        limits = difficulty.get(field) if isinstance(difficulty, Mapping) else None
        if limits is not None and not (limits[0] <= actual <= limits[1]):
            raise ShotBankError(
                f"shot {shot_id!r} {field}={actual} is outside manifest range {limits}"
            )

    return record


def _parse_manifest(text: str, *, source: str) -> dict[str, Any]:
    manifest = _strict_json_loads(text, source=source)
    if not isinstance(manifest, dict):
        raise ShotBankError(f"{source} must contain a JSON object")
    required = {
        "schema_version",
        "benchmark",
        "task",
        "sport",
        "status",
        "leaderboard_eligible",
        "coordinate_system",
        "episode",
        "levels",
        "ordering",
        "digest",
        "splits",
    }
    missing = required - set(manifest)
    if missing:
        raise ShotBankError(f"manifest is missing fields: {sorted(missing)}")
    if manifest["schema_version"] != 1:
        raise ShotBankError(f"unsupported shot-bank schema version: {manifest['schema_version']!r}")
    if manifest["ordering"] != "shot_id_lexicographic":
        raise ShotBankError(f"unsupported shot ordering: {manifest['ordering']!r}")
    digest = manifest["digest"]
    if not isinstance(digest, Mapping) or digest.get("algorithm") != "sha256":
        raise ShotBankError("manifest digest.algorithm must be 'sha256'")
    levels = manifest["levels"]
    if not isinstance(levels, Mapping) or not levels:
        raise ShotBankError("manifest levels must be a non-empty object")
    invalid_levels = set(levels) - set(VALID_LEVELS)
    if invalid_levels:
        raise ShotBankError(f"manifest contains invalid levels: {sorted(invalid_levels)}")
    if not isinstance(manifest["splits"], Mapping) or not manifest["splits"]:
        raise ShotBankError("manifest splits must be a non-empty object")
    return manifest


def _parse_jsonl(
    data: bytes,
    *,
    source: str,
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ShotBankError(f"{source} is not valid UTF-8: {exc}") from exc
    if not text.endswith("\n"):
        raise ShotBankError(f"{source} must end with a newline")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ShotBankError(f"{source}:{line_number} is blank")
        raw = _strict_json_loads(line, source=f"{source}:{line_number}")
        records.append(
            _validate_record(raw, line_number=line_number, source=source, manifest=manifest)
        )

    ids = [record["shot_id"] for record in records]
    duplicates = sorted(shot_id for shot_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ShotBankError(f"duplicate shot_id values: {duplicates}")
    if ids != sorted(ids):
        raise ShotBankError("shot records must be ordered lexicographically by shot_id")
    return records, sha256(data).hexdigest()


def _shot_from_mapping(record: Mapping[str, Any]) -> ShotSpec:
    for method_name in ("from_mapping", "from_dict"):
        factory = getattr(ShotSpec, method_name, None)
        if callable(factory):
            try:
                return factory(record)
            except (TypeError, ValueError) as exc:
                raise ShotBankError(f"invalid ShotSpec {record.get('shot_id')!r}: {exc}") from exc
    try:
        return ShotSpec(**record)
    except (TypeError, ValueError) as exc:
        raise ShotBankError(f"invalid ShotSpec {record.get('shot_id')!r}: {exc}") from exc


def _canonical_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = b"".join(
        (
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )
    return sha256(payload).hexdigest()


def _validate_pass_bucket_coverage(
    records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], *, split: str
) -> None:
    groups = manifest.get("bucket_groups", {})
    for level, level_spec in manifest["levels"].items():
        pass_buckets = level_spec.get("pass_buckets", ())
        if not pass_buckets:
            continue
        level_records = [record for record in records if record["level"] == level]
        missing: list[str] = []
        for bucket in pass_buckets:
            group_tags = groups.get(bucket, (bucket,)) if isinstance(groups, Mapping) else (bucket,)
            if not any(set(record["tags"]).intersection(group_tags) for record in level_records):
                missing.append(bucket)
        if missing:
            raise ShotBankError(
                f"split {split!r} level {level} has no sample for pass buckets: {missing}"
            )


class ShotBank(Sequence[ShotSpec]):
    """An immutable, ordered collection of validated :class:`ShotSpec` objects."""

    def __init__(
        self,
        shots: Sequence[ShotSpec],
        records: Sequence[Mapping[str, Any]],
        *,
        manifest: Mapping[str, Any],
        manifest_digest: str,
        split: str,
        digest: str,
        source: str,
    ) -> None:
        self._shots = tuple(shots)
        self._records = tuple(deepcopy(dict(record)) for record in records)
        self._manifest = deepcopy(dict(manifest))
        self._manifest_digest = manifest_digest
        self._split = split
        self._digest = digest
        self._source = source
        self._by_id = {shot.shot_id: shot for shot in self._shots}

    @classmethod
    def _from_payload(
        cls,
        manifest_text: str,
        data: bytes,
        *,
        split: str,
        manifest_source: str,
        data_source: str,
        verify_digest: bool,
    ) -> ShotBank:
        manifest = _parse_manifest(manifest_text, source=manifest_source)
        split_spec = manifest["splits"].get(split)
        if not isinstance(split_spec, Mapping):
            expected = sorted(manifest["splits"])
            raise ShotBankError(f"unknown split {split!r}; expected one of {expected}")
        records, digest = _parse_jsonl(data, source=data_source, manifest=manifest)

        expected_count = split_spec.get("count")
        if isinstance(expected_count, bool) or not isinstance(expected_count, int):
            raise ShotBankError(f"manifest split {split!r} count must be an integer")
        if len(records) != expected_count:
            raise ShotBankError(
                f"split {split!r} contains {len(records)} shots; manifest declares {expected_count}"
            )

        expected_level_counts = split_spec.get("level_counts")
        actual_level_counts = Counter(record["level"] for record in records)
        if not isinstance(expected_level_counts, Mapping):
            raise ShotBankError(f"manifest split {split!r} must define level_counts")
        if dict(actual_level_counts) != dict(expected_level_counts):
            raise ShotBankError(
                f"split {split!r} level counts {dict(actual_level_counts)} do not match "
                f"manifest {dict(expected_level_counts)}"
            )
        _validate_pass_bucket_coverage(records, manifest, split=split)

        expected_digest = split_spec.get("sha256")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ShotBankError(f"manifest split {split!r} has an invalid sha256 digest")
        if verify_digest and digest != expected_digest:
            raise ShotBankError(
                f"split {split!r} sha256 mismatch: expected {expected_digest}, got {digest}"
            )

        shots = [_shot_from_mapping(record) for record in records]
        return cls(
            shots,
            records,
            manifest=manifest,
            manifest_digest=sha256(manifest_text.encode("utf-8")).hexdigest(),
            split=split,
            digest=digest,
            source=data_source,
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        split: str | None = None,
        verify_digest: bool = True,
    ) -> ShotBank:
        """Load a split from a bank directory, manifest, or JSONL path."""
        requested = Path(path).expanduser()
        if requested.is_dir():
            manifest_path = requested / "manifest.json"
            selected_split = split or "dev"
        elif requested.name == "manifest.json":
            manifest_path = requested
            selected_split = split or "dev"
        elif requested.suffix == ".jsonl":
            manifest_path = requested.parent / "manifest.json"
            selected_split = split or requested.stem
        else:
            raise ShotBankError(
                f"shot-bank path must be a directory, manifest.json, or JSONL: {path}"
            )

        try:
            manifest_text = manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ShotBankError(f"cannot read shot-bank manifest {manifest_path}: {exc}") from exc
        manifest = _parse_manifest(manifest_text, source=str(manifest_path))
        split_spec = manifest["splits"].get(selected_split)
        if not isinstance(split_spec, Mapping):
            raise ShotBankError(
                f"unknown split {selected_split!r}; expected one of {sorted(manifest['splits'])}"
            )
        filename = split_spec.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ShotBankError(f"manifest split {selected_split!r} has an unsafe file name")
        data_path = manifest_path.parent / filename
        if requested.suffix == ".jsonl" and requested.resolve() != data_path.resolve():
            raise ShotBankError(
                f"JSONL path {requested} does not match manifest split file {data_path}"
            )
        try:
            data = data_path.read_bytes()
        except OSError as exc:
            raise ShotBankError(f"cannot read shot split {data_path}: {exc}") from exc
        return cls._from_payload(
            manifest_text,
            data,
            split=selected_split,
            manifest_source=str(manifest_path),
            data_source=str(data_path),
            verify_digest=verify_digest,
        )

    @staticmethod
    def _packaged_manifest(task: str) -> tuple[Any, str, dict[str, Any]]:
        """Resolve a packaged bank to its root, manifest text and parsed manifest."""
        parts = tuple(part for part in task.split("/") if part)
        if not parts or any(part in {".", ".."} for part in parts):
            raise ShotBankError(f"invalid packaged task path: {task!r}")
        root = resources.files("multisport_sim.benchmark").joinpath("data", *parts)
        manifest_resource = root.joinpath("manifest.json")
        try:
            manifest_text = manifest_resource.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError) as exc:
            raise ShotBankError(f"cannot read packaged manifest for {task!r}: {exc}") from exc
        manifest = _parse_manifest(manifest_text, source=str(manifest_resource))
        return root, manifest_text, manifest

    @classmethod
    def available_splits(cls, task: str = "table_tennis/return-v0") -> tuple[str, ...]:
        """Splits a packaged bank publishes, sorted.

        Callers that fit anything -- calibration constants, controller gains --
        use this to pick the training split rather than assuming one exists.
        """
        _, _, manifest = cls._packaged_manifest(task)
        return tuple(sorted(manifest["splits"]))

    @classmethod
    def from_resource(
        cls,
        *,
        split: str = "dev",
        task: str = "table_tennis/return-v0",
        verify_digest: bool = True,
    ) -> ShotBank:
        """Load a packaged benchmark split without relying on a checkout-relative path."""
        root, manifest_text, manifest = cls._packaged_manifest(task)
        manifest_resource = root.joinpath("manifest.json")
        split_spec = manifest["splits"].get(split)
        if not isinstance(split_spec, Mapping):
            expected = sorted(manifest["splits"])
            raise ShotBankError(f"unknown split {split!r}; expected one of {expected}")
        filename = split_spec.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ShotBankError(f"manifest split {split!r} has an unsafe file name")
        data_resource = root.joinpath(filename)
        try:
            data = data_resource.read_bytes()
        except (FileNotFoundError, OSError) as exc:
            raise ShotBankError(f"cannot read packaged split {split!r}: {exc}") from exc
        return cls._from_payload(
            manifest_text,
            data,
            split=split,
            manifest_source=str(manifest_resource),
            data_source=str(data_resource),
            verify_digest=verify_digest,
        )

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a defensive copy of the versioned task manifest."""
        return deepcopy(self._manifest)

    @property
    def split(self) -> str:
        return self._split

    @property
    def manifest_digest(self) -> str:
        """SHA-256 of the exact versioned manifest text."""
        return self._manifest_digest

    @property
    def digest(self) -> str:
        """SHA-256 of the exact source JSONL, including its record order."""
        return self._digest

    @property
    def source(self) -> str:
        return self._source

    @property
    def shot_ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    def get(self, shot_id: str) -> ShotSpec:
        try:
            return self._by_id[shot_id]
        except KeyError as exc:
            raise KeyError(f"unknown shot_id: {shot_id}") from exc

    def filter(
        self,
        *,
        level: str | None = None,
        tags: Iterable[str] | str = (),
    ) -> ShotBank:
        """Return a stable-order subset matching a level and all requested tags."""
        if level is not None and level not in VALID_LEVELS:
            raise ShotBankError(f"invalid level: {level!r}")
        required_tags = {tags} if isinstance(tags, str) else set(tags)
        if any(not isinstance(tag, str) or not tag for tag in required_tags):
            raise ShotBankError("filter tags must be non-empty strings")
        indices = [
            index
            for index, shot in enumerate(self._shots)
            if (level is None or shot.level == level) and required_tags.issubset(set(shot.tags))
        ]
        records = [self._records[index] for index in indices]
        shots = [self._shots[index] for index in indices]
        return ShotBank(
            shots,
            records,
            manifest=self._manifest,
            manifest_digest=self._manifest_digest,
            split=self._split,
            digest=_canonical_digest(records),
            source=f"{self._source} (filtered)",
        )

    def __len__(self) -> int:
        return len(self._shots)

    @overload
    def __getitem__(self, index: int) -> ShotSpec: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ShotSpec, ...]: ...

    def __getitem__(self, index: int | slice) -> ShotSpec | tuple[ShotSpec, ...]:
        return self._shots[index]

    def __iter__(self) -> Iterator[ShotSpec]:
        return iter(self._shots)


__all__ = ["VALID_LEVELS", "ShotBank", "ShotBankError"]
