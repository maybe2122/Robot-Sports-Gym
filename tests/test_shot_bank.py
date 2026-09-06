from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256

import pytest

from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
from multisport_sim.benchmark.rules.table_tennis import TableTennisReturnJudge
from multisport_sim.benchmark.shot_bank import VALID_LEVELS, ShotBank, ShotBankError

EXPECTED_DIGESTS = {
    "dev": "d42e257ac2d14229da32e103aa2f8b67ed294d01de296ba15ea57ac3efff67e8",
    "test": "39627b091d614ac1513a33a2746351718bc61361f46a0a4331001b3f418ae572",
}


@pytest.mark.parametrize(("split", "count"), [("dev", 12), ("test", 24)])
def test_packaged_bank_is_fixed_experimental_and_complete(split: str, count: int) -> None:
    bank = ShotBank.from_resource(split=split)

    assert len(bank) == count
    assert bank.digest == EXPECTED_DIGESTS[split]
    assert len(bank.manifest_digest) == 64
    assert bank.shot_ids == tuple(sorted(bank.shot_ids))
    assert Counter(shot.level for shot in bank) == {f"L{level}": count // 6 for level in range(6)}
    assert bank.manifest["benchmark"] == "multisport-shot-skill-v0"
    assert bank.manifest["status"] == "experimental"
    assert bank.manifest["leaderboard_eligible"] is False

    for shot in bank:
        assert shot.position[0] > 0.0
        assert shot.linear_velocity[0] < 0.0
        assert shot.tags == tuple(sorted(shot.tags))
        assert shot.target is not None if shot.level == "L3" else shot.target is None
        if shot.target is not None:
            assert 0.0 < shot.target.center_xy[0] <= 1.37
            assert abs(shot.target.center_xy[1]) <= 0.7625


def test_filter_preserves_order_and_requires_all_tags() -> None:
    bank = ShotBank.from_resource(split="test")

    fast_l5 = bank.filter(level="L5", tags=("fast", "held-out"))

    assert fast_l5.shot_ids == (
        "tt-return-v0-test-l5-0001",
        "tt-return-v0-test-l5-0002",
        "tt-return-v0-test-l5-0004",
    )
    assert len(fast_l5.digest) == 64
    assert fast_l5.digest != bank.digest
    assert fast_l5.manifest_digest == bank.manifest_digest
    assert bank.get(fast_l5.shot_ids[0]) is fast_l5[0]


def test_each_manifest_pass_bucket_has_samples_in_every_split() -> None:
    for split in ("dev", "test"):
        bank = ShotBank.from_resource(split=split)
        manifest = bank.manifest
        groups = manifest["bucket_groups"]
        for level, level_spec in manifest["levels"].items():
            level_shots = bank.filter(level=level)
            for bucket in level_spec.get("pass_buckets", ()):
                accepted_tags = set(groups.get(bucket, (bucket,)))
                assert any(accepted_tags.intersection(shot.tags) for shot in level_shots), (
                    split,
                    level,
                    bucket,
                )


def _write_tampered_bank(tmp_path, lines: list[str], *, level_counts: dict[str, int]) -> None:
    packaged = ShotBank.from_resource(split="dev")
    manifest = packaged.manifest
    selected_levels = {
        level: manifest["levels"][level]
        for level in level_counts
        if level in manifest["levels"]
    }
    manifest["levels"] = selected_levels or {"L0": manifest["levels"]["L0"]}
    payload = "\n".join(lines) + "\n"
    manifest["splits"] = {
        "dev": {
            "file": "dev.jsonl",
            "count": len(lines),
            "level_counts": level_counts,
            "sha256": sha256(payload.encode("utf-8")).hexdigest(),
        }
    }
    (tmp_path / "dev.jsonl").write_text(payload, encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_loader_rejects_duplicate_shot_ids(tmp_path) -> None:
    bank = ShotBank.from_resource(split="dev")
    line = json.dumps(bank[0].to_dict(), separators=(",", ":"))
    _write_tampered_bank(tmp_path, [line, line], level_counts={"L0": 2})

    with pytest.raises(ShotBankError, match="duplicate shot_id"):
        ShotBank.from_path(tmp_path)


def test_loader_rejects_invalid_levels_and_nan(tmp_path) -> None:
    bank = ShotBank.from_resource(split="dev")
    invalid_level = bank[0].to_dict()
    invalid_level["level"] = "L9"
    _write_tampered_bank(
        tmp_path,
        [json.dumps(invalid_level, separators=(",", ":"))],
        level_counts={"L9": 1},
    )
    with pytest.raises(ShotBankError, match="invalid level"):
        ShotBank.from_path(tmp_path)

    nan_line = json.dumps(bank[0].to_dict(), separators=(",", ":")).replace(
        '"position":[1.8', '"position":[NaN'
    )
    _write_tampered_bank(tmp_path, [nan_line], level_counts={"L0": 1})
    with pytest.raises(ShotBankError, match="non-finite JSON number"):
        ShotBank.from_path(tmp_path)


def test_loader_rejects_target_on_robot_side(tmp_path) -> None:
    bank = ShotBank.from_resource(split="dev")
    invalid_target = bank.filter(level="L3")[0].to_dict()
    invalid_target["target"]["center_xy"] = [-0.5, 0.0]
    _write_tampered_bank(
        tmp_path,
        [json.dumps(invalid_target, separators=(",", ":"))],
        level_counts={"L3": 1},
    )

    with pytest.raises(ShotBankError, match="opponent side"):
        ShotBank.from_path(tmp_path)


def test_loader_accepts_a_declared_robot_side_stationary_serve(tmp_path) -> None:
    bank = ShotBank.from_resource(split="dev")
    serve = bank[0].to_dict()
    serve["position"][0] = -1.0
    serve["linear_velocity"][0] = 0.0
    line = json.dumps(serve, separators=(",", ":"))
    _write_tampered_bank(tmp_path, [line], level_counts={"L0": 1})
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coordinate_system"]["shot_origin"] = "robot_side"
    manifest["coordinate_system"]["initial_motion"] = (
        "stationary_or_toward_opponent"
    )
    manifest["difficulty_ranges"]["L0"]["speed_x_mps"] = [0.0, 0.0]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    loaded = ShotBank.from_path(tmp_path)

    assert loaded[0].position[0] < 0.0
    assert loaded[0].linear_velocity[0] == 0.0


def test_loader_rejects_a_serve_moving_in_the_declared_wrong_direction(tmp_path) -> None:
    bank = ShotBank.from_resource(split="dev")
    serve = bank[0].to_dict()
    serve["position"][0] = -1.0
    serve["linear_velocity"][0] = -0.1
    line = json.dumps(serve, separators=(",", ":"))
    _write_tampered_bank(tmp_path, [line], level_counts={"L0": 1})
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coordinate_system"]["shot_origin"] = "robot_side"
    manifest["coordinate_system"]["initial_motion"] = (
        "stationary_or_toward_opponent"
    )
    manifest["difficulty_ranges"]["L0"]["speed_x_mps"] = [-0.1, -0.1]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ShotBankError, match="initial_motion"):
        ShotBank.from_path(tmp_path)


@pytest.mark.parametrize("split", ["dev", "test"])
def test_packaged_incoming_trajectories_are_valid_in_real_mujoco(split: str) -> None:
    bank = ShotBank.from_resource(split=split)
    backend = MujocoShotBackend()
    timeout_s = bank.manifest["episode"]["timeout_s"]

    invalid: list[str] = []
    for shot in bank:
        backend.reset()
        backend.launch_ball(shot)
        judge = TableTennisReturnJudge(timeout_s=timeout_s)
        judge.reset(shot)
        for _ in range(int(timeout_s / backend.timestep) + 2):
            backend.step()
            judge.update(
                time_s=backend.time,
                ball=backend.get_ball_state(),
                contacts=backend.semantic_contacts(),
            )
            if judge.result.incoming_valid or judge.done:
                break
        if not judge.result.incoming_valid:
            invalid.append(shot.shot_id)

    assert invalid == []


class TestStatisticallySufficientBank:
    """`return-v1` exists to satisfy BENCHMARK_SPEC section 8, so that is tested.

    A bank can be large and still useless: if a level's worst-bucket criterion
    names a bucket the bank never filled, that level cannot be assessed at all.
    These tests check the properties the protocol actually depends on.
    """

    RESOURCE = "table_tennis/return-v1"

    def test_every_split_loads_and_verifies_its_own_digest(self) -> None:
        for split, expected in (("train", 1200), ("dev", 300), ("test", 600)):
            bank = ShotBank.from_resource(split=split, task=self.RESOURCE)
            assert len(bank) == expected
            assert len(bank.digest) == 64

    def test_the_test_split_carries_a_hundred_episodes_per_level(self) -> None:
        bank = ShotBank.from_resource(split="test", task=self.RESOURCE)

        for level in VALID_LEVELS:
            assert len(bank.filter(level=level)) >= 100, level

    def test_train_dev_and_test_shots_are_disjoint(self) -> None:
        """Non-overlapping seeds are worth nothing if the shots coincide."""
        seen: dict[str, set[tuple]] = {}
        for split in ("train", "dev", "test"):
            bank = ShotBank.from_resource(split=split, task=self.RESOURCE)
            seen[split] = {
                (shot.position, shot.linear_velocity, shot.angular_velocity) for shot in bank
            }
            assert len({shot.shot_id for shot in bank}) == len(bank)

        assert not seen["train"] & seen["dev"]
        assert not seen["train"] & seen["test"]
        assert not seen["dev"] & seen["test"]

    def test_every_bucket_a_pass_criterion_names_is_actually_populated(self) -> None:
        bank = ShotBank.from_resource(split="test", task=self.RESOURCE)
        manifest = bank.manifest
        groups = manifest["bucket_groups"]

        for level, spec in manifest["levels"].items():
            required = spec.get("pass_buckets", ())
            if not required:
                continue
            tags = {tag for shot in bank.filter(level=level) for tag in shot.tags}
            for bucket in required:
                members = set(groups.get(bucket, (bucket,)))
                assert tags & members, f"{level} has no shots in bucket {bucket}"

    def test_the_manifest_records_how_the_bank_was_made(self) -> None:
        """A generated bank has to say what generated it, or it cannot be redone."""
        manifest = ShotBank.from_resource(split="dev", task=self.RESOURCE).manifest

        generation = manifest["generation"]
        assert generation["script"] == "scripts/generate_shot_bank.py"
        assert len(set(generation["split_seeds"].values())) == 3
        assert "incoming_valid" in generation["acceptance"]
        assert manifest["derived_from"]["bank"] == "table_tennis/return-v0"
        # Larger, but still not a leaderboard.
        assert manifest["leaderboard_eligible"] is False

    def test_the_frozen_v0_bank_is_untouched(self) -> None:
        """Every score ever produced on v0 stays valid, so its digests are pinned."""
        dev = ShotBank.from_resource(split="dev")
        test = ShotBank.from_resource(split="test")

        assert dev.digest == "d42e257ac2d14229da32e103aa2f8b67ed294d01de296ba15ea57ac3efff67e8"
        assert test.digest == "39627b091d614ac1513a33a2746351718bc61361f46a0a4331001b3f418ae572"
        assert len(dev) == 12
        assert len(test) == 24
