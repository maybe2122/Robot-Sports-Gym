from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json

import pytest

from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
from multisport_sim.benchmark.rules.table_tennis import TableTennisReturnJudge
from multisport_sim.benchmark.shot_bank import ShotBank, ShotBankError


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
