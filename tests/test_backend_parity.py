"""The backend-parity comparison, on synthetic trajectories.

The Isaac rollout needs an Isaac Sim runtime, but the comparison that turns two
rollouts into the published numbers does not, so it is pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import backend_parity  # noqa: E402

DT = 0.005


def _trajectory(*, z0: float, vz0: float, restitution: float, offset: float = 0.0):
    """A ball falling onto z=0.76, bouncing once, and rising to its apex."""
    samples, z, vz = [], z0, vz0
    for step in range(1, 200):
        vz -= 9.81 * DT
        z += vz * DT
        if z < 0.76 and vz < 0.0:
            z, vz = 0.76, -restitution * vz
        samples.append([step * DT, -0.5 * step * DT + offset, 0.0, z, -0.5, 0.0, vz])
    return samples


def _rollout(backend: str, samples, *, valid: bool = True):
    result = {"level": "L1", "incoming_valid": valid, "failure_reason": "miss"}
    return {
        "backend": backend,
        "engine": backend,
        "bank": "table_tennis/return-v1",
        "split": "dev",
        "physics_dt": 0.001,
        "control_dt": DT,
        "episodes": [{"shot_id": "tt-return-v1-dev-l1-0001", "result": result, "samples": samples}],
    }


def test_identical_rollouts_have_no_divergence() -> None:
    samples = _trajectory(z0=1.2, vz0=0.0, restitution=0.9)
    report = backend_parity.compare(
        _rollout("a", samples), _rollout("b", samples), post_window_s=0.1
    )
    summary = report["summary"]
    assert summary["shots_compared"] == 1
    assert summary["incoming_valid_agreement"] == 1.0
    assert summary["flight_divergence_before_first_bounce_m"]["max"] == 0.0
    assert summary["first_bounce_time_difference_s"]["max"] == 0.0
    assert summary["post_bounce_apex_z_difference_m"]["max"] == 0.0
    assert not report["disagreements"]


def test_a_lower_restitution_shows_up_as_a_lower_apex_not_a_flight_error() -> None:
    reference = _rollout("a", _trajectory(z0=1.2, vz0=0.0, restitution=0.9))
    candidate = _rollout("b", _trajectory(z0=1.2, vz0=0.0, restitution=0.8), valid=False)
    report = backend_parity.compare(reference, candidate, post_window_s=0.1)
    summary = report["summary"]
    assert summary["flight_divergence_before_first_bounce_m"]["max"] == 0.0
    apex = summary["post_bounce_apex_z_m"]
    assert apex["candidate"]["median"] < apex["reference"]["median"]
    # e^2 scaling of the rebound height above the table.
    drop = 1.2 - 0.76
    assert apex["reference"]["median"] - 0.76 == pytest.approx(0.81 * drop, rel=0.05)
    assert report["disagreements"][0]["reference"]["incoming_valid"] is True
    assert report["disagreements"][0]["candidate"]["incoming_valid"] is False


def test_rollouts_on_different_shots_are_refused() -> None:
    samples = _trajectory(z0=1.2, vz0=0.0, restitution=0.9)
    other = _rollout("b", samples)
    other["split"] = "test"
    with pytest.raises(SystemExit):
        backend_parity.compare(_rollout("a", samples), other, post_window_s=0.1)
