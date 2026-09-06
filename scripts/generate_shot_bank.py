#!/usr/bin/env python3
"""Generate a statistically sufficient frozen shot bank.

``return-v0`` was hand-written: twelve development shots and twenty-four test
shots, two to four per level.  It is enough to check that the rule engine, the
report chain and a robot adapter work, and far too small to support a claim
about a policy -- a 4-episode level has a 95% Wilson interval roughly 60 points
wide.  ``BENCHMARK_SPEC`` section 8 asks for at least 100 episodes per test
condition and for non-overlapping train, dev and test seeds.

This script produces such a bank.  Two properties matter more than the count:

* **Every shot is verified by simulation, not by arithmetic.**  A candidate is
  launched in the real MuJoCo scene and kept only if the judge calls it a legal
  incoming ball.  Drag and Magnus make the difference between a ballistic
  estimate and the truth large enough that a hand-derived bank would contain
  shots no policy could legally return.
* **Tags come from what the ball did**, not from what was sampled.  ``short``,
  ``deep`` and ``edge`` are read off the measured first bounce, so a bucket
  means the same thing in every split.

``return-v0`` is not touched.  It stays frozen as the regression fixture for
the rule engine, and its digests keep every score ever produced on it valid.

Usage::

    python scripts/generate_shot_bank.py --out src/multisport_sim/benchmark/data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np

from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
from multisport_sim.benchmark.rules.table_tennis import TableTennisReturnJudge
from multisport_sim.benchmark.rules.tennis import TennisReturnJudge
from multisport_sim.benchmark.types import ShotSpec

SEED_NAMESPACE = "multisport-shot-bank-v1"

JUDGES = {
    "table_tennis": TableTennisReturnJudge,
    "tennis": TennisReturnJudge,
}


@dataclass(frozen=True)
class TagThresholds:
    """Where one sport draws the line between its own tag vocabulary.

    The tags mean the same thing in every sport -- "fast", "deep", "near the
    sideline" -- but the numbers behind them differ by an order of magnitude
    between a 2.7 g ball on a 2.74 m table and a 57 g ball on a 23.77 m court.
    """

    slow_below_mps: float
    fast_above_mps: float
    center_within_m: float
    no_spin_below_radps: float
    low_spin_below_radps: float
    sidespin_above_radps: float
    deep_below_x_m: float
    short_above_x_m: float
    edge_beyond_y_m: float
    low_below_z_m: float


@dataclass(frozen=True)
class BankPlan:
    """Everything needed to generate one sport's bank."""

    sport: str
    task: str
    bank: str
    source_bank: str
    launch_x: float
    timeout_s: float
    thresholds: TagThresholds
    levels: dict[str, LevelPlan]
    targets: tuple[tuple[str, tuple[float, float], float], ...]
    convergence: tuple[float, float] = (0.55, 0.85)
    # The playing surface, as the manifest's coordinate_system states it.  The
    # loader validates L3 target centres against these bounds, so a sport that
    # inherited another sport's table would reject its own targets.
    surface_bounds_x: tuple[float, float] = (-1.37, 1.37)
    surface_bounds_y: tuple[float, float] = (-0.7625, 0.7625)
    surface_top_z_m: float = 0.76
    # What L5 perturbs, in this sport's units.  Copying table tennis's numbers
    # onto a court would declare a perturbation that does nothing: 6 mm of
    # position noise is meaningless for a 67 mm ball crossing 20 m at 25 m/s.
    l5_observation_noise: dict[str, float] = field(
        default_factory=lambda: {
            "ball_position_m": 0.006,
            "ball_velocity_mps": 0.15,
            "ball_spin_radps": 2.0,
            "joint_position_rad": 0.002,
            "joint_velocity_radps": 0.02,
            "camera_pixel": 4.0,
        }
    )
    l5_domain_randomization: dict[str, object] = field(
        default_factory=lambda: {
            "ball_mass_scale": [0.96, 1.04],
            "drag_scale": [0.90, 1.10],
            "camera_position_m": 0.01,
        }
    )
    l5_latency: dict[str, int] = field(
        default_factory=lambda: {"observation_steps": 2, "action_steps": 1}
    )
    l5_summary: dict[str, str] = field(
        default_factory=lambda: {
            "domain_randomization": (
                "ball mass +/-4%, air density +/-10%, camera mount sigma 10 mm"
            ),
            "observation_noise": (
                "ball position sigma 6 mm, velocity sigma 0.15 m/s, camera sigma 4 counts"
            ),
            "latency": (
                "2 control steps on observations, 1 on actions (10 ms and 5 ms at 200 Hz)"
            ),
        }
    )

# Split sizes per level.  Test carries the 100 episodes per condition the
# protocol asks for; dev is half that, which is enough to tune against without
# inviting anyone to quote it; train exists so that a learned policy has seeds
# that provably do not overlap the ones it is scored on.
SPLIT_LEVEL_COUNTS = {"train": 200, "dev": 50, "test": 100}


@dataclass(frozen=True)
class Condition:
    """One sub-population of a level, sampled independently.

    Levels whose pass criterion is a worst-bucket rate are stratified over
    exactly the buckets that criterion names: leaving a bucket to chance would
    let a bank be generated on which the level cannot be assessed at all.
    """

    name: str
    y: tuple[float, float]
    z: tuple[float, float]
    vx: tuple[float, float]
    vz: tuple[float, float]
    spin_y: tuple[float, float]
    spin_z: tuple[float, float] = (0.0, 0.0)
    # Lateral velocity.  Left unset, the ball is aimed back toward the middle
    # the way a served ball is; an edge condition sets it explicitly, because a
    # shot that converges on the centre cannot land near a sideline.
    vy: tuple[float, float] | None = None
    extra_tags: tuple[str, ...] = ()
    # Tags the *measured* shot must carry for the candidate to be accepted.
    # A level whose pass criterion is a worst-bucket rate cannot be assessed at
    # all if one of its buckets came out empty, and sampling a range that
    # usually produces a bucket is not the same as producing it.
    require_any: tuple[str, ...] = ()


@dataclass(frozen=True)
class LevelPlan:
    conditions: tuple[Condition, ...]
    tags: tuple[str, ...] = ()
    targets: bool = False
    held_out: bool = False


def _range(*values: tuple[float, float]) -> tuple[float, float]:
    return (min(item[0] for item in values), max(item[1] for item in values))


def _plan_ranges(conditions: Sequence[Condition]) -> dict[str, list[float]]:
    """The manifest ranges a level's shots must satisfy, from its conditions."""
    return {
        "lateral_y_m": list(_range(*(item.y for item in conditions))),
        "height_z_m": list(_range(*(item.z for item in conditions))),
        "speed_x_mps": list(_range(*(item.vx for item in conditions))),
        "spin_y_radps": list(_range(*(item.spin_y for item in conditions))),
    }


SPIN_TAGS = ("topspin", "underspin", "sidespin", "mixed-spin")
"""The manifest's ``spin`` bucket group; a spin condition has to land in it."""

TABLE_TENNIS_LEVELS: dict[str, LevelPlan] = {
    "L0": LevelPlan(
        conditions=(
            Condition("center", y=(-0.10, 0.10), z=(1.02, 1.12), vx=(-4.5, -3.8),
                      vz=(2.7, 3.3), spin_y=(0.0, 0.0)),
        ),
        tags=("physics",),
    ),
    "L1": LevelPlan(
        conditions=(
            Condition("center", y=(-0.12, 0.12), z=(1.00, 1.14), vx=(-4.7, -4.1),
                      vz=(2.5, 3.2), spin_y=(-8.0, 8.0)),
        ),
    ),
    "L2": LevelPlan(
        conditions=(
            Condition("forehand", y=(0.10, 0.42), z=(0.98, 1.16), vx=(-6.0, -4.9),
                      vz=(2.4, 3.2), spin_y=(0.0, 55.0)),
            Condition("backhand", y=(-0.42, -0.10), z=(0.98, 1.16), vx=(-6.0, -4.9),
                      vz=(2.4, 3.2), spin_y=(0.0, 55.0)),
        ),
    ),
    "L3": LevelPlan(
        conditions=(
            Condition("placement", y=(-0.36, 0.36), z=(0.98, 1.16), vx=(-5.8, -5.1),
                      vz=(2.4, 3.1), spin_y=(10.0, 30.0)),
        ),
        tags=("placement",),
        targets=True,
    ),
    "L4": LevelPlan(
        conditions=(
            Condition("fast", y=(-0.45, 0.45), z=(0.98, 1.12), vx=(-7.7, -6.6),
                      vz=(2.2, 3.0), spin_y=(-10.0, 30.0), require_any=("fast",)),
            Condition("spin", y=(-0.45, 0.45), z=(0.98, 1.16), vx=(-6.6, -5.9),
                      vz=(2.4, 3.1), spin_y=(-50.0, 80.0), spin_z=(-30.0, 30.0),
                      require_any=SPIN_TAGS),
            Condition("edge", y=(-0.66, -0.52), z=(0.98, 1.14), vx=(-6.9, -5.9),
                      vz=(2.3, 3.0), spin_y=(-20.0, 40.0), vy=(-0.10, 0.05),
                      require_any=("edge",)),
            Condition("edge-far", y=(0.52, 0.66), z=(0.98, 1.14), vx=(-6.9, -5.9),
                      vz=(2.3, 3.0), spin_y=(-20.0, 40.0), vy=(-0.05, 0.10),
                      require_any=("edge",)),
            Condition("short", y=(-0.35, 0.35), z=(1.02, 1.20), vx=(-6.2, -5.2),
                      vz=(1.1, 1.7), spin_y=(-30.0, 60.0), require_any=("short",)),
            Condition("deep", y=(-0.40, 0.40), z=(1.00, 1.18), vx=(-6.8, -5.9),
                      vz=(2.2, 3.3), spin_y=(-30.0, 60.0), require_any=("deep",)),
            Condition("low", y=(-0.40, 0.40), z=(0.86, 0.94), vx=(-6.8, -5.9),
                      vz=(2.2, 2.9), spin_y=(-20.0, 50.0), require_any=("low",)),
        ),
    ),
    "L5": LevelPlan(
        conditions=(
            Condition("fast", y=(-0.45, 0.45), z=(0.96, 1.12), vx=(-8.1, -7.0),
                      vz=(2.1, 2.9), spin_y=(-15.0, 40.0), require_any=("fast",)),
            Condition("spin", y=(-0.45, 0.45), z=(0.96, 1.16), vx=(-7.0, -6.2),
                      vz=(2.3, 3.0), spin_y=(-65.0, 85.0), spin_z=(-40.0, 40.0),
                      require_any=SPIN_TAGS),
            Condition("edge", y=(-0.66, -0.52), z=(0.96, 1.14), vx=(-7.2, -6.2),
                      vz=(2.2, 3.0), spin_y=(-30.0, 50.0), vy=(-0.10, 0.05),
                      require_any=("edge",)),
            Condition("edge-far", y=(0.52, 0.66), z=(0.96, 1.14), vx=(-7.2, -6.2),
                      vz=(2.2, 3.0), spin_y=(-30.0, 50.0), vy=(-0.05, 0.10),
                      require_any=("edge",)),
            Condition("short", y=(-0.35, 0.35), z=(1.02, 1.20), vx=(-6.6, -6.2),
                      vz=(1.2, 1.8), spin_y=(-40.0, 70.0), require_any=("short",)),
            Condition("deep", y=(-0.40, 0.40), z=(0.98, 1.18), vx=(-7.2, -6.2),
                      vz=(1.9, 2.9), spin_y=(-40.0, 70.0), require_any=("deep",)),
            Condition("low", y=(-0.40, 0.40), z=(0.84, 0.94), vx=(-7.2, -6.2),
                      vz=(2.1, 2.8), spin_y=(-30.0, 60.0), require_any=("low",)),
        ),
        held_out=True,
    ),
}

TABLE_TENNIS_TARGETS = (
    ("target-left", (0.78, -0.45), 0.25),
    ("target-center", (0.78, 0.0), 0.25),
    ("target-right", (0.78, 0.45), 0.25),
    ("target-deep", (1.12, 0.0), 0.20),
)


TENNIS_LEVELS: dict[str, LevelPlan] = {
    "L0": LevelPlan(
        conditions=(
            Condition("center", y=(-0.8, 0.8), z=(1.0, 1.6), vx=(-20.0, -17.0),
                      vz=(1.5, 3.5), spin_y=(0.0, 0.0)),
        ),
        tags=("physics",),
    ),
    "L1": LevelPlan(
        conditions=(
            Condition("center", y=(-1.0, 1.0), z=(1.0, 1.7), vx=(-21.0, -18.0),
                      vz=(1.5, 3.6), spin_y=(-40.0, 40.0)),
        ),
    ),
    "L2": LevelPlan(
        conditions=(
            Condition("forehand", y=(0.8, 3.0), z=(0.9, 1.8), vx=(-25.0, -20.0),
                      vz=(1.4, 3.6), spin_y=(0.0, 260.0)),
            Condition("backhand", y=(-3.0, -0.8), z=(0.9, 1.8), vx=(-25.0, -20.0),
                      vz=(1.4, 3.6), spin_y=(0.0, 260.0)),
        ),
    ),
    "L3": LevelPlan(
        conditions=(
            Condition("placement", y=(-2.6, 2.6), z=(0.9, 1.8), vx=(-24.0, -21.0),
                      vz=(1.5, 3.4), spin_y=(60.0, 200.0)),
        ),
        tags=("placement",),
        targets=True,
    ),
    "L4": LevelPlan(
        conditions=(
            Condition("fast", y=(-3.2, 3.2), z=(0.9, 1.7), vx=(-31.0, -27.0),
                      vz=(1.2, 3.0), spin_y=(-60.0, 180.0), require_any=("fast",)),
            Condition("spin", y=(-3.2, 3.2), z=(0.9, 1.8), vx=(-26.0, -23.0),
                      vz=(1.4, 3.4), spin_y=(-320.0, 420.0), spin_z=(-160.0, 160.0),
                      require_any=SPIN_TAGS),
            Condition("edge", y=(-4.0, -3.2), z=(0.9, 1.7), vx=(-27.0, -23.0),
                      vz=(1.4, 3.2), spin_y=(-120.0, 260.0), vy=(-0.6, 0.3),
                      require_any=("edge",)),
            Condition("edge-far", y=(3.2, 4.0), z=(0.9, 1.7), vx=(-27.0, -23.0),
                      vz=(1.4, 3.2), spin_y=(-120.0, 260.0), vy=(-0.3, 0.6),
                      require_any=("edge",)),
            Condition("short", y=(-2.6, 2.6), z=(1.1, 1.9), vx=(-22.0, -18.0),
                      vz=(0.2, 1.4), spin_y=(-100.0, 300.0), require_any=("short",)),
            Condition("deep", y=(-3.0, 3.0), z=(0.9, 1.8), vx=(-27.0, -23.0),
                      vz=(1.8, 3.6), spin_y=(-100.0, 300.0), require_any=("deep",)),
            Condition("low", y=(-3.0, 3.0), z=(0.45, 0.70), vx=(-26.0, -22.0),
                      vz=(1.6, 3.2), spin_y=(-80.0, 240.0), require_any=("low",)),
        ),
    ),
    "L5": LevelPlan(
        conditions=(
            Condition("fast", y=(-3.2, 3.2), z=(0.9, 1.7), vx=(-34.0, -30.0),
                      vz=(1.0, 2.8), spin_y=(-80.0, 200.0), require_any=("fast",)),
            Condition("spin", y=(-3.2, 3.2), z=(0.9, 1.8), vx=(-28.0, -25.0),
                      vz=(1.3, 3.3), spin_y=(-420.0, 520.0), spin_z=(-220.0, 220.0),
                      require_any=SPIN_TAGS),
            Condition("edge", y=(-4.05, -3.3), z=(0.9, 1.7), vx=(-29.0, -25.0),
                      vz=(1.3, 3.1), spin_y=(-160.0, 320.0), vy=(-0.6, 0.3),
                      require_any=("edge",)),
            Condition("edge-far", y=(3.3, 4.05), z=(0.9, 1.7), vx=(-29.0, -25.0),
                      vz=(1.3, 3.1), spin_y=(-160.0, 320.0), vy=(-0.3, 0.6),
                      require_any=("edge",)),
            Condition("short", y=(-2.6, 2.6), z=(1.1, 1.9), vx=(-23.0, -19.0),
                      vz=(0.2, 1.4), spin_y=(-140.0, 360.0), require_any=("short",)),
            Condition("deep", y=(-3.0, 3.0), z=(0.9, 1.8), vx=(-29.0, -25.0),
                      vz=(1.7, 3.5), spin_y=(-140.0, 360.0), require_any=("deep",)),
            Condition("low", y=(-3.0, 3.0), z=(0.42, 0.68), vx=(-28.0, -24.0),
                      vz=(1.5, 3.0), spin_y=(-120.0, 300.0), require_any=("low",)),
        ),
        held_out=True,
    ),
}

TENNIS_TARGETS = (
    ("target-left", (7.5, -2.6), 1.20),
    ("target-center", (7.5, 0.0), 1.20),
    ("target-right", (7.5, 2.6), 1.20),
    ("target-deep", (10.5, 0.0), 1.00),
)

BANKS: dict[str, BankPlan] = {
    "table_tennis": BankPlan(
        sport="table_tennis",
        task="tt-return-v1",
        bank="return-v1",
        source_bank="return-v0",
        launch_x=1.8,
        timeout_s=2.0,
        thresholds=TagThresholds(
            slow_below_mps=5.0,
            fast_above_mps=6.2,
            center_within_m=0.12,
            no_spin_below_radps=3.0,
            low_spin_below_radps=12.0,
            sidespin_above_radps=15.0,
            deep_below_x_m=-0.95,
            short_above_x_m=-0.45,
            edge_beyond_y_m=0.58,
            low_below_z_m=0.95,
        ),
        levels=TABLE_TENNIS_LEVELS,
        targets=TABLE_TENNIS_TARGETS,
    ),
    # Every threshold is the tennis equivalent of the table-tennis one: a
    # rally ball at 20-30 m/s instead of 4-8, spin in the hundreds of rad/s
    # instead of the tens, and a court whose half is 11.9 m deep instead of
    # 1.37 m.  The level *meanings* are unchanged, which is the point.
    "tennis": BankPlan(
        sport="tennis",
        task="tennis-return-v0",
        bank="return-v0",
        source_bank="return-v0",
        launch_x=11.0,
        timeout_s=3.0,
        thresholds=TagThresholds(
            slow_below_mps=20.0,
            fast_above_mps=26.0,
            center_within_m=0.9,
            no_spin_below_radps=20.0,
            low_spin_below_radps=80.0,
            sidespin_above_radps=100.0,
            deep_below_x_m=-8.0,
            short_above_x_m=-4.0,
            edge_beyond_y_m=3.5,
            low_below_z_m=0.75,
        ),
        levels=TENNIS_LEVELS,
        targets=TENNIS_TARGETS,
        surface_bounds_x=(-11.885, 11.885),
        surface_bounds_y=(-4.115, 4.115),
        surface_top_z_m=0.0,
        # Scaled to the sport: a 67 mm ball tracked from ten metres away, at
        # rally speed, with the ITF mass tolerance of roughly +/-3%.
        l5_observation_noise={
            "ball_position_m": 0.020,
            "ball_velocity_mps": 0.60,
            "ball_spin_radps": 15.0,
            "joint_position_rad": 0.002,
            "joint_velocity_radps": 0.02,
            "camera_pixel": 4.0,
        },
        l5_domain_randomization={
            "ball_mass_scale": [0.97, 1.03],
            "drag_scale": [0.90, 1.10],
            "camera_position_m": 0.02,
        },
        l5_summary={
            "domain_randomization": (
                "ball mass +/-3% (ITF tolerance), air density +/-10%, "
                "camera mount sigma 20 mm"
            ),
            "observation_noise": (
                "ball position sigma 20 mm, velocity sigma 0.6 m/s, "
                "spin sigma 15 rad/s, camera sigma 4 counts"
            ),
            "latency": (
                "2 control steps on observations, 1 on actions (10 ms and 5 ms at 200 Hz)"
            ),
        },
    ),
}


def _split_seed(plan: BankPlan, split: str) -> int:
    """Distinct, documented seeds so train, dev and test cannot overlap."""
    payload = f"{SEED_NAMESPACE}:{plan.sport}:{split}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return float(low if low == high else rng.uniform(low, high))


def _speed_tag(vx: float, limits: TagThresholds) -> str:
    speed = abs(vx)
    if speed <= limits.slow_below_mps:
        return "slow"
    return "fast" if speed > limits.fast_above_mps else "normal-speed"


def _lateral_tag(y: float, limits: TagThresholds) -> str:
    if abs(y) <= limits.center_within_m:
        return "center"
    return "forehand" if y > 0.0 else "backhand"


def _spin_tags(spin_y: float, spin_z: float, limits: TagThresholds) -> tuple[str, ...]:
    lateral = abs(spin_z) >= limits.sidespin_above_radps
    if abs(spin_y) <= limits.no_spin_below_radps:
        return ("sidespin",) if lateral else ("no-spin",)
    if lateral:
        return ("mixed-spin",)
    if abs(spin_y) <= limits.low_spin_below_radps:
        return ("low-spin",)
    return ("topspin",) if spin_y > 0.0 else ("underspin",)


def _bounce_tags(bounce: tuple[float, float], limits: TagThresholds) -> tuple[str, ...]:
    """Depth and edge come from where the ball actually landed."""
    tags: list[str] = []
    if bounce[0] <= limits.deep_below_x_m:
        tags.append("deep")
    elif bounce[0] >= limits.short_above_x_m:
        tags.append("short")
    if abs(bounce[1]) >= limits.edge_beyond_y_m:
        tags.append("edge")
    return tuple(tags)


def _simulate(
    backend: MujocoShotBackend, shot: ShotSpec, bank: BankPlan
) -> tuple[bool, tuple[float, float]] | None:
    """Launch a candidate and report whether the judge accepts it, and where it bounced.

    ``None`` means the ball never produced a legal bounce on the robot's half,
    which is the common rejection: sampled combinations overshoot or clip the
    net far more often than a hand-written bank suggests.
    """
    backend.reset()
    backend.launch_ball(shot)
    judge = JUDGES[bank.sport](timeout_s=bank.timeout_s)
    judge.reset(shot)
    bounce: tuple[float, float] | None = None
    steps = int(bank.timeout_s / backend.timestep) + 2
    for _ in range(steps):
        backend.step()
        contacts = backend.semantic_contacts()
        if bounce is None:
            for contact in contacts:
                if "table" in contact.pair and contact.position is not None:
                    bounce = (float(contact.position[0]), float(contact.position[1]))
                    break
        judge.update(
            time_s=backend.time,
            ball=backend.get_ball_state(),
            contacts=contacts,
        )
        if judge.done:
            break
    if bounce is None or not judge.result.incoming_valid:
        return None
    return True, bounce


def _candidate(
    rng: np.random.Generator,
    *,
    shot_id: str,
    level: str,
    plan: LevelPlan,
    condition: Condition,
    index: int,
    bank: BankPlan,
) -> tuple[ShotSpec, dict]:
    y = _uniform(rng, condition.y)
    z = _uniform(rng, condition.z)
    vx = _uniform(rng, condition.vx)
    vz = _uniform(rng, condition.vz)
    spin_y = _uniform(rng, condition.spin_y)
    spin_z = _uniform(rng, condition.spin_z)
    # Aim the ball back toward the middle, as a served ball is.  Without this a
    # laterally launched shot leaves the table before it reaches the robot.
    vy = (
        _uniform(rng, condition.vy)
        if condition.vy is not None
        else -y * float(rng.uniform(*bank.convergence))
    )
    record = {
        "shot_id": shot_id,
        "sport": bank.sport,
        "level": level,
        "position": [bank.launch_x, round(y, 4), round(z, 4)],
        "linear_velocity": [round(vx, 4), round(vy, 4), round(vz, 4)],
        "angular_velocity": [0.0, round(spin_y, 4), round(spin_z, 4)],
    }
    if plan.targets:
        name, centre, radius = bank.targets[index % len(bank.targets)]
        record["target"] = {"center_xy": list(centre), "radius_m": radius}
        record["_target_tag"] = name
    return ShotSpec.from_mapping({k: v for k, v in record.items() if not k.startswith("_")}), record


def _tags_for(
    record: dict,
    *,
    plan: LevelPlan,
    condition: Condition,
    bounce: tuple[float, float],
    bank: BankPlan,
) -> set[str]:
    """Everything a shot is tagged with, from what was sampled and what happened."""
    limits = bank.thresholds
    tags = {
        _speed_tag(record["linear_velocity"][0], limits),
        _lateral_tag(record["position"][1], limits),
        *_spin_tags(record["angular_velocity"][1], record["angular_velocity"][2], limits),
        *_bounce_tags(bounce, limits),
        *plan.tags,
        *condition.extra_tags,
    }
    if record["position"][2] <= limits.low_below_z_m:
        tags.add("low")
    if plan.held_out:
        tags.add("held-out")
    target_tag = record.get("_target_tag")
    if target_tag is not None:
        tags.add(target_tag)
    return tags


def _generate_split(
    backend: MujocoShotBackend,
    split: str,
    *,
    per_level: int,
    bank: BankPlan,
    attempts: int = 200,
) -> Iterator[dict]:
    rng = np.random.default_rng(_split_seed(bank, split))
    for level, plan in bank.levels.items():
        conditions = plan.conditions
        accepted = 0
        index = 0
        while accepted < per_level:
            condition = conditions[index % len(conditions)]
            shot_id = f"{bank.task}-{split}-{level.lower()}-{accepted + 1:04d}"
            found = None
            for _ in range(attempts):
                shot, record = _candidate(
                    rng,
                    shot_id=shot_id,
                    level=level,
                    plan=plan,
                    condition=condition,
                    index=index,
                    bank=bank,
                )
                outcome = _simulate(backend, shot, bank)
                if outcome is None:
                    continue
                tags = _tags_for(
                    record, plan=plan, condition=condition, bounce=outcome[1], bank=bank
                )
                if condition.require_any and not set(condition.require_any) & tags:
                    continue
                record.pop("_target_tag", None)
                record["tags"] = sorted(tags)
                found = record
                break
            if found is None:
                raise SystemExit(
                    f"could not find a legal {level} shot for condition "
                    f"{condition.name!r} after {attempts} attempts"
                )
            yield found
            accepted += 1
            index += 1


def _manifest(
    bank: BankPlan, digests: dict[str, str], counts: dict[str, dict[str, int]], *, root: Path
) -> dict:
    """Start from the frozen v0 manifest so levels and thresholds cannot drift.

    Every field that describes *the task* -- coordinate system, level names,
    primary metrics, pass thresholds, bucket groups, failure reasons -- is
    inherited verbatim.  Only what describes *this bank* is replaced.  A new
    sport that quietly redefined "L2" would not be a new sport, it would be a
    new benchmark wearing the same level names.
    """
    source = json.loads(
        (root / "table_tennis" / bank.source_bank / "manifest.json").read_text(encoding="utf-8")
    )
    manifest = dict(source)
    manifest["task"] = bank.task
    manifest["sport"] = bank.sport
    manifest["coordinate_system"] = {
        **source["coordinate_system"],
        "table_top_z_m": bank.surface_top_z_m,
        "table_bounds_m": {
            "x": list(bank.surface_bounds_x),
            "y": list(bank.surface_bounds_y),
        },
    }
    manifest["episode"] = {**source["episode"], "timeout_s": bank.timeout_s}
    manifest["status"] = "experimental"
    manifest["leaderboard_eligible"] = False
    manifest["warning"] = (
        "Statistically sufficient per level (100 test episodes) but still an "
        "experimental bank: per-bucket counts remain below 100, and no "
        "leaderboard has been opened."
    )
    manifest["difficulty_ranges"] = {
        level: _plan_ranges(plan.conditions) for level, plan in bank.levels.items()
    }
    manifest["derived_from"] = {
        "bank": f"table_tennis/{bank.source_bank}",
        "relationship": (
            "same levels, thresholds, coordinate system and judge; larger fixed "
            "sets with verified-by-simulation shots"
        ),
    }
    manifest["generation"] = {
        "script": "scripts/generate_shot_bank.py",
        "seed_namespace": SEED_NAMESPACE,
        "split_seeds": {split: _split_seed(bank, split) for split in SPLIT_LEVEL_COUNTS},
        "acceptance": "simulated in MuJoCo; kept only when the judge reports incoming_valid",
        "tag_source": "speed/lateral/spin from the sampled shot; short, deep and edge from the measured first bounce",
    }
    # L5 is the generalization level, so it is the level that actually perturbs.
    # v0 declared these three conditions and then said "not_implemented_in_v0";
    # here they are stated in the units the implementation reads.
    manifest["perturbations"] = {
        level: (
            {
                "observation_noise": dict(bank.l5_observation_noise),
                "latency": dict(bank.l5_latency),
                "domain_randomization": dict(bank.l5_domain_randomization),
            }
            if level == "L5"
            else {}
        )
        for level in bank.levels
    }
    manifest["l5_conditions"] = dict(bank.l5_summary)
    manifest["splits"] = {
        split: {
            "file": f"{split}.jsonl",
            "count": sum(counts[split].values()),
            "level_counts": counts[split],
            "sha256": digests[split],
        }
        for split in SPLIT_LEVEL_COUNTS
    }
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sport", choices=sorted(BANKS), default="table_tennis")
    parser.add_argument(
        "--out", type=Path, default=Path("src/multisport_sim/benchmark/data")
    )
    parser.add_argument(
        "--splits", nargs="+", default=list(SPLIT_LEVEL_COUNTS), choices=list(SPLIT_LEVEL_COUNTS)
    )
    args = parser.parse_args(argv)

    bank = BANKS[args.sport]
    destination = args.out / bank.sport / bank.bank
    destination.mkdir(parents=True, exist_ok=True)
    backend = MujocoShotBackend(sport=bank.sport)

    digests: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for split in args.splits:
        per_level = SPLIT_LEVEL_COUNTS[split]
        records = list(_generate_split(backend, split, per_level=per_level, bank=bank))
        lines = "".join(
            json.dumps(record, separators=(",", ":"), sort_keys=False) + "\n"
            for record in records
        )
        path = destination / f"{split}.jsonl"
        path.write_text(lines, encoding="utf-8")
        digests[split] = sha256(path.read_bytes()).hexdigest()
        counts[split] = {
            level: sum(1 for record in records if record["level"] == level)
            for level in bank.levels
        }
        print(f"{split}: {len(records)} shots -> {path}", file=sys.stderr)

    if set(args.splits) == set(SPLIT_LEVEL_COUNTS):
        manifest = _manifest(bank, digests, counts, root=args.out)
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"manifest -> {destination / 'manifest.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
