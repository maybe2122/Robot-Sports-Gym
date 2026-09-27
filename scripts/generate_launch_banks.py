#!/usr/bin/env python3
"""Generate frozen shot banks for robot-side launch tasks.

A launch task starts with the object on the robot's side -- a shuttle released
from the server's hand, a ball resting on the pitch, a ball held for a shot --
and the robot strikes it once toward a goal.  ``generate_shot_bank.py`` builds
banks of *incoming* shots, whose difficulty lives in the flight; here the
difficulty lives in where the object starts and how it is moving, and in how
far and where it must be sent.

The protocol is the return banks': 200/50/100 shots per level for
train/dev/test, split seeds derived from a documented namespace so the splits
cannot overlap, every shot simulated before it is accepted (L0: left alone, the
object must do what the bank claims), stratified conditions so every
worst-bucket criterion can be assessed, and a manifest that states the levels,
thresholds and perturbations in the sport's own units.

Usage::

    PYTHONPATH=src python scripts/generate_launch_banks.py --sport badminton
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np

from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
from multisport_sim.benchmark.types import ShotSpec

SEED_NAMESPACE = "multisport-launch-bank-v1"
SPLIT_LEVEL_COUNTS = {"train": 200, "dev": 50, "test": 100}


@dataclass(frozen=True)
class Condition:
    """One independently sampled sub-population of a level."""

    name: str
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]
    vx: tuple[float, float] = (0.0, 0.0)
    vy: tuple[float, float] = (0.0, 0.0)
    vz: tuple[float, float] = (0.0, 0.0)
    tags: tuple[str, ...] = ()
    # Sample y from +/- this range, so both service courts (or both flanks)
    # appear in every condition.
    mirror_y: bool = True
    # A ball moving on the ground starts rolling, not sliding: with only a
    # linear velocity, friction takes about 40% of it away in the first few
    # tenths of a second, and a "rolling" ball would really be a braking one.
    rolling_radius_m: float | None = None


@dataclass(frozen=True)
class LevelPlan:
    name: str
    distribution: str
    primary_metric: str
    pass_threshold: float
    conditions: tuple[Condition, ...]
    targets: bool = False
    held_out: bool = False
    pass_buckets: tuple[str, ...] = ()
    aggregate_metric: str | None = None
    aggregate_threshold: float | None = None


@dataclass(frozen=True)
class LaunchBankPlan:
    sport: str
    task: str
    bank: str
    timeout_s: float
    levels: dict[str, LevelPlan]
    # (tag, centre, radius); ``target_transform`` maps a centre to this shot,
    # given the shot's starting y -- badminton mirrors it to the receiving court.
    targets: tuple[tuple[str, tuple[float, float], float], ...]
    target_transform: Callable[[float, tuple[float, float]], tuple[float, float]]
    surface_bounds_x: tuple[float, float]
    surface_bounds_y: tuple[float, float]
    surface_top_z_m: float
    success_sequence: tuple[str, ...]
    landing_error: str
    lateral_tags: Callable[[float], tuple[str, ...]]
    judge_factory: Callable[[float], object]
    bucket_groups: dict[str, list[str]] = field(default_factory=dict)
    initial_motion: str = "stationary_or_toward_opponent"
    origin_label: str = "court_center_at_floor"
    # Where placement is measured, when it is not the playing surface.
    placement_plane: dict[str, object] | None = None
    l5_observation_noise: dict[str, float] = field(default_factory=dict)
    l5_domain_randomization: dict[str, object] = field(default_factory=dict)
    l5_latency: dict[str, int] = field(
        default_factory=lambda: {"observation_steps": 2, "action_steps": 1}
    )
    l5_summary: dict[str, str] = field(default_factory=dict)
    notes: str = ""


def _badminton_judge(timeout_s: float):
    from multisport_sim.benchmark.rules.badminton import BadmintonServeJudge

    return BadmintonServeJudge(timeout_s=timeout_s)


def _court_tags(y: float) -> tuple[str, ...]:
    # Facing +x the server's right hand is -y.
    return ("right-court",) if y < 0.0 else ("left-court",)


# Where a server stands: behind the short service line (x = -1.98), inside
# the singles width, shuttle held at roughly hip height.  Every serve is
# struck below 1.15 m, so a release much lower than ~0.95 m leaves almost no
# fall before the strike.
NOMINAL_SERVE = Condition("nominal", x=(-2.9, -2.4), y=(0.5, 1.2), z=(1.00, 1.08))
BADMINTON_LEVELS: dict[str, LevelPlan] = {
    "L0": LevelPlan(
        name="Physics",
        distribution="nominal releases without robot action; the shuttle must fall on the server's half",
        primary_metric="incoming_valid_rate",
        pass_threshold=1.0,
        conditions=(NOMINAL_SERVE,),
    ),
    "L1": LevelPlan(
        name="Contact",
        distribution="nominal releases from both service courts",
        primary_metric="hit_rate",
        pass_threshold=0.9,
        conditions=(NOMINAL_SERVE,),
    ),
    "L2": LevelPlan(
        name="Serve",
        distribution="releases anywhere a server stands in either service court",
        primary_metric="valid_return_rate",
        pass_threshold=0.8,
        conditions=(Condition("wide", x=(-3.3, -2.2), y=(0.3, 2.2), z=(0.97, 1.12)),),
    ),
    "L3": LevelPlan(
        name="Placement",
        distribution="L2 releases with short, mid, deep and wide landing targets in the receiving court",
        primary_metric="target_rate",
        pass_threshold=0.7,
        conditions=(Condition("wide", x=(-3.3, -2.2), y=(0.3, 2.2), z=(0.97, 1.12)),),
        targets=True,
    ),
    "L4": LevelPlan(
        name="Robustness",
        distribution="fixed combinations of far, wide, low and tossed releases",
        primary_metric="worst_bucket_valid_return_rate",
        pass_threshold=0.6,
        conditions=(
            Condition("far", x=(-4.2, -3.5), y=(0.3, 2.0), z=(0.97, 1.10), tags=("far-release",)),
            Condition("wide", x=(-3.2, -2.3), y=(2.0, 2.45), z=(0.97, 1.10), tags=("wide-release",)),
            Condition("low", x=(-3.2, -2.3), y=(0.3, 2.0), z=(0.92, 0.96), tags=("low-release",)),
            Condition(
                "tossed", x=(-3.2, -2.3), y=(0.3, 2.0), z=(0.97, 1.10),
                vx=(0.0, 0.6), vz=(-1.2, -0.4), tags=("tossed",),
            ),
        ),
        pass_buckets=("far-release", "wide-release", "low-release", "tossed"),
    ),
    "L5": LevelPlan(
        name="Generalization",
        distribution="held-out release combinations under observation noise, latency and domain randomization",
        primary_metric="worst_bucket_valid_return_rate",
        pass_threshold=0.5,
        aggregate_metric="valid_return_rate",
        aggregate_threshold=0.6,
        conditions=(
            Condition("far", x=(-4.4, -3.5), y=(0.3, 2.2), z=(0.95, 1.12), tags=("far-release",)),
            Condition(
                "tossed", x=(-3.4, -2.2), y=(0.3, 2.2), z=(0.95, 1.12),
                vx=(0.0, 0.8), vz=(-1.5, -0.3), tags=("tossed",),
            ),
            Condition("wide", x=(-3.4, -2.2), y=(2.0, 2.5), z=(0.95, 1.12), tags=("wide-release",)),
        ),
        held_out=True,
        pass_buckets=("far-release", "tossed", "wide-release", "held-out"),
    ),
}

def _football_judge(timeout_s: float):
    from multisport_sim.benchmark.rules.football import FootballKickJudge

    return FootballKickJudge(timeout_s=timeout_s)


def _basketball_judge(timeout_s: float):
    from multisport_sim.benchmark.rules.basketball import BasketballShootJudge

    return BasketballShootJudge(timeout_s=timeout_s)


def _flank_tags(y: float) -> tuple[str, ...]:
    # Facing +x the kicker's or shooter's left hand is +y.
    if abs(y) < 1.0:
        return ("central",)
    return ("left-side",) if y > 0.0 else ("right-side",)


FOOTBALL_BALL_Z = 0.11
"""A ball resting on the pitch: its centre is one radius up."""


def _on_ground(name, x, y, *, vx=(0.0, 0.0), vy=(0.0, 0.0), tags=()) -> Condition:
    return Condition(
        name, x=x, y=y, z=(FOOTBALL_BALL_Z, FOOTBALL_BALL_Z), vx=vx, vy=vy, tags=tags,
        rolling_radius_m=FOOTBALL_BALL_Z,
    )


# The task frame's origin is the centre of the goal line; the penalty mark is
# at x = -11 and the edge of the penalty area at x = -16.5.
FOOTBALL_LEVELS: dict[str, LevelPlan] = {
    "L0": LevelPlan(
        name="Physics",
        distribution="stationary balls inside the penalty area without robot action; the ball must stay put on the pitch",
        primary_metric="incoming_valid_rate",
        pass_threshold=1.0,
        conditions=(_on_ground("box", x=(-16.0, -11.0), y=(0.0, 3.0)),),
    ),
    "L1": LevelPlan(
        name="Contact",
        distribution="stationary balls inside the penalty area",
        primary_metric="hit_rate",
        pass_threshold=0.9,
        conditions=(_on_ground("box", x=(-16.0, -11.0), y=(0.0, 3.0)),),
    ),
    "L2": LevelPlan(
        name="Goal",
        distribution="stationary balls around the penalty area and its arc",
        primary_metric="valid_return_rate",
        pass_threshold=0.8,
        conditions=(_on_ground("around-box", x=(-22.0, -11.0), y=(0.0, 12.0)),),
    ),
    "L3": LevelPlan(
        name="Placement",
        distribution="L2 balls with top-corner and low-corner targets in the goal mouth",
        primary_metric="target_rate",
        pass_threshold=0.7,
        conditions=(_on_ground("around-box", x=(-22.0, -11.0), y=(0.0, 12.0)),),
        targets=True,
    ),
    "L4": LevelPlan(
        name="Robustness",
        distribution="fixed combinations of long range, tight angle, a ball rolling back to the kicker and a ball rolling across",
        primary_metric="worst_bucket_valid_return_rate",
        pass_threshold=0.6,
        conditions=(
            _on_ground("far", x=(-30.0, -24.0), y=(0.0, 8.0), tags=("long-range",)),
            _on_ground("angle", x=(-14.0, -8.0), y=(14.0, 18.0), tags=("tight-angle",)),
            _on_ground("rolling", x=(-20.0, -12.0), y=(0.0, 8.0), vx=(-4.0, -1.0), tags=("rolling",)),
            _on_ground("across", x=(-20.0, -12.0), y=(0.0, 8.0), vy=(2.0, 4.0), tags=("cross",)),
        ),
        pass_buckets=("long-range", "tight-angle", "rolling", "cross"),
    ),
    "L5": LevelPlan(
        name="Generalization",
        distribution="held-out combinations under observation noise, latency and domain randomization",
        primary_metric="worst_bucket_valid_return_rate",
        pass_threshold=0.5,
        aggregate_metric="valid_return_rate",
        aggregate_threshold=0.6,
        conditions=(
            _on_ground("far-rolling", x=(-28.0, -20.0), y=(0.0, 10.0), vx=(-3.0, -0.5),
                       tags=("long-range", "rolling")),
            _on_ground("angle-cross", x=(-16.0, -9.0), y=(10.0, 16.0), vy=(1.0, 3.0),
                       tags=("cross", "tight-angle")),
            _on_ground("box", x=(-18.0, -11.0), y=(0.0, 6.0), tags=("box",)),
        ),
        held_out=True,
        pass_buckets=("long-range", "tight-angle", "held-out"),
    ),
}


BASKETBALL_LEVELS: dict[str, LevelPlan] = {
    # The task frame's origin is the floor below the rim's centre; the free
    # throw line is 4.225 m in front of it and the three-point arc 6.75 m out.
    "L0": LevelPlan(
        name="Physics",
        distribution="releases at the free-throw line without robot action; the ball must fall in front of the basket",
        primary_metric="incoming_valid_rate",
        pass_threshold=1.0,
        conditions=(Condition("free-throw", x=(-4.5, -3.95), y=(0.0, 0.3), z=(2.0, 2.2)),),
    ),
    "L1": LevelPlan(
        name="Contact",
        distribution="releases at the free-throw line",
        primary_metric="hit_rate",
        pass_threshold=0.9,
        conditions=(Condition("free-throw", x=(-4.5, -3.95), y=(0.0, 0.3), z=(2.0, 2.2)),),
    ),
    "L2": LevelPlan(
        name="Basket",
        distribution="mid-range releases, 2.5-5.5 m in front of the rim",
        primary_metric="valid_return_rate",
        pass_threshold=0.8,
        conditions=(Condition("mid-range", x=(-5.5, -2.5), y=(0.0, 3.0), z=(1.9, 2.3)),),
    ),
    "L3": LevelPlan(
        name="Placement",
        distribution="L2 releases with a clean-entry target at the rim's centre or offset entries",
        primary_metric="target_rate",
        pass_threshold=0.7,
        conditions=(Condition("mid-range", x=(-5.5, -2.5), y=(0.0, 3.0), z=(1.9, 2.3)),),
        targets=True,
    ),
    "L4": LevelPlan(
        name="Robustness",
        distribution="fixed combinations of three-point range, the corners, a low release and a tossed ball",
        primary_metric="worst_bucket_valid_return_rate",
        pass_threshold=0.6,
        conditions=(
            Condition("three", x=(-7.3, -6.8), y=(0.0, 1.5), z=(2.0, 2.3), tags=("three-point",)),
            Condition("corner", x=(-1.0, -0.5), y=(5.6, 6.4), z=(2.0, 2.3), tags=("corner",)),
            Condition("low", x=(-5.0, -3.0), y=(0.0, 2.5), z=(1.55, 1.75), tags=("low-release",)),
            Condition("tossed", x=(-5.0, -3.0), y=(0.0, 2.5), z=(1.9, 2.2),
                      vx=(0.0, 0.8), vz=(0.5, 1.5), tags=("tossed",)),
        ),
        pass_buckets=("three-point", "corner", "low-release", "tossed"),
    ),
    "L5": LevelPlan(
        name="Generalization",
        distribution="held-out combinations under observation noise, latency and domain randomization",
        primary_metric="worst_bucket_valid_return_rate",
        pass_threshold=0.5,
        aggregate_metric="valid_return_rate",
        aggregate_threshold=0.6,
        conditions=(
            Condition("three-tossed", x=(-7.4, -6.8), y=(0.0, 2.5), z=(1.9, 2.2),
                      vx=(0.0, 0.6), vz=(0.3, 1.2), tags=("three-point", "tossed")),
            Condition("wing", x=(-4.0, -2.0), y=(3.0, 5.0), z=(1.8, 2.3), tags=("wing",)),
            Condition("low-corner", x=(-1.2, -0.6), y=(5.0, 6.2), z=(1.6, 1.8),
                      tags=("corner", "low-release")),
        ),
        held_out=True,
        pass_buckets=("three-point", "corner", "held-out"),
    ),
}


BANKS: dict[str, LaunchBankPlan] = {
    "badminton": LaunchBankPlan(
        sport="badminton",
        task="badminton-serve-v0",
        bank="serve-v0",
        timeout_s=4.0,
        levels=BADMINTON_LEVELS,
        targets=(
            ("short-target", (2.6, 0.9), 0.6),
            ("mid-target", (4.0, 1.3), 0.6),
            ("deep-target", (5.6, 1.3), 0.6),
            ("wide-target", (4.6, 2.1), 0.5),
        ),
        # Diagonal: the receiving court is across the centre line.
        target_transform=lambda server_y, centre: (
            centre[0],
            (-1.0 if server_y > 0.0 else 1.0) * centre[1],
        ),
        surface_bounds_x=(-6.70, 6.70),
        surface_bounds_y=(-2.59, 2.59),
        surface_top_z_m=0.0,
        success_sequence=(
            "robot_racket_contact_below_1.15_m",
            "cross_net_toward_opponent",
            "first_landing_in_diagonal_singles_service_court",
        ),
        landing_error="euclidean distance from target center in court x-y coordinates",
        lateral_tags=_court_tags,
        judge_factory=_badminton_judge,
        bucket_groups={"court": ["right-court", "left-court"]},
        l5_observation_noise={
            "ball_position_m": 0.010,
            "ball_velocity_mps": 0.30,
            "ball_spin_radps": 0.0,
            "joint_position_rad": 0.002,
            "joint_velocity_radps": 0.02,
            "camera_pixel": 4.0,
        },
        l5_domain_randomization={
            # BWF allows 4.74-5.50 g against a nominal 5 g (the model's mass).
            "ball_mass_scale": [0.95, 1.10],
            "drag_scale": [0.90, 1.10],
            "camera_position_m": 0.01,
        },
        l5_summary={
            "domain_randomization": (
                "shuttle mass 4.75-5.50 g (BWF tolerance), air density +/-10%, "
                "camera mount sigma 10 mm"
            ),
            "observation_noise": "shuttle position sigma 10 mm, velocity sigma 0.3 m/s",
            "latency": "2 control steps on observations, 1 on actions (10 ms and 5 ms at 200 Hz)",
        },
        notes=(
            "Serve task: the shuttle is released on the server's side and waits to be "
            "struck.  Level metric names are the return tasks'; incoming_valid means "
            "the release is legal and valid_return means a legal serve."
        ),
    ),
    "football": LaunchBankPlan(
        sport="football",
        task="football-kick-v0",
        bank="kick-v0",
        timeout_s=3.0,
        levels=FOOTBALL_LEVELS,
        targets=(
            ("top-left-target", (2.8, 1.8), 0.5),
            ("top-right-target", (-2.8, 1.8), 0.5),
            ("low-left-target", (2.8, 0.4), 0.5),
            ("low-right-target", (-2.8, 0.4), 0.5),
        ),
        target_transform=lambda start_y, centre: centre,
        surface_bounds_x=(-105.0, 0.0),
        surface_bounds_y=(-34.0, 34.0),
        surface_top_z_m=0.0,
        success_sequence=(
            "robot_boot_contact",
            "whole_ball_over_the_goal_line_between_the_posts_and_under_the_crossbar",
        ),
        landing_error="euclidean distance from target center in the goal mouth (y, z)",
        lateral_tags=_flank_tags,
        judge_factory=_football_judge,
        initial_motion="stationary_or_toward_robot",
        origin_label="goal_line_center_at_ground",
        placement_plane={
            "axes": ["y", "z"],
            "bounds_m": {"y": [-3.66, 3.66], "z": [0.0, 2.44]},
        },
        l5_observation_noise={
            "ball_position_m": 0.030,
            "ball_velocity_mps": 0.30,
            "ball_spin_radps": 2.0,
            "joint_position_rad": 0.002,
            "joint_velocity_radps": 0.02,
            "camera_pixel": 4.0,
        },
        l5_domain_randomization={
            # IFAB size 5: 410-450 g against the model's 430 g.
            "ball_mass_scale": [0.95, 1.05],
            "drag_scale": [0.90, 1.10],
            "camera_position_m": 0.02,
        },
        l5_summary={
            "domain_randomization": "ball mass 410-450 g (IFAB), air density +/-10%, camera mount sigma 20 mm",
            "observation_noise": "ball position sigma 30 mm, velocity sigma 0.3 m/s",
            "latency": "2 control steps on observations, 1 on actions (10 ms and 5 ms at 200 Hz)",
        },
        notes=(
            "Kick task: the ball rests or rolls on the pitch in front of the goal and is "
            "kicked once.  Task frame origin is the centre of the goal line.  valid_return "
            "means a goal; placement is measured in the goal mouth (y, z)."
        ),
    ),
    "basketball": LaunchBankPlan(
        sport="basketball",
        task="basketball-shoot-v0",
        bank="shoot-v0",
        timeout_s=4.0,
        levels=BASKETBALL_LEVELS,
        targets=(
            ("clean-target", (0.0, 0.0), 0.06),
            ("front-target", (-0.06, 0.0), 0.05),
            ("back-target", (0.06, 0.0), 0.05),
        ),
        target_transform=lambda start_y, centre: centre,
        surface_bounds_x=(-12.425, 1.575),
        surface_bounds_y=(-7.5, 7.5),
        surface_top_z_m=0.0,
        success_sequence=(
            "robot_plate_contact",
            "ball_centre_descends_through_the_ring",
        ),
        landing_error="euclidean distance from target center in the rim plane (x, y)",
        lateral_tags=_flank_tags,
        judge_factory=_basketball_judge,
        origin_label="floor_below_rim_center",
        placement_plane={
            "axes": ["x", "y"],
            "bounds_m": {"x": [-0.225, 0.225], "y": [-0.225, 0.225]},
        },
        l5_observation_noise={
            "ball_position_m": 0.015,
            "ball_velocity_mps": 0.15,
            "ball_spin_radps": 1.0,
            "joint_position_rad": 0.002,
            "joint_velocity_radps": 0.02,
            "camera_pixel": 4.0,
        },
        l5_domain_randomization={
            # FIBA size 7: 567-650 g against the model's 600 g.
            "ball_mass_scale": [0.95, 1.08],
            "drag_scale": [0.90, 1.10],
            "camera_position_m": 0.01,
        },
        l5_summary={
            "domain_randomization": "ball mass 567-650 g (FIBA), air density +/-10%, camera mount sigma 10 mm",
            "observation_noise": "ball position sigma 15 mm, velocity sigma 0.15 m/s",
            "latency": "2 control steps on observations, 1 on actions (10 ms and 5 ms at 200 Hz)",
        },
        notes=(
            "Shoot task: the ball is released in front of the basket and struck once by a "
            "launcher plate.  Task frame origin is the floor below the rim's centre.  "
            "valid_return means a made basket; placement is measured in the rim plane (x, y)."
        ),
    ),
}


def _split_seed(plan: LaunchBankPlan, split: str) -> int:
    payload = f"{SEED_NAMESPACE}:{plan.sport}:{split}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return float(low if low == high else rng.uniform(low, high))


def _candidate(
    rng: np.random.Generator,
    *,
    shot_id: str,
    level: str,
    plan: LevelPlan,
    condition: Condition,
    index: int,
    bank: LaunchBankPlan,
) -> tuple[ShotSpec, dict]:
    x = _uniform(rng, condition.x)
    y = _uniform(rng, condition.y)
    mirrored = condition.mirror_y and rng.random() < 0.5
    if mirrored:
        y = -y
    # Draw order is part of the frozen banks: x, y, mirror, z, vx, vy, vz.
    z = round(_uniform(rng, condition.z), 4)
    vx = round(_uniform(rng, condition.vx), 4)
    vy = round(_uniform(rng, condition.vy), 4)
    if mirrored:
        # A ball rolling across keeps rolling toward the same touchline side
        # relative to where it starts.
        vy = -vy if vy else vy
    vz = round(_uniform(rng, condition.vz), 4)
    spin = [0.0, 0.0, 0.0]
    if condition.rolling_radius_m is not None:
        radius = condition.rolling_radius_m
        spin = [round(-vy / radius, 4), round(vx / radius, 4), 0.0]
    record = {
        "shot_id": shot_id,
        "sport": bank.sport,
        "level": level,
        "position": [round(x, 4), round(y, 4), z],
        "linear_velocity": [vx, vy, vz],
        "angular_velocity": spin,
    }
    tags = {*bank.lateral_tags(y), *condition.tags}
    if plan.held_out:
        tags.add("held-out")
    if plan.targets:
        name, centre, radius = bank.targets[index % len(bank.targets)]
        u, v = bank.target_transform(y, centre)
        record["target"] = {"center_xy": [round(u, 4), round(v, 4)], "radius_m": radius}
        tags.add(name)
    record["tags"] = sorted(tags)
    shot = ShotSpec.from_mapping(record)
    return shot, record


def _accept(backend: MujocoShotBackend, shot: ShotSpec, bank: LaunchBankPlan) -> bool:
    """Left alone, the object must do what an L0 shot of this bank claims."""
    level_zero = ShotSpec.from_mapping({**shot.to_dict(), "level": "L0", "target": None})
    backend.reset()
    backend.launch_ball(level_zero)
    judge = bank.judge_factory(bank.timeout_s)
    judge.reset(level_zero)
    for _ in range(int(bank.timeout_s / backend.timestep) + 2):
        backend.step()
        judge.update(
            time_s=backend.time,
            ball=backend.get_ball_state(),
            contacts=backend.semantic_contacts(),
        )
        if judge.done:
            break
    return bool(judge.done and judge.result.incoming_valid)


def _generate_split(
    backend: MujocoShotBackend, split: str, *, per_level: int, bank: LaunchBankPlan
) -> Iterator[dict]:
    rng = np.random.default_rng(_split_seed(bank, split))
    for level, plan in bank.levels.items():
        for index in range(per_level):
            condition = plan.conditions[index % len(plan.conditions)]
            shot_id = f"{bank.task}-{split}-{level.lower()}-{index + 1:04d}"
            for _ in range(50):
                shot, record = _candidate(
                    rng, shot_id=shot_id, level=level, plan=plan,
                    condition=condition, index=index, bank=bank,
                )
                if _accept(backend, shot, bank):
                    yield record
                    break
            else:
                raise SystemExit(f"no acceptable {level} shot for condition {condition.name!r}")


def _ranges(conditions: tuple[Condition, ...]) -> dict[str, list[float]]:
    def span(values):
        return [min(v[0] for v in values), max(v[1] for v in values)]

    ys = [(-c.y[1], c.y[1]) if c.mirror_y else c.y for c in conditions]
    return {
        "lateral_y_m": span(ys),
        "height_z_m": span([c.z for c in conditions]),
        "speed_x_mps": span([c.vx for c in conditions]),
        "spin_y_radps": span(
            [
                (c.vx[0] / c.rolling_radius_m, c.vx[1] / c.rolling_radius_m)
                if c.rolling_radius_m
                else (0.0, 0.0)
                for c in conditions
            ]
        ),
    }


def _manifest(bank: LaunchBankPlan, digests, counts) -> dict:
    levels = {}
    for level, plan in bank.levels.items():
        spec: dict[str, object] = {
            "name": plan.name,
            "distribution": plan.distribution,
            "primary_metric": plan.primary_metric,
            "pass_threshold": plan.pass_threshold,
        }
        if plan.aggregate_metric is not None:
            spec["aggregate_metric"] = plan.aggregate_metric
            spec["aggregate_threshold"] = plan.aggregate_threshold
        if plan.pass_buckets:
            spec["pass_buckets"] = list(plan.pass_buckets)
        levels[level] = spec
    return {
        "schema_version": 1,
        "benchmark": "multisport-shot-skill-v0",
        "task": bank.task,
        "sport": bank.sport,
        "status": "experimental",
        "leaderboard_eligible": False,
        "warning": (
            "Statistically sufficient per level (100 test episodes) but still an "
            "experimental bank: per-bucket counts remain below 100, and no "
            "leaderboard has been opened."
        ),
        "notes": bank.notes,
        "units": "SI",
        "coordinate_system": {
            "origin": bank.origin_label,
            "x_axis": "robot_side_to_opponent_side",
            "y_axis": "court_lateral",
            "z_axis": "up",
            "net_plane_x_m": 0.0,
            "robot_side": "x < 0",
            "opponent_side": "x > 0",
            "shot_origin": "robot_side",
            "initial_motion": bank.initial_motion,
            "table_top_z_m": bank.surface_top_z_m,
            "table_bounds_m": {
                "x": list(bank.surface_bounds_x),
                "y": list(bank.surface_bounds_y),
            },
        },
        "episode": {
            "timeout_s": bank.timeout_s,
            "success_sequence": list(bank.success_sequence),
            "timeout_is_truncation": True,
        },
        "levels": levels,
        "difficulty_ranges": {
            level: _ranges(plan.conditions) for level, plan in bank.levels.items()
        },
        "targets": {
            "representation": "circle",
            "center_field": "target.center_xy",
            "radius_field": "target.radius_m",
            "landing_error": bank.landing_error,
            **({"placement_plane": bank.placement_plane} if bank.placement_plane else {}),
        },
        "bucket_groups": bank.bucket_groups,
        "l5_conditions": dict(bank.l5_summary),
        "failure_reasons": [
            "miss", "net", "own_side", "out", "floor", "timeout", "safety", "numerical", "fault",
        ],
        "ordering": "shot_id_lexicographic",
        "digest": {
            "algorithm": "sha256",
            "scope": "exact UTF-8 JSONL file bytes including the final newline",
        },
        "splits": {
            split: {
                "file": f"{split}.jsonl",
                "count": sum(counts[split].values()),
                "level_counts": counts[split],
                "sha256": digests[split],
            }
            for split in SPLIT_LEVEL_COUNTS
        },
        "generation": {
            "script": "scripts/generate_launch_banks.py",
            "seed_namespace": SEED_NAMESPACE,
            "split_seeds": {split: _split_seed(bank, split) for split in SPLIT_LEVEL_COUNTS},
            "acceptance": "simulated in MuJoCo without robot action; kept only when the judge's L0 verdict is incoming_valid",
            "tag_source": "court side and release condition from the sampled shot",
        },
        "perturbations": {
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
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sport", choices=sorted(BANKS), required=True)
    parser.add_argument("--out", type=Path, default=Path("src/multisport_sim/benchmark/data"))
    args = parser.parse_args(argv)

    bank = BANKS[args.sport]
    destination = args.out / bank.sport / bank.bank
    destination.mkdir(parents=True, exist_ok=True)
    backend = MujocoShotBackend(sport=bank.sport)
    digests: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for split, per_level in SPLIT_LEVEL_COUNTS.items():
        records = sorted(
            _generate_split(backend, split, per_level=per_level, bank=bank),
            key=lambda record: record["shot_id"],
        )
        path = destination / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records),
            encoding="utf-8",
        )
        digests[split] = sha256(path.read_bytes()).hexdigest()
        counts[split] = {
            level: sum(1 for r in records if r["level"] == level) for level in bank.levels
        }
        print(f"{split}: {len(records)} shots -> {path}", file=sys.stderr)
    (destination / "manifest.json").write_text(
        json.dumps(_manifest(bank, digests, counts), indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest -> {destination / 'manifest.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
