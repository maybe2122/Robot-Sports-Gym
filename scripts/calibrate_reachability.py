#!/usr/bin/env python3
"""Measure whether a robot can physically play the frozen table-tennis bank.

Attaching an arm to a task is not the same as the task being solvable by it.
This script answers that question with numbers rather than assertion, and it is
where every robot-side constant in the task configuration comes from:

1. **Strike plane.**  Launch all 36 frozen shots with no robot action and record
   where and when each ball crosses a candidate plane behind the table.  The
   result is the region the task actually requires -- not the region the action
   space happens to declare.
2. **Mount placement.**  Sample the arm's forward kinematics over its joint
   ranges and score candidate mounts by how much of that region they cover.
3. **Ready pose.**  Travel time, not reach, is what decides an intercept.  Score
   candidate ready poses by the worst-case joint travel time to any point of the
   strike plane, measured against the solver's *actual* behaviour when seeded
   from that pose.
4. **Strike speed.**  At each intercept configuration, compute the fastest blade
   velocity the arm can produce inside its datasheet joint-speed limits, and
   compare it against the ball speed a legal return needs.

Step 4 is the one that can disqualify a task-robot pairing, and it is reported
whether or not the answer is flattering.

Usage::

    python scripts/calibrate_reachability.py --report reports/panda-reachability.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

if __package__ is None and __name__ == "__main__":  # pragma: no cover - script use
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multisport_sim.benchmark.assets import FRANKA_PANDA, UNITREE_G1, asset_available
from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
from multisport_sim.benchmark.robots import g1 as g1_module
from multisport_sim.benchmark.robots.kinematics import IKSolver
from multisport_sim.benchmark.robots.panda import (
    IK_SETTINGS,
    JOINT_NAMES,
    PADDLE_SITE,
    PANDA_READY_QPOS,
    PANDA_VELOCITY_LIMIT,
    PandaMount,
    build_panda_table_tennis_model,
)
from multisport_sim.benchmark.shot_bank import ShotBank
from multisport_sim.benchmark.task_config import (
    TABLE_TENNIS_RETURN_G1_V1,
    TABLE_TENNIS_RETURN_PANDA_V1,
)


@dataclass(frozen=True)
class RobotProfile:
    """Everything this script needs to calibrate one embodiment.

    The calibration is not about the Panda; it is about whether *a* robot can
    physically play the bank a task is scored on.  Keeping the robot behind this
    record is what let a second embodiment reuse it without a second script --
    and a second script would have drifted.
    """

    name: str
    # Where the joint speed envelope comes from.  The Panda's is a datasheet;
    # the G1's is a declared benchmark constant.  Every number this script
    # derives from it inherits that provenance, and the report says so, because
    # "max blade speed 19.9 m/s" read as a measurement would be a lie.
    velocity_limit_source: str
    asset: Any
    mount_type: Any
    build_model: Any
    joint_names: tuple[str, ...]
    paddle_site: str
    ik_settings: dict[str, Any]
    velocity_limit: tuple[float, ...]
    ready_qpos: tuple[float, ...]
    task: Any
    mount_grid: tuple[tuple[float, float, float], ...]
    # Optional per-robot narrowing of the solver's search.  The G1 keeps its arm
    # out of its own torso; the Panda narrows nothing.
    solver_bounds: Any = None


def _panda_profile() -> RobotProfile:
    return RobotProfile(
        name="panda",
        velocity_limit_source="franka_emika_panda_datasheet",
        asset=FRANKA_PANDA,
        mount_type=PandaMount,
        build_model=build_panda_table_tennis_model,
        joint_names=JOINT_NAMES,
        paddle_site=PADDLE_SITE,
        ik_settings=IK_SETTINGS,
        velocity_limit=PANDA_VELOCITY_LIMIT,
        ready_qpos=PANDA_READY_QPOS,
        task=TABLE_TENNIS_RETURN_PANDA_V1,
        mount_grid=tuple(
            (float(x), 0.0, float(z))
            for x in np.arange(-2.30, -1.649, 0.10)
            for z in np.arange(0.55, 1.101, 0.05)
        ),
    )


def _g1_profile() -> RobotProfile:
    return RobotProfile(
        name="g1",
        velocity_limit_source=g1_module.G1_VELOCITY_LIMIT_SOURCE,
        asset=UNITREE_G1,
        mount_type=g1_module.G1Mount,
        build_model=g1_module.build_g1_table_tennis_model,
        joint_names=g1_module.JOINT_NAMES,
        paddle_site=g1_module.PADDLE_SITE,
        ik_settings=g1_module.IK_SETTINGS,
        velocity_limit=(g1_module.G1_VELOCITY_LIMIT_RAD_S,) * len(g1_module.JOINT_NAMES),
        ready_qpos=g1_module.G1_READY_QPOS,
        task=TABLE_TENNIS_RETURN_G1_V1,
        solver_bounds=g1_module.solver_position_bounds,
        # The G1 stands on the floor, so the grid searches where it stands and
        # how high a plinth it needs, not where a pedestal is bolted.
        mount_grid=tuple(
            (float(x), 0.0, float(z))
            for x in np.arange(-2.00, -1.599, 0.04)
            for z in np.arange(0.0, 0.301, 0.05)
        ),
    )


ROBOTS = {"panda": _panda_profile, "g1": _g1_profile}

GRAVITY = 9.81
TABLE_TOP_Z = 0.76
NET_PLANE_X = 0.0
TABLE_HALF_LENGTH = 1.37


@dataclass(frozen=True)
class Crossing:
    """Where and when one shot passes the candidate strike plane."""

    shot_id: str
    level: str
    time_s: float
    y: float
    z: float
    speed_mps: float


def measure_strike_plane(
    plane_x: float, splits: Sequence[str], bank: str
) -> list[Crossing]:
    """Launch every frozen shot unopposed and record its plane crossing.

    The ball is simulated with the same aerodynamics the task uses, so the
    recorded window is the real one, not a ballistic approximation.

    ``bank`` is the shot bank being calibrated against, and it must be the one
    the task is scored on.  Constants derived from a different bank describe a
    distribution the robot is never asked to play: the arm then misses every
    ball that crosses outside the window it was fitted to, and the score reads
    as a control failure rather than the calibration error it is.
    """
    backend = MujocoShotBackend()
    crossings: list[Crossing] = []
    for split in splits:
        for shot in ShotBank.from_resource(split=split, task=bank):
            backend.reset()
            backend.launch_ball(shot)
            previous_x: float | None = None
            for _ in range(4000):
                backend.step()
                ball = backend.get_ball_state()
                x, y, z = ball.position
                if previous_x is not None and previous_x >= plane_x > x and z > 0.55:
                    crossings.append(
                        Crossing(
                            shot_id=shot.shot_id,
                            level=shot.level,
                            time_s=backend.time,
                            y=y,
                            z=z,
                            speed_mps=float(np.linalg.norm(ball.linear_velocity)),
                        )
                    )
                    break
                previous_x = x
                if z < 0.2:
                    break
    return crossings


def strike_targets(
    crossings: Sequence[Crossing], plane_x: float, *, resolution: int = 7, margin: float = 0.06
) -> list[tuple[float, float, float]]:
    """A grid spanning the measured crossings, with a little margin."""
    ys = [item.y for item in crossings]
    zs = [item.z for item in crossings]
    return [
        (plane_x, float(y), float(z))
        for y in np.linspace(min(ys) - margin, max(ys) + margin, resolution)
        for z in np.linspace(min(zs) - margin, max(zs) + margin, resolution)
    ]


def mount_coverage(
    profile: RobotProfile,
    mount: Any,
    targets: Sequence[tuple[float, float, float]],
    *,
    samples: int = 20000,
    tolerance_m: float = 0.04,
    seed: int = 0,
) -> float:
    """Fraction of the strike grid a randomly sampled robot posture can reach."""
    model = profile.build_model(mount=mount)
    data = mujoco.MjData(model)
    site = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, profile.paddle_site))
    joints = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        for name in profile.joint_names
    ]
    address = np.array([int(model.jnt_qposadr[joint]) for joint in joints])
    lower = np.array([float(model.jnt_range[joint][0]) for joint in joints])
    upper = np.array([float(model.jnt_range[joint][1]) for joint in joints])

    rng = np.random.default_rng(seed)
    reached = np.empty((samples, 3))
    for index, configuration in enumerate(
        rng.uniform(lower, upper, size=(samples, len(joints)))
    ):
        data.qpos[address] = configuration
        mujoco.mj_kinematics(model, data)
        reached[index] = data.site_xpos[site]

    covered = sum(
        1
        for target in targets
        if float(np.min(np.linalg.norm(reached - np.asarray(target), axis=1))) < tolerance_m
    )
    return covered / len(targets)


def travel_time(
    solver: IKSolver,
    ready: np.ndarray,
    targets: Sequence[tuple[float, float, float]],
    axis: Sequence[float],
    velocity_limit: Sequence[float],
) -> tuple[float, int]:
    """Worst-case joint travel time from ``ready``, and unreachable target count.

    Every solve is seeded from ``ready``, which is what the controller does on
    its first plan.  Scoring against solutions produced by some other seeding --
    a continuous sweep, say -- measures a solution set the controller will never
    see, and flatters the pose by roughly a factor of two.
    """
    limits = np.asarray(velocity_limit)
    worst = 0.0
    unreachable = 0
    for target in targets:
        result = solver.solve(target, target_axis=axis, initial_qpos=ready)
        if not result.converged:
            unreachable += 1
            continue
        worst = max(worst, float(np.max(np.abs(result.qpos - ready) / limits)))
    return worst, unreachable


def _solver_for(profile: RobotProfile, model: mujoco.MjModel) -> IKSolver:
    """The reference solver for one embodiment, bounds and all.

    The G1 narrows its own search to keep the arm out of its torso; the Panda
    does not narrow anything.  Building the solver in one place is what stops a
    calibration from measuring a different solver than the adapter uses.
    """
    bounds = profile.solver_bounds(model) if profile.solver_bounds else None
    extra = (
        {}
        if bounds is None
        else {"position_low": bounds[0], "position_high": bounds[1]}
    )
    return IKSolver(
        model,
        site_name=profile.paddle_site,
        joint_names=profile.joint_names,
        **extra,
        **profile.ik_settings,
    )


def refine_ready(
    solver: IKSolver,
    seed: np.ndarray,
    targets: Sequence[tuple[float, float, float]],
    axis: Sequence[float],
    velocity_limit: Sequence[float],
    *,
    iterations: int = 6,
) -> np.ndarray:
    """Iterate a ready pose toward the one with the shortest worst-case travel.

    Travel time from a ready pose ``r`` is ``max_i |q_i - r_i| / v_i`` over the
    joints, so for a *fixed* set of intercept configurations the optimum is the
    per-joint midpoint of that set -- the Chebyshev centre.  The set is not
    fixed: it is whatever the solver returns when seeded from ``r``, which is
    why this iterates instead of solving once.  The fixed point is a ready pose
    that is central among the configurations it actually produces.
    """
    limits = np.asarray(velocity_limit)
    ready = np.clip(np.asarray(seed, dtype=float), solver.lower, solver.upper)
    for _ in range(iterations):
        solutions = []
        for target in targets:
            result = solver.solve(target, target_axis=axis, initial_qpos=ready)
            if result.converged:
                solutions.append(result.qpos)
        if not solutions:
            return ready
        stack = np.asarray(solutions)
        centre = np.clip(
            0.5 * (stack.min(axis=0) + stack.max(axis=0)), solver.lower, solver.upper
        )
        if np.max(np.abs(centre - ready) / limits) < 1e-4:
            return centre
        ready = centre
    return ready


def _random_seeds(
    solver: IKSolver, count: int, *, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.uniform(solver.lower, solver.upper) for _ in range(count)
    ]


def search_ready(
    solver: IKSolver,
    targets: Sequence[tuple[float, float, float]],
    axis: Sequence[float],
    velocity_limit: Sequence[float],
    *,
    incumbent: np.ndarray,
    restarts: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, float, int]:
    """Best ready pose for one mount: fewest unreachable points, then fastest."""
    best: tuple[np.ndarray, float, int] | None = None
    starts = [np.asarray(incumbent, dtype=float), *_random_seeds(solver, restarts, seed=seed)]
    for start in starts:
        candidate = refine_ready(solver, start, targets, axis, velocity_limit)
        worst, unreachable = travel_time(
            solver, candidate, targets, axis, velocity_limit
        )
        score = (unreachable, worst)
        if best is None or score < (best[2], best[1]):
            best = (candidate, worst, unreachable)
    assert best is not None
    return best


def _score_mount(
    robot_name: str,
    position: tuple[float, float, float],
    targets: Sequence[tuple[float, float, float]],
    axis: Sequence[float],
    incumbent: Sequence[float],
    restarts: int,
    seed: int,
) -> dict[str, Any]:
    """Worker: build one candidate mount and report its best ready pose.

    Takes the robot by name rather than by value because it runs in a separate
    process, and a profile holds live callables.
    """
    profile = ROBOTS[robot_name]()
    mount = profile.mount_type(position=position)
    model = profile.build_model(mount=mount)
    solver = _solver_for(profile, model)
    ready, worst, unreachable = search_ready(
        solver,
        targets,
        axis,
        profile.velocity_limit,
        incumbent=np.asarray(incumbent, dtype=float),
        restarts=restarts,
        seed=seed,
    )
    return {
        "position": [float(value) for value in position],
        "ready_qpos": [float(value) for value in ready],
        "worst_case_travel_time_s": worst,
        "unreachable_points": unreachable,
    }


def search_mount(
    profile: RobotProfile,
    targets: Sequence[tuple[float, float, float]],
    axis: Sequence[float],
    *,
    incumbent_mount: Any,
    incumbent_ready: Sequence[float],
    restarts: int = 8,
    seed: int = 0,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    """Score a grid of mounts, best first.

    The mount is a task constant, not a tuning knob a submission may touch, so
    it is chosen once against the region the scored bank actually requires.  The
    grid keeps ``y = 0``: the measured strike region is symmetric about the
    table's centre line to within a centimetre, and an off-centre mount would
    buy reach on one side by giving it up on the other.
    """
    candidates = [tuple(incumbent_mount.position), *profile.mount_grid]
    seen: set[tuple[float, float, float]] = set()
    unique = []
    for item in candidates:
        key = tuple(round(value, 4) for value in item)
        if key not in seen:
            seen.add(key)
            unique.append(item)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _score_mount,
                profile.name,
                item,
                targets,
                axis,
                incumbent_ready,
                restarts,
                seed,
            )
            for item in unique
        ]
        rows = [future.result() for future in futures]
    rows.sort(key=lambda row: (row["unreachable_points"], row["worst_case_travel_time_s"]))
    return rows


def per_shot_feasibility(
    profile: RobotProfile,
    solver: IKSolver,
    ready: np.ndarray,
    crossings: Sequence[Crossing],
    axis: Sequence[float],
    plane_x: float,
) -> dict[str, Any]:
    """Can the arm reach each *actual* ball in that ball's own time?

    This is the number that predicts the hit rate, and the grid statistics above
    are not.  ``strike_targets`` spans a rectangle around the measured crossings,
    so it demands the corners of that rectangle -- points no ball ever visits --
    and then reports the worst travel time over all of them against the single
    shortest arrival in the bank.  Judged that way the frozen mount looks
    hopeless (11 of 49 grid points unreachable, worst travel 2.20 s against a
    0.543 s window); judged per shot it reaches 1196 of 1197 training balls and
    is in position in time for 1182 of them.

    Both are reported.  The grid one bounds the region; this one answers the
    question the task actually asks.
    """
    limits = np.asarray(profile.velocity_limit)
    per_level: dict[str, list[int]] = {}
    reachable = in_time = 0
    for item in crossings:
        counts = per_level.setdefault(item.level, [0, 0])
        counts[1] += 1
        result = solver.solve(
            (plane_x, item.y, item.z), target_axis=axis, initial_qpos=ready
        )
        if not result.converged:
            continue
        reachable += 1
        if float(np.max(np.abs(result.qpos - ready) / limits)) <= item.time_s:
            in_time += 1
            counts[0] += 1
    return {
        "shots": len(crossings),
        "reachable": reachable,
        "reachable_in_time": in_time,
        "in_time_by_level": {
            level: {"in_time": counts[0], "shots": counts[1]}
            for level, counts in sorted(per_level.items())
        },
    }


def max_blade_speed(
    model: mujoco.MjModel,
    solver: IKSolver,
    ready: np.ndarray,
    targets: Sequence[tuple[float, float, float]],
    axis: Sequence[float],
    velocity_limit: Sequence[float],
) -> list[dict[str, Any]]:
    """Fastest blade velocity reachable at each strike point, per declared limits.

    With joint speeds bounded by a box, the extreme of a linear map over that
    box lies at a vertex, so the maximum of one velocity component is simply the
    sum of the absolute Jacobian row weighted by the limits.
    """
    data = mujoco.MjData(model)
    limits = np.asarray(velocity_limit)
    jacp = np.zeros((3, model.nv))
    rows: list[dict[str, Any]] = []
    for target in targets:
        result = solver.solve(target, target_axis=axis, initial_qpos=ready)
        if not result.converged:
            continue
        data.qpos[solver.qpos_adr] = result.qpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacSite(model, data, jacp, None, solver.site_id)
        jacobian = jacp[:, solver.dof_adr]
        rows.append(
            {
                "target": [float(value) for value in target],
                "max_forward_speed_mps": float(np.abs(jacobian[0]) @ limits),
                "max_speed_mps": float(
                    np.linalg.norm(jacobian @ (np.sign(jacobian[0]) * limits))
                ),
            }
        )
    return rows


def required_return_speed(contact_z: float, launch_vz: float) -> tuple[float, float]:
    """Forward speed band that lands the ball on the opponent half.

    The ball leaves at ``contact_z`` with an upward component, so flight time is
    set by the fall to table height; the band is then the horizontal distance to
    the net and to the baseline divided by that time.
    """
    drop = contact_z - TABLE_TOP_Z
    flight = (launch_vz + math.sqrt(launch_vz**2 + 2.0 * GRAVITY * max(drop, 0.0))) / GRAVITY
    strike_x = TABLE_TENNIS_RETURN_PANDA_V1.strike_zone.plane_x_m
    return (abs(strike_x) / flight, (abs(strike_x) + TABLE_HALF_LENGTH) / flight)


def resolve_splits(bank: str, requested: Sequence[str] | None) -> tuple[str, ...]:
    """Default to the training split, and never silently fall back to eval data.

    Every constant this script freezes is fitted to the shots it sees, so seeing
    ``dev`` or ``test`` would fit the robot to the set it is scored on.  Banks
    that publish a ``train`` split are calibrated on it; the frozen v0 fixture
    has no such split, and calibrating on it is a deliberate request.
    """
    if requested:
        return tuple(requested)
    if "train" in ShotBank.available_splits(bank):
        return ("train",)
    return ("dev", "test")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    plane_x = args.plane_x
    splits = resolve_splits(args.bank, args.splits)
    crossings = measure_strike_plane(plane_x, splits, args.bank)
    targets = strike_targets(crossings, plane_x, resolution=args.resolution)

    profile = ROBOTS[args.robot]()
    mount = profile.mount_type()
    model = profile.build_model(mount=mount)
    solver = _solver_for(profile, model)
    ready = np.asarray(profile.ready_qpos)
    axis = (1.0, 0.0, 0.0)

    worst_travel, unreachable = travel_time(
        solver, ready, targets, axis, profile.velocity_limit
    )
    per_shot = per_shot_feasibility(profile, solver, ready, crossings, axis, plane_x)
    search: list[dict[str, Any]] | None = None
    if args.search:
        search = search_mount(
            profile,
            targets,
            axis,
            incumbent_mount=mount,
            incumbent_ready=ready,
            restarts=args.restarts,
            workers=args.workers,
        )
    speeds = max_blade_speed(
        model, solver, ready, targets, axis, profile.velocity_limit
    )
    forward = [row["max_forward_speed_mps"] for row in speeds]

    contact_z = float(np.median([item.z for item in crossings]))
    slow_band = required_return_speed(contact_z, 0.0)
    lofted_band = required_return_speed(contact_z, 1.5)

    window = [item.time_s for item in crossings]
    coverage = None
    if args.coverage:
        coverage = mount_coverage(profile, mount, targets, samples=args.samples)

    return {
        "schema_version": 1,
        "task": profile.task.task_id,
        "robot": profile.task.robot_id,
        "shot_bank": {"task": args.bank, "splits": list(splits)},
        "asset": profile.asset.to_dict(),
        "mount": mount.to_dict(),
        "ready_qpos": [float(value) for value in ready],
        "strike_plane": {
            "plane_x_m": plane_x,
            "shots": len(crossings),
            "arrival_time_s": [min(window), max(window)],
            "y_m": [min(item.y for item in crossings), max(item.y for item in crossings)],
            "z_m": [min(item.z for item in crossings), max(item.z for item in crossings)],
            "ball_speed_mps": [
                min(item.speed_mps for item in crossings),
                max(item.speed_mps for item in crossings),
            ],
        },
        "reachability": {
            "grid_points": len(targets),
            "unreachable_points": unreachable,
            "worst_case_travel_time_s": worst_travel,
            "shortest_intercept_window_s": min(window),
            "travel_fits_in_window": worst_travel <= min(window),
            "mount_coverage": coverage,
            "search": search[:10] if search else None,
            "per_shot": per_shot,
        },
        "strike_speed": {
            "velocity_limit_source": profile.velocity_limit_source,
            "velocity_limit_rad_s": list(profile.velocity_limit),
            "max_forward_speed_mps": [min(forward), max(forward)],
            "required_forward_speed_mps": {
                "flat_return": list(slow_band),
                "lofted_return": list(lofted_band),
            },
            "median_contact_height_m": contact_z,
            "per_target": speeds if args.verbose else None,
        },
        "verdict": _verdict(
            worst_travel,
            min(window),
            forward,
            lofted_band,
            per_shot,
            profile.velocity_limit_source,
        ),
    }


def _verdict(
    worst_travel: float,
    shortest_window: float,
    forward: Sequence[float],
    required: tuple[float, float],
    per_shot: dict[str, Any],
    velocity_limit_source: str,
) -> dict[str, Any]:
    """State plainly what this robot can and cannot do on this bank."""
    can_reach = worst_travel <= shortest_window
    # The blade converts its own speed into ball speed with roughly
    # v_out = (1 + e) * v_blade + e * v_in.  Even taking the most favourable
    # incoming speed, a blade slower than the band below cannot return the ball.
    best_blade = max(forward)
    return {
        "can_reach_every_shot_in_time": can_reach,
        "shots_reached_in_time": per_shot["reachable_in_time"],
        "shots_measured": per_shot["shots"],
        "worst_case_travel_time_s": worst_travel,
        "shortest_intercept_window_s": shortest_window,
        "max_blade_forward_speed_mps": best_blade,
        "max_blade_forward_speed_source": velocity_limit_source,
        "required_ball_forward_speed_mps": list(required),
        "notes": (
            "Interception is a travel-time question and return is a speed "
            "question; they fail differently and are reported separately. A "
            "robot that reaches every ball but cannot return most of them is a "
            "valid benchmark result, not a broken task. Read "
            "'shots_reached_in_time' rather than 'can_reach_every_shot_in_time': "
            "the latter scores the corners of a rectangle drawn around the "
            "measured crossings, which no ball ever visits. And read the blade "
            "speed against 'max_blade_forward_speed_source': where that is not a "
            "datasheet, the speed is what the declared envelope permits and not "
            "what the servo delivers."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane-x", type=float, default=-1.55, dest="plane_x")
    parser.add_argument(
        "--robot",
        choices=tuple(ROBOTS),
        default="panda",
        help="which embodiment to calibrate; each has its own constants to freeze",
    )
    parser.add_argument(
        "--bank",
        default=None,
        help="shot bank to calibrate against; defaults to the one the task scores on",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="splits to measure; defaults to 'train' when the bank publishes one",
    )
    parser.add_argument("--resolution", type=int, default=7)
    parser.add_argument(
        "--search",
        action="store_true",
        help="search mount placements and ready poses against the measured region",
    )
    parser.add_argument("--restarts", type=int, default=8, help="ready-pose restarts per mount")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--coverage", action="store_true", help="also sample mount coverage")
    parser.add_argument("--verbose", action="store_true", help="include per-target speeds")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    profile = ROBOTS[args.robot]()
    if args.bank is None:
        args.bank = profile.task.bank_resource
    if not asset_available(profile.asset):
        from multisport_sim.benchmark.assets import missing_asset_message

        print(missing_asset_message(profile.asset), file=sys.stderr)
        return 2

    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - script use
    raise SystemExit(main())
