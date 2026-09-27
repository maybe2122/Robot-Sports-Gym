"""Score an arbitrary policy on the embodied table-tennis task, level by level.

The CLI runs the built-in reference baselines.  This module runs *someone
else's* policy through the same scored path -- same frozen shot bank, same
judge, same report envelope -- so that a submission's numbers and a baseline's
numbers are the same kind of number.

A policy is any callable ``obs -> action``:

* ``obs`` on the **state track** -- a ``float32`` array of 33 numbers, exactly
  what ``MultiSportRobot/TableTennisReturn-Panda-v1`` gives, packed by
  :func:`~multisport_sim.benchmark.envs.panda_observation_vector`.
* ``obs`` on the **vision track** -- a
  :class:`~multisport_sim.benchmark.vision.VisionObservation` with ``time_s``,
  ``robot`` and ``sensors``, and no ball state.  That absence is the track.
* ``action`` -- seven joint-position setpoints in radians, in the arm's own
  units.  A setpoint outside the declared joint limits is rejected rather than
  clipped: a policy that commands the impossible has to find out that it did.

Optional hooks: ``reset(seed=...)`` is called once per episode when present,
and an object exposing ``act`` is accepted in place of a bare callable.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from .envs import robot_observation_vector
from .metrics import robustness_gap
from .perturbations import PerturbedBackend, for_level
from .reporting import report_markdown, write_report
from .robot import ControlMode, JointCommand, WorkspaceBox
from .runner import RunConfig, run_shots
from .shot_bank import VALID_LEVELS, ShotBank
from .task_config import TABLE_TENNIS_RETURN_PANDA_V1
from .types import EpisodeResult
from .vision import TABLE_TENNIS_VISION_SENSORS, VisionTrackBackend

TRACKS = ("state", "vision")

TASK = TABLE_TENNIS_RETURN_PANDA_V1


class PolicyController:
    """Adapt a ``obs -> action`` callable to the benchmark controller protocol."""

    def __init__(self, policy: Any, *, policy_id: str, track: str = "state", task=TASK) -> None:
        act = getattr(policy, "act", None)
        self._call = act if callable(act) else policy
        if not callable(self._call):
            raise TypeError("policy must be callable or expose an act() method")
        self._policy = policy
        self.controller_id = policy_id
        if track not in TRACKS:
            raise ValueError(f"track must be one of {', '.join(TRACKS)}")
        self.track = track
        self.task = task
        # float32, matching the Gymnasium action space: a limit rounded to
        # float32 must not read as a violation of itself.
        self._low, self._high = (
            np.asarray(bound, dtype=np.float32) for bound in task.action_bounds()
        )

    def reset(self, shot, *, seed: int | None = None) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset(seed=seed)

    def act(self, observation: Any) -> JointCommand:
        # On the vision track the observation is handed over unchanged: there is
        # no privileged state to pack, and flattening images would decide the
        # policy's architecture for it.
        payload = (
            observation
            if self.track == "vision"
            else robot_observation_vector(observation, self.task)
        )
        action = np.asarray(self._call(payload), dtype=np.float32)
        if action.shape != (self.task.ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError(
                f"policy must return {self.task.ACTION_DIM} finite joint setpoints, "
                f"got shape {action.shape}"
            )
        if np.any(action < self._low) or np.any(action > self._high):
            raise ValueError(
                "policy commanded a joint setpoint outside the declared limits: "
                f"{action.tolist()}"
            )
        return JointCommand(tuple(float(value) for value in action), ControlMode.JOINT_POSITION)


def load_policy(spec: str) -> Any:
    """Resolve ``module:attribute``, calling it as a factory when that works."""
    if ":" not in spec:
        raise ValueError("--policy must look like 'module:attribute'")
    module_name, _, attribute = spec.partition(":")
    target = import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    if callable(target):
        try:
            built = target()
        except TypeError:
            return target
        if built is not None:
            return built
    return target


def evaluate_level(
    policy: Any,
    *,
    level: str,
    split: str,
    seed: int,
    policy_id: str,
    observation_label: str,
    episodes: int | None,
    track: str = "state",
    bank: str = TASK.bank_resource,
) -> dict[str, Any]:
    """Run every shot at one level and return the standard report body."""
    from .backends.mujoco_robot import MujocoPandaTableTennisBackend

    source_bank = ShotBank.from_resource(split=split, task=bank)
    level_bank = source_bank.filter(level=level)
    if not level_bank:
        raise SystemExit(f"split {split!r} contains no shots for {level}")
    shots = tuple(level_bank)[:episodes] if episodes else tuple(level_bank)

    timeout_s = float(source_bank.manifest["episode"]["timeout_s"])
    task = replace(TASK, split=split, timeout_s=timeout_s)
    vision_track = track == "vision"
    backend = MujocoPandaTableTennisBackend(
        workspace=WorkspaceBox(task.workspace.position_low, task.workspace.position_high),
        sensors=TABLE_TENNIS_VISION_SENSORS if vision_track else (),
    )
    controller = PolicyController(policy, policy_id=policy_id, track=track)
    runner_backend = VisionTrackBackend(backend) if vision_track else backend
    perturbations = for_level(source_bank.manifest, level)
    if not perturbations.is_identity:
        runner_backend = PerturbedBackend(runner_backend, perturbations)
    output = run_shots(
        runner_backend,
        controller,
        shots,
        config=RunConfig.from_task_config(task, seed=seed),
    )
    from ..benchmark_cli import _assemble_report, robot_metadata

    args = argparse.Namespace(
        robot="panda", seed=seed, level=level, control_hz=task.control_hz, track=track, bank=bank
    )
    return _assemble_report(
        args,
        source_bank,
        level_bank,
        shots,
        backend,
        task,
        output,
        robot_metadata(backend, task),
        {
            "id": policy_id,
            "observation": observation_label,
            "track": track,
            # Set by the harness, not by the policy: the bank is still an
            # experimental fixture, so nothing run on it is submittable.
            "benchmark_eligible": False,
        },
        timeout_s,
    )


def difficulty_table(reports: dict[str, dict[str, Any]]) -> str:
    """One row per level: the level's own pass criterion plus shared metrics."""
    lines = [
        "| Level | Episodes | Primary metric | Value | Threshold | Pass | Hit | Return | Target | Safety | Energy (J) |",
        "|---|---:|---|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for level, report in sorted(reports.items()):
        assessment = report.get("assessment") or {}
        metrics = report.get("metrics", {})
        robot = report.get("robot_metrics", {})
        value = assessment.get("primary_value")
        threshold = assessment.get("pass_threshold")
        lines.append(
            "| {level} | {n} | `{metric}` | {value} | {threshold} | {passed} | {hit:.0%} | "
            "{ret:.0%} | {tgt:.0%} | {safety} | {energy:.1f} |".format(
                level=level,
                n=report.get("episodes", "?"),
                metric=assessment.get("primary_metric", "n/a"),
                value="n/a" if value is None else f"{value:.0%}",
                threshold="n/a" if threshold is None else f"{threshold:.0%}",
                passed="PASS" if assessment.get("passed") else "FAIL",
                hit=metrics.get("hit_rate", float("nan")),
                ret=metrics.get("valid_return_rate", float("nan")),
                tgt=metrics.get("target_rate", float("nan")),
                safety=robot.get("total_safety_violations", "n/a"),
                energy=robot.get("mean_energy_joule", float("nan")),
            )
        )
    return "\n".join(lines)


def cross_level_robustness_gap(reports: Mapping[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Nominal-minus-perturbed success rate, over every level that was run.

    A single level cannot show it -- the quantity is a comparison between two
    groups of levels -- so it is computed across the sweep rather than inside a
    per-level report.
    """
    episodes = [
        EpisodeResult.from_dict(record)
        for report in reports.values()
        for record in report.get("results", ())
    ]
    return robustness_gap(episodes)


def robustness_line(gap: Mapping[str, Any] | None) -> str:
    """One human-readable sentence about the gap, or about why there is none."""
    if gap is None:
        return (
            "Robustness gap: not measured -- it compares L1-L3 against L4-L5, "
            "so the run has to cover both."
        )
    return (
        f"Robustness gap (`{gap['metric']}`): "
        f"{gap['in_distribution']:.0%} on L1-L3 ({gap['in_distribution_episodes']} episodes) "
        f"minus {gap['perturbed']:.0%} on L4-L5 ({gap['perturbed_episodes']} episodes) "
        f"= **{gap['gap']:.0%}**."
    )


def default_observation_label(track: str) -> str:
    """What the report should say a policy on this track was allowed to read."""
    return "stereo-vision+proprioception" if track == "vision" else "task-observation-33"


def evaluate_levels(
    policy: Any,
    *,
    levels: Sequence[str],
    split: str,
    seed: int = 0,
    policy_id: str,
    track: str = "state",
    observation_label: str | None = None,
    episodes: int | None = None,
    bank: str = TASK.bank_resource,
    out: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run several levels and, when ``out`` is given, write each level's report."""
    unknown = [level for level in levels if level not in VALID_LEVELS]
    if unknown:
        raise ValueError(f"unknown levels: {', '.join(unknown)}")
    reports: dict[str, dict[str, Any]] = {}
    for level in levels:
        report = evaluate_level(
            policy,
            level=level,
            split=split,
            seed=seed,
            policy_id=policy_id,
            observation_label=observation_label or default_observation_label(track),
            episodes=episodes,
            track=track,
            bank=bank,
        )
        reports[level] = report
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            write_report(report, out / f"{split}-{track}-{level}.json")
            (out / f"{split}-{track}-{level}.md").write_text(
                report_markdown(report), encoding="utf-8"
            )
    return reports


__all__ = [
    "TRACKS",
    "PolicyController",
    "cross_level_robustness_gap",
    "default_observation_label",
    "difficulty_table",
    "evaluate_level",
    "evaluate_levels",
    "load_policy",
    "robustness_line",
]
