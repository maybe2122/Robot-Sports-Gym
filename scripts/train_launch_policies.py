#!/usr/bin/env python3
"""Train and evaluate the learned primitive baselines (M5).

For one fixture task, trains a PPO policy over the swing primitive of
``multisport_sim.benchmark.learning`` on the *train* split, one run per seed,
then scores every seed on every level of the *test* split through the same
report path as every other baseline.  A ``random-primitive`` row -- uniformly
random primitive parameters -- is scored alongside, so the learned rows can be
read against what the primitive gives for free.

    PYTHONPATH=src python scripts/train_launch_policies.py --sport football \\
        [--seeds 0 1 2 3 4] [--episodes 3000] [--workers 5]

Writes ``baselines/learned/<sport>/seed<N>.zip`` (policy weights),
``seed<N>.json`` (training curve, wall time, hardware) and
``reports/learned-<sport>-baselines.{json,md}``.  Needs the ``train`` extra
(stable-baselines3, torch).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import numpy as np

LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def _train(job: tuple[str, int, int, str]) -> dict:
    sport, seed, episodes, out_dir = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from multisport_sim.benchmark.learning import PrimitiveLaunchEnv

    torch.set_num_threads(1)

    class Curve(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.rewards: list[float] = []

        def _on_step(self) -> bool:
            self.rewards.extend(float(r) for r in self.locals["rewards"])
            return True

    env = PrimitiveLaunchEnv(sport)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=128,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.0,
        gae_lambda=1.0,
        ent_coef=0.0,
        policy_kwargs={"net_arch": [64, 64], "log_std_init": -0.5},
        seed=seed,
        device="cpu",
        verbose=0,
    )
    curve = Curve()
    started = time.perf_counter()
    model.learn(total_timesteps=episodes, callback=curve)
    wall = time.perf_counter() - started
    destination = Path(out_dir) / sport
    destination.mkdir(parents=True, exist_ok=True)
    weights = destination / f"seed{seed}.zip"
    model.save(weights)
    window = 128
    blocks = [
        mean(curve.rewards[index : index + window])
        for index in range(0, len(curve.rewards), window)
        if curve.rewards[index : index + window]
    ]
    record = {
        "sport": sport,
        "seed": seed,
        "episodes": episodes,
        "algorithm": "PPO (stable-baselines3), MLP 64x64, one decision per episode",
        "hyperparameters": {
            "n_steps": 256,
            "batch_size": 128,
            "n_epochs": 10,
            "learning_rate": 3e-4,
            "gamma": 0.0,
            "ent_coef": 0.0,
            "net_arch": [64, 64],
            "log_std_init": -0.5,
        },
        "train_split": "train",
        "train_levels": ["L1", "L2", "L3", "L4"],
        "reward": "0.1*hit + valid_return + target_hit + 0.5*exp(-(target_error/radius)^2) on success",
        "mean_reward_per_128_episodes": blocks,
        "wall_time_s": wall,
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "weights": str(weights),
    }
    (destination / f"seed{seed}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


class _MidPrimitive:
    """Every parameter at the middle of its range: the untrained policy's mean.

    PPO starts from a zero-mean Gaussian, so this is what the deterministic
    policy does before any learning.  When a primitive's mid-range happens to
    sit near a good swing (tennis: 15 m/s, 15 degrees, trigger at -5.5 m, close
    to the hand-tuned fixture), a learned row that merely matches it shows the
    prior, not learning.
    """

    def predict(self, features, deterministic: bool = True):
        del features, deterministic
        return np.zeros(3, dtype=np.float32), None


class _RandomPrimitive:
    """Uniform primitive parameters: the floor a learned policy must beat."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def predict(self, features, deterministic: bool = True):
        del features, deterministic
        return self.rng.uniform(-1.0, 1.0, size=3).astype(np.float32), None


def _request(sport: str, level: str, seed: int) -> argparse.Namespace:
    from multisport_sim.benchmark.learning import RETURN_SPECS

    # Score on the bank the policy trained on: the table-tennis fixture
    # defaults to the tiny frozen return-v0, but learns on return-v1.
    bank = RETURN_SPECS[sport].bank if sport in RETURN_SPECS else None
    return argparse.Namespace(
        sport=sport, task=None, backend="mujoco", level=level, split="test", bank=bank,
        shot_bank=None, episodes=None, seed=seed, robot="none", track="state",
        controller="scripted", control_hz=200.0, report=None, markdown=None,
        require_pass=False, learned_policy=None,
    )


def _evaluate(job: tuple[str, str, str | None, int]) -> dict:
    sport, name, weights, seed = job
    from multisport_sim.benchmark.learning import (
        LearnedLaunchController,
        load_learned_controller,
    )
    from multisport_sim.benchmark_cli import run_from_args

    rows = {}
    for level in LEVELS:
        if name == "untrained-primitive":
            controller = LearnedLaunchController(
                sport, _MidPrimitive(), policy_id="untrained-primitive"
            )
        elif weights is None:
            controller = LearnedLaunchController(
                sport, _RandomPrimitive(seed), policy_id="random-primitive"
            )
        else:
            controller = load_learned_controller(sport, weights)
        report = run_from_args(_request(sport, level, 0), controller=controller)
        assessment = report["assessment"] or {}
        rows[level] = {
            "primary_metric": assessment.get("primary_metric"),
            "primary_value": assessment.get("primary_value"),
            "passed": assessment.get("passed"),
            "hit_rate": report["metrics"]["hit_rate"],
            "valid_return_rate": report["metrics"]["valid_return_rate"],
            "target_rate": report["metrics"]["target_rate"],
            "episodes": report["episodes"],
        }
    return {"name": name, "seed": seed, "weights": weights, "levels": rows}


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci95": None}
    if len(values) == 1:
        return {"mean": values[0], "ci95": None}
    half = T_975.get(len(values) - 1, 1.96) * stdev(values) / sqrt(len(values))
    return {"mean": mean(values), "ci95": [mean(values) - half, mean(values) + half]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sport",
        choices=("badminton", "football", "basketball", "tennis", "table_tennis"),
        required=True,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("baselines/learned"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args(argv)

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.skip_training:
            training = [
                json.loads((args.out_dir / args.sport / f"seed{seed}.json").read_text())
                for seed in args.seeds
            ]
        else:
            training = list(
                pool.map(
                    _train,
                    [(args.sport, seed, args.episodes, str(args.out_dir)) for seed in args.seeds],
                )
            )
        jobs = [(args.sport, "learned-primitive-ppo", record["weights"], record["seed"])
                for record in training]
        jobs.append((args.sport, "random-primitive", None, 0))
        jobs.append((args.sport, "untrained-primitive", None, 0))
        evaluations = list(pool.map(_evaluate, jobs))

    learned = [item for item in evaluations if item["name"] == "learned-primitive-ppo"]
    random_row = next(item for item in evaluations if item["name"] == "random-primitive")
    untrained_row = next(item for item in evaluations if item["name"] == "untrained-primitive")
    aggregate = {
        level: {
            "primary_metric": learned[0]["levels"][level]["primary_metric"],
            "primary_value": _summary(
                [item["levels"][level]["primary_value"] or 0.0 for item in learned]
            ),
            "valid_return_rate": _summary(
                [item["levels"][level]["valid_return_rate"] for item in learned]
            ),
            "seeds_passed": sum(bool(item["levels"][level]["passed"]) for item in learned),
        }
        for level in LEVELS
    }
    payload = {
        "schema": "multisport-learned-baseline-v0",
        "sport": args.sport,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_split": "test",
        "seeds": args.seeds,
        "training": training,
        "per_seed": learned,
        "random_primitive": random_row,
        "untrained_primitive": untrained_row,
        "aggregate": aggregate,
    }
    args.reports.mkdir(parents=True, exist_ok=True)
    stem = args.reports / f"learned-{args.sport}-baselines"
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    stem.with_suffix(".md").write_text(markdown(payload))
    print(markdown(payload))
    return 0


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def markdown(payload: dict) -> str:
    sport = payload["sport"]
    training = payload["training"]
    lines = [
        f"# Learned baseline: {sport}",
        "",
        (f"PPO over the swing primitive (`multisport_sim.benchmark.learning`), "
        f"{len(training)} seeds x {training[0]['episodes']} training episodes on the train "
        "split (L1-L4), scored on every level of the test split. Regenerate with "
        f"`python scripts/train_launch_policies.py --sport {sport}`."),
        "",
        ("The policy sees observable geometry only (contact distance and height, ball "
        "velocity, target) and knows nothing the simulator was calibrated with. "
        "`untrained-primitive` sets every parameter to the middle of its range (the untrained "
        "policy's mean); `random-primitive` draws them uniformly."),
        "",
        ("| Level | Metric | Learned (mean over seeds) | 95% CI (t) | Seeds passing | "
        "Untrained primitive | Random primitive |"),
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for level in LEVELS:
        entry = payload["aggregate"][level]
        ci = entry["primary_value"]["ci95"]
        lines.append(
            f"| {level} | `{entry['primary_metric']}` | {_pct(entry['primary_value']['mean'])} | "
            f"{'—' if ci is None else f'{max(ci[0], 0):.0%}–{min(ci[1], 1):.0%}'} | "
            f"{entry['seeds_passed']}/{len(payload['per_seed'])} | "
            f"{_pct(payload['untrained_primitive']['levels'][level]['primary_value'])} | "
            f"{_pct(payload['random_primitive']['levels'][level]['primary_value'])} |"
        )
    lines += ["", "## Training", "", "| Seed | Wall time | First 128 | Last 128 |", "|---:|---:|---:|---:|"]
    for record in training:
        curve = record["mean_reward_per_128_episodes"]
        lines.append(
            f"| {record['seed']} | {record['wall_time_s'] / 60:.1f} min | "
            f"{curve[0]:.2f} | {curve[-1]:.2f} |"
        )
    lines += [
        "",
        (f"Hardware: {training[0]['hardware']}. Reward per episode: "
        f"`{training[0]['reward']}`."),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
