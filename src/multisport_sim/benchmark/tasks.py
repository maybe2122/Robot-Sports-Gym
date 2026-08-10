"""Concrete benchmark task registrations.

Importing this module publishes every shipped task.  Factories import their
simulator lazily so that listing tasks stays cheap and import-safe.
"""

from __future__ import annotations

from .registry import TaskEntry, register_task
from .rules.base import ShotJudge
from .task_config import TABLE_TENNIS_RETURN_V0, ShotTaskConfig, TableTennisReturnTaskConfig


def _table_tennis_judge(config: ShotTaskConfig) -> ShotJudge:
    from .rules.table_tennis import TABLE_TENNIS, TableTennisReturnJudge

    table = config.table if isinstance(config, TableTennisReturnTaskConfig) else TABLE_TENNIS
    return TableTennisReturnJudge(timeout_s=config.timeout_s, table_spec=table)


def _table_tennis_backend(config: ShotTaskConfig) -> object:
    from .backends.mujoco import MujocoShotBackend

    return MujocoShotBackend(sport=config.sport)


TABLE_TENNIS_RETURN = TaskEntry(
    config=TABLE_TENNIS_RETURN_V0,
    judge_factory=_table_tennis_judge,
    backend_factory=_table_tennis_backend,
    env_entry_point="multisport_sim.benchmark.envs:TableTennisReturnEnv",
)


def register_builtin_tasks() -> None:
    """Register every shipped task; safe to call repeatedly."""
    register_task(TABLE_TENNIS_RETURN, replace=True)


register_builtin_tasks()
