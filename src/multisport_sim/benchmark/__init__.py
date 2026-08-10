"""Versioned robot Shot Skill benchmark primitives."""

from .envs import TableTennisReturnEnv, register_envs
from .metrics import aggregate_results, assess_level, build_benchmark_report
from .registry import TaskEntry, get_task, iter_tasks, register_task, task_ids
from .rules.table_tennis import TableTennisReturnJudge
from .runner import RunConfig, RunOutput, run_shots
from .shot_bank import ShotBank, ShotBankError
from .task_config import (
    TABLE_TENNIS_RETURN_V0,
    CoordinateConvention,
    ShotTaskConfig,
    TableTennisReturnTaskConfig,
    TaskFrame,
)
from .types import BallState, EpisodeResult, SemanticContact, ShotSpec, TargetSpec

register_envs()

__all__ = [
    "TABLE_TENNIS_RETURN_V0",
    "BallState",
    "CoordinateConvention",
    "EpisodeResult",
    "RunConfig",
    "RunOutput",
    "SemanticContact",
    "ShotBank",
    "ShotBankError",
    "ShotSpec",
    "ShotTaskConfig",
    "TableTennisReturnEnv",
    "TableTennisReturnJudge",
    "TableTennisReturnTaskConfig",
    "TargetSpec",
    "TaskEntry",
    "TaskFrame",
    "aggregate_results",
    "assess_level",
    "build_benchmark_report",
    "get_task",
    "iter_tasks",
    "register_envs",
    "register_task",
    "run_shots",
    "task_ids",
]
