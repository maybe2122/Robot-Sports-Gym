"""Versioned robot Shot Skill benchmark primitives."""

from .envs import TableTennisReturnEnv, register_envs
from .metrics import aggregate_results, assess_level, build_benchmark_report
from .rules.table_tennis import TableTennisReturnJudge
from .runner import RunConfig, RunOutput, run_shots
from .shot_bank import ShotBank, ShotBankError
from .task_config import (
    TABLE_TENNIS_RETURN_V0,
    CoordinateConvention,
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
    "TableTennisReturnEnv",
    "TableTennisReturnJudge",
    "TableTennisReturnTaskConfig",
    "TargetSpec",
    "TaskFrame",
    "aggregate_results",
    "assess_level",
    "build_benchmark_report",
    "register_envs",
    "run_shots",
]
