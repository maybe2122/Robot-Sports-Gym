"""Versioned robot Shot Skill benchmark primitives."""

from .assets import FRANKA_PANDA, AssetSource, asset_available, license_manifest
from .envs import TableTennisReturnEnv, TableTennisReturnPandaEnv, register_envs
from .metrics import aggregate_results, assess_level, build_benchmark_report
from .registry import TaskEntry, get_task, iter_tasks, register_task, task_ids
from .robot import (
    ControlMode,
    EffectorPoseCommand,
    JointCommand,
    JointLimits,
    RobotAdapter,
    RobotObservation,
    SafetyLimits,
    SafetyViolation,
    WorkspaceBox,
)
from .rules.table_tennis import TableTennisReturnJudge
from .runner import RunConfig, RunOutput, run_shots
from .shot_bank import ShotBank, ShotBankError
from .task_config import (
    TABLE_TENNIS_RETURN_PANDA_V1,
    TABLE_TENNIS_RETURN_V0,
    CoordinateConvention,
    ShotTaskConfig,
    StrikeZone,
    TableTennisReturnPandaTaskConfig,
    TableTennisReturnTaskConfig,
    TaskFrame,
)
from .types import BallState, EpisodeResult, SemanticContact, ShotSpec, TargetSpec

register_envs()

__all__ = [
    "FRANKA_PANDA",
    "TABLE_TENNIS_RETURN_PANDA_V1",
    "TABLE_TENNIS_RETURN_V0",
    "AssetSource",
    "BallState",
    "ControlMode",
    "CoordinateConvention",
    "EffectorPoseCommand",
    "EpisodeResult",
    "JointCommand",
    "JointLimits",
    "RobotAdapter",
    "RobotObservation",
    "RunConfig",
    "RunOutput",
    "SafetyLimits",
    "SafetyViolation",
    "SemanticContact",
    "ShotBank",
    "ShotBankError",
    "ShotSpec",
    "ShotTaskConfig",
    "StrikeZone",
    "TableTennisReturnEnv",
    "TableTennisReturnJudge",
    "TableTennisReturnPandaEnv",
    "TableTennisReturnPandaTaskConfig",
    "TableTennisReturnTaskConfig",
    "TargetSpec",
    "TaskEntry",
    "TaskFrame",
    "WorkspaceBox",
    "aggregate_results",
    "assess_level",
    "asset_available",
    "build_benchmark_report",
    "get_task",
    "iter_tasks",
    "license_manifest",
    "register_envs",
    "register_task",
    "run_shots",
    "task_ids",
]
