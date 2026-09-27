"""Versioned robot Shot Skill benchmark primitives."""

from .assets import FRANKA_PANDA, AssetSource, asset_available, license_manifest
from .envs import TableTennisReturnEnv, TableTennisReturnPandaEnv, register_envs
from .metrics import aggregate_results, assess_level, build_benchmark_report, robustness_gap
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
from .rules.badminton import BadmintonServeJudge
from .rules.basketball import BasketballShootJudge
from .rules.football import FootballKickJudge
from .rules.launch import LaunchJudge
from .rules.net_return import NetReturnJudge
from .rules.table_tennis import TableTennisReturnJudge
from .rules.tennis import TennisReturnJudge
from .runner import RunConfig, RunOutput, run_shots
from .shot_bank import ShotBank, ShotBankError
from .task_config import (
    BADMINTON_SERVE_V0,
    BASKETBALL_SHOOT_V0,
    FOOTBALL_KICK_V0,
    TABLE_TENNIS_RETURN_G1_V1,
    TABLE_TENNIS_RETURN_PANDA_V1,
    TABLE_TENNIS_RETURN_V0,
    TENNIS_RETURN_V0,
    CoordinateConvention,
    ShotTaskConfig,
    StrikeZone,
    TableTennisReturnPandaTaskConfig,
    TableTennisReturnTaskConfig,
    TaskFrame,
)
from .types import BallState, EpisodeResult, SemanticContact, ShotSpec, TargetSpec
from .wrappers import TargetObservation

register_envs()

__all__ = [
    "BADMINTON_SERVE_V0",
    "BASKETBALL_SHOOT_V0",
    "FOOTBALL_KICK_V0",
    "FRANKA_PANDA",
    "TABLE_TENNIS_RETURN_G1_V1",
    "TABLE_TENNIS_RETURN_PANDA_V1",
    "TABLE_TENNIS_RETURN_V0",
    "TENNIS_RETURN_V0",
    "AssetSource",
    "BadmintonServeJudge",
    "BallState",
    "BasketballShootJudge",
    "ControlMode",
    "CoordinateConvention",
    "EffectorPoseCommand",
    "EpisodeResult",
    "FootballKickJudge",
    "JointCommand",
    "JointLimits",
    "LaunchJudge",
    "NetReturnJudge",
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
    "TargetObservation",
    "TargetSpec",
    "TaskEntry",
    "TaskFrame",
    "TennisReturnJudge",
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
    "robustness_gap",
    "run_shots",
    "task_ids",
]
