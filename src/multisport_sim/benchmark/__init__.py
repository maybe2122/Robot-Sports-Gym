"""Versioned robot Shot Skill benchmark primitives."""

from .metrics import aggregate_results, assess_level, build_benchmark_report
from .rules.table_tennis import TableTennisReturnJudge
from .runner import RunConfig, RunOutput, run_shots
from .shot_bank import ShotBank, ShotBankError
from .types import BallState, EpisodeResult, SemanticContact, ShotSpec, TargetSpec

__all__ = [
    "BallState",
    "EpisodeResult",
    "RunConfig",
    "RunOutput",
    "SemanticContact",
    "ShotBank",
    "ShotBankError",
    "ShotSpec",
    "TableTennisReturnJudge",
    "TargetSpec",
    "aggregate_results",
    "assess_level",
    "build_benchmark_report",
    "run_shots",
]
