"""Concrete benchmark task registrations.

Importing this module publishes every shipped task.  Factories import their
simulator lazily so that listing tasks stays cheap and import-safe.
"""

from __future__ import annotations

from .registry import TaskEntry, register_task
from .rules.base import ShotJudge
from .task_config import (
    TABLE_TENNIS_RETURN_G1_V1,
    TABLE_TENNIS_RETURN_PANDA_V1,
    TABLE_TENNIS_RETURN_STANDING_G1_V2,
    TABLE_TENNIS_RETURN_V0,
    TENNIS_RETURN_V0,
    ShotTaskConfig,
    TableTennisReturnTaskConfig,
    TennisReturnTaskConfig,
)


def _table_tennis_judge(config: ShotTaskConfig) -> ShotJudge:
    from .rules.table_tennis import TABLE_TENNIS, TableTennisReturnJudge

    table = config.table if isinstance(config, TableTennisReturnTaskConfig) else TABLE_TENNIS
    return TableTennisReturnJudge(timeout_s=config.timeout_s, table_spec=table)


def _mocap_backend(config: ShotTaskConfig) -> object:
    """The mocap fixture for whichever sport the config names."""
    from .backends.mujoco import MujocoShotBackend

    return MujocoShotBackend(sport=config.sport)


def _table_tennis_panda_backend(config: ShotTaskConfig) -> object:
    """Build the embodied backend, importing MuJoCo and the asset lazily.

    The asset lives outside the repository, so this raises
    :class:`~multisport_sim.benchmark.assets.AssetUnavailableError` when the
    robot is not installed -- at construction, not at import, so listing tasks
    still works on a machine without it.
    """
    from .backends.mujoco_robot import MujocoPandaTableTennisBackend
    from .robot import WorkspaceBox

    workspace = WorkspaceBox(
        config.workspace.position_low, config.workspace.position_high
    )
    return MujocoPandaTableTennisBackend(workspace=workspace)


TABLE_TENNIS_RETURN = TaskEntry(
    config=TABLE_TENNIS_RETURN_V0,
    judge_factory=_table_tennis_judge,
    backend_factory=_mocap_backend,
    env_entry_point="multisport_sim.benchmark.envs:TableTennisReturnEnv",
)


TABLE_TENNIS_RETURN_PANDA = TaskEntry(
    config=TABLE_TENNIS_RETURN_PANDA_V1,
    judge_factory=_table_tennis_judge,
    backend_factory=_table_tennis_panda_backend,
    env_entry_point="multisport_sim.benchmark.envs:TableTennisReturnPandaEnv",
    vision_env_entry_point=(
        "multisport_sim.benchmark.envs:TableTennisReturnPandaVisionEnv"
    ),
)


def _table_tennis_g1_backend(config: ShotTaskConfig) -> object:
    """The same task on a fixed-base Unitree G1; imported lazily like the Panda."""
    from .backends.mujoco_robot import MujocoG1TableTennisBackend
    from .robot import WorkspaceBox

    workspace = WorkspaceBox(
        config.workspace.position_low, config.workspace.position_high
    )
    return MujocoG1TableTennisBackend(workspace=workspace)


TABLE_TENNIS_RETURN_G1 = TaskEntry(
    config=TABLE_TENNIS_RETURN_G1_V1,
    judge_factory=_table_tennis_judge,
    backend_factory=_table_tennis_g1_backend,
    env_entry_point="multisport_sim.benchmark.envs:TableTennisReturnG1Env",
    vision_env_entry_point="multisport_sim.benchmark.envs:TableTennisReturnG1VisionEnv",
)


def _tennis_judge(config: ShotTaskConfig) -> ShotJudge:
    from .rules.tennis import TENNIS, TennisReturnJudge

    court = config.court if isinstance(config, TennisReturnTaskConfig) else TENNIS
    return TennisReturnJudge(timeout_s=config.timeout_s, court_spec=court)


TENNIS_RETURN = TaskEntry(
    config=TENNIS_RETURN_V0,
    judge_factory=_tennis_judge,
    backend_factory=_mocap_backend,
    env_entry_point="multisport_sim.benchmark.envs:TennisReturnEnv",
)


def _standing_g1_backend(config):
    from .backends.mujoco_standing import MujocoStandingG1TableTennisBackend
    from .robot import WorkspaceBox

    return MujocoStandingG1TableTennisBackend(
        workspace=WorkspaceBox(config.workspace.position_low, config.workspace.position_high))


TABLE_TENNIS_RETURN_STANDING_G1 = TaskEntry(
    config=TABLE_TENNIS_RETURN_STANDING_G1_V2,
    judge_factory=_table_tennis_judge,
    backend_factory=_standing_g1_backend,
    env_entry_point="multisport_sim.benchmark.envs:TableTennisReturnStandingG1Env",
    vision_env_entry_point="multisport_sim.benchmark.envs:TableTennisReturnStandingG1VisionEnv",
)


def register_builtin_tasks() -> None:
    """Register every shipped task; safe to call repeatedly."""
    register_task(TABLE_TENNIS_RETURN, replace=True)
    register_task(TABLE_TENNIS_RETURN_PANDA, replace=True)
    register_task(TABLE_TENNIS_RETURN_G1, replace=True)
    register_task(TABLE_TENNIS_RETURN_STANDING_G1, replace=True)
    register_task(TENNIS_RETURN, replace=True)


register_builtin_tasks()
