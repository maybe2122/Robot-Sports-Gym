from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import pytest

from multisport_sim.benchmark import registry
from multisport_sim.benchmark.backends.mujoco import PROFILES, MujocoShotBackend
from multisport_sim.benchmark.rules.base import RectangularSurface, ShotJudge
from multisport_sim.benchmark.rules.net_return import NetReturnJudge
from multisport_sim.benchmark.rules.table_tennis import TABLE_TENNIS, TableTennisReturnJudge
from multisport_sim.benchmark.task_config import TABLE_TENNIS_RETURN_V0
from multisport_sim.benchmark.tasks import TABLE_TENNIS_RETURN, register_builtin_tasks
from multisport_sim.benchmark.types import BallState, SemanticContact, ShotSpec
from multisport_sim.specs import Sport

register_builtin_tasks()


def test_builtin_task_is_registered_and_addressable() -> None:
    assert "table-tennis-return-v0" in registry.task_ids()

    entry = registry.get_task("table-tennis-return-v0")
    assert entry.config is TABLE_TENNIS_RETURN_V0
    assert entry.env_entry_point.endswith(":TableTennisReturnEnv")

    with pytest.raises(KeyError, match="unknown task"):
        registry.get_task("no-such-task-v0")
    with pytest.raises(KeyError, match="no task is registered"):
        registry.task_for_sport("curling")


def test_a_sport_with_several_embodiments_must_be_addressed_by_task_id() -> None:
    """Table tennis is played by three tasks, so the sport names none of them.

    A mocap fixture, a Franka Panda and a fixed-base Unitree G1.  They share a
    judge, a shot bank and a set of thresholds; what they do not share is a
    score, which is why resolving by sport has to fail.
    """
    entries = registry.tasks_for_sport(Sport.TABLE_TENNIS.value)

    assert [entry.task_id for entry in entries] == [
        "table-tennis-return-g1-standing-v2",
        "table-tennis-return-g1-v1",
        "table-tennis-return-panda-v1",
        "table-tennis-return-v0",
    ]
    with pytest.raises(KeyError, match="table-tennis-return-panda-v1"):
        registry.task_for_sport(Sport.TABLE_TENNIS.value)
    with pytest.raises(KeyError, match="no task is registered"):
        registry.tasks_for_sport("curling")


def test_registered_env_ids_come_from_the_registry() -> None:
    from multisport_sim.benchmark import register_envs

    register_envs()
    for entry in registry.iter_tasks():
        assert entry.config.env_id in gym.registry


def test_entry_factories_build_a_matching_judge_and_backend() -> None:
    entry = registry.get_task("table-tennis-return-v0")

    judge = entry.make_judge()
    assert isinstance(judge, ShotJudge)
    assert isinstance(judge, TableTennisReturnJudge)

    # An override reaches the judge instead of being silently dropped.
    slow = entry.make_judge(replace(entry.config, timeout_s=0.5))
    assert slow.timeout_s == 0.5

    backend = entry.make_backend()
    assert isinstance(backend, MujocoShotBackend)
    assert backend.sport is Sport.TABLE_TENNIS


def test_registering_a_conflicting_task_is_refused() -> None:
    conflicting = replace(
        TABLE_TENNIS_RETURN, env_entry_point="multisport_sim.benchmark.envs:Other"
    )

    with pytest.raises(ValueError, match="already registered"):
        registry.register_task(conflicting)

    # Re-registering the identical entry is a no-op, so imports stay idempotent.
    assert registry.register_task(TABLE_TENNIS_RETURN) is TABLE_TENNIS_RETURN
    assert registry.get_task("table-tennis-return-v0") is TABLE_TENNIS_RETURN


def test_mujoco_profile_names_follow_the_scene_convention() -> None:
    profile = PROFILES[Sport.TABLE_TENNIS]

    assert profile.ball_joint == "table_tennis_ball_free"
    assert profile.ball_geom == "table_tennis_ball_geom"
    semantic = profile.semantic_geom_names()
    assert semantic[profile.effector_geom] == "robot_racket"
    assert semantic["table_tennis_table"] == "table"
    assert semantic["table_tennis_net"] == "net"
    # Every sport's own surface and the campus ground are floor by convention.
    assert semantic["table_tennis_surface"] == "floor"
    assert semantic["campus_ground"] == "floor"


def test_unprofiled_sport_is_rejected_with_a_clear_error() -> None:
    with pytest.raises(ValueError, match="no MuJoCo benchmark profile"):
        MujocoShotBackend(sport=Sport.SQUASH)


def test_net_return_judge_scores_any_surface_not_only_a_table() -> None:
    """A half-size practice court proves the engine is not table-tennis-only."""
    court = RectangularSurface(length_m=4.0, width_m=2.0, top_height_m=0.0)
    judge = NetReturnJudge(sport="tennis", surface=court, timeout_s=2.0)
    shot = ShotSpec(
        shot_id="mini-0001",
        sport="tennis",
        level="L2",
        position=(1.5, 0.0, 0.6),
        linear_velocity=(-6.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )
    judge.reset(shot)

    def step(time_s: float, x: float, vx: float, contacts=()) -> None:
        judge.update(
            time_s=time_s,
            ball=BallState(
                position=(x, 0.0, 0.05),
                linear_velocity=(vx, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
            ),
            contacts=contacts,
        )

    bounce = SemanticContact.between("ball", "table", position=(-1.0, 0.0, 0.0))
    strike = SemanticContact.between("ball", "robot_racket")
    landing = SemanticContact.between("ball", "table", position=(1.2, 0.0, 0.0))

    step(0.00, 1.5, -6.0)
    step(0.05, 0.5, -6.0)
    step(0.10, -1.0, -6.0, (bounce,))
    step(0.15, -1.4, -6.0)
    step(0.20, -1.5, 6.0, (strike,))
    step(0.25, -0.5, 6.0)
    step(0.30, 0.8, 6.0)
    step(0.35, 1.2, 6.0, (landing,))

    result = judge.result
    assert judge.done
    assert (result.incoming_valid, result.hit, result.crossed_net) == (True, True, True)
    assert result.valid_return
    assert result.failure_reason is None


def test_table_tennis_judge_is_the_shared_engine_with_regulation_geometry() -> None:
    judge = TableTennisReturnJudge()

    assert isinstance(judge, NetReturnJudge)
    assert judge.sport == Sport.TABLE_TENNIS.value
    assert judge.table_spec is TABLE_TENNIS
    assert judge.surface is TABLE_TENNIS
    assert TABLE_TENNIS.side_at(-1.0, 0.0) == "robot"
    assert TABLE_TENNIS.side_at(1.0, 0.0) == "opponent"
    assert TABLE_TENNIS.side_at(0.0, 5.0) is None
    # Legacy attribute names still resolve for downstream integrations.
    assert TABLE_TENNIS.tabletop_height_tolerance_m == TABLE_TENNIS.height_tolerance_m
    assert TABLE_TENNIS.minimum_tabletop_normal_z == TABLE_TENNIS.minimum_normal_z
