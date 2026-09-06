"""The tennis return task: the second sport on the same framework.

The point of these tests is not that tennis works -- it is that tennis is the
*same task*.  The judge, the level thresholds, the report schema and the
environment contract are the ones table tennis already uses; what differs is
geometry, speed and scale.  Anywhere that stopped being true, the framework
would have grown a second benchmark instead of a second sport.
"""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from multisport_sim.benchmark import registry
from multisport_sim.benchmark.backends.mujoco import PROFILES, MujocoShotBackend
from multisport_sim.benchmark.controllers import scripted_controller_for
from multisport_sim.benchmark.rules.net_return import NetReturnJudge
from multisport_sim.benchmark.rules.tennis import (
    NET_HEIGHT_M,
    SINGLES_WIDTH_M,
    TENNIS,
    TennisReturnJudge,
)
from multisport_sim.benchmark.runner import RunConfig, run_shots
from multisport_sim.benchmark.shot_bank import ShotBank
from multisport_sim.benchmark.task_config import TENNIS_RETURN_V0
from multisport_sim.benchmark.tasks import register_builtin_tasks
from multisport_sim.scene import BENCHMARK_EFFECTORS, build_model, build_xml
from multisport_sim.specs import Sport

register_builtin_tasks()


def test_the_court_is_the_regulation_singles_court() -> None:
    assert TENNIS.length_m == pytest.approx(23.77)
    assert TENNIS.width_m == pytest.approx(SINGLES_WIDTH_M)
    # The court is the ground, so a landing is at z = 0 rather than on furniture.
    assert TENNIS.top_height_m == 0.0
    assert TENNIS.net_height_m == pytest.approx(NET_HEIGHT_M)
    assert TENNIS.side_at(-5.0, 0.0) == "robot"
    assert TENNIS.side_at(5.0, 0.0) == "opponent"
    # The doubles alleys are a different game.
    assert TENNIS.side_at(5.0, 4.5) is None


def test_the_judge_is_the_shared_engine_with_a_different_surface() -> None:
    judge = TennisReturnJudge()

    assert isinstance(judge, NetReturnJudge)
    assert judge.sport == Sport.TENNIS.value
    assert judge.court_spec is TENNIS
    assert judge.surface is TENNIS


def test_the_task_reuses_every_shared_threshold() -> None:
    """A sport that redefined the levels would be a different benchmark."""
    tennis = ShotBank.from_resource(split="dev", task="tennis/return-v0").manifest
    table_tennis = ShotBank.from_resource(split="dev").manifest

    assert tennis["levels"] == table_tennis["levels"]
    assert tennis["bucket_groups"] == table_tennis["bucket_groups"]
    assert tennis["failure_reasons"] == table_tennis["failure_reasons"]
    assert tennis["episode"]["success_sequence"] == table_tennis["episode"]["success_sequence"]
    # What legitimately differs: the sport, the geometry and the episode length.
    assert tennis["sport"] == "tennis"
    assert tennis["episode"]["timeout_s"] == 3.0
    assert tennis["coordinate_system"]["table_bounds_m"]["x"] == [-11.885, 11.885]


def test_the_benchmark_racket_exists_only_when_it_is_asked_for() -> None:
    assert "tennis_benchmark_paddle" not in build_xml("tennis")
    assert "tennis_benchmark_paddle" in build_xml("tennis", benchmark_paddle=True)

    model = build_model("tennis", benchmark_paddle=True)
    assert model.nmocap == 1


def test_the_string_bed_has_its_own_calibrated_contact() -> None:
    """Inheriting the scene default would return the ball at e = 0.2."""
    effector = BENCHMARK_EFFECTORS[Sport.TENNIS]

    assert effector.contact_solref is not None
    assert "tennis_benchmark_paddle_contact" in build_xml("tennis", benchmark_paddle=True)
    # Table tennis keeps the contact its frozen v0 scores were measured with.
    assert BENCHMARK_EFFECTORS[Sport.TABLE_TENNIS].contact_solref is None
    assert "table_tennis_benchmark_paddle_contact" not in build_xml(
        "table_tennis", benchmark_paddle=True
    )


def test_the_racket_is_parked_inside_its_own_action_space() -> None:
    """An effector starting outside its action space makes reset() invalid."""
    parked = BENCHMARK_EFFECTORS[Sport.TENNIS].parked_at
    workspace = TENNIS_RETURN_V0.workspace

    assert all(
        low <= value <= high
        for value, low, high in zip(parked, workspace.position_low, workspace.position_high)
    )


def test_the_sport_profile_maps_the_court_as_the_playing_surface() -> None:
    semantic = PROFILES[Sport.TENNIS].semantic_geom_names()

    assert semantic["tennis_benchmark_paddle_blade"] == "robot_racket"
    assert semantic["tennis_surface"] == "table"
    assert semantic["tennis_net"] == "net"
    # The line markings are part of the same surface, not a separate object.
    assert semantic["tennis_singles_line_n"] == "table"


def test_the_task_is_registered_and_addressable() -> None:
    entry = registry.get_task("tennis-return-v0")

    assert entry.config is TENNIS_RETURN_V0
    assert entry.env_entry_point.endswith(":TennisReturnEnv")
    assert isinstance(entry.make_judge(), TennisReturnJudge)
    assert isinstance(entry.make_backend(), MujocoShotBackend)


def test_the_environment_matches_the_shared_contract() -> None:
    from gymnasium.utils.env_checker import check_env

    from multisport_sim.benchmark import register_envs

    register_envs()
    env = gym.make("MultiSportRobot/TennisReturn-v0").unwrapped
    try:
        check_env(env, skip_render_check=True)
        observation, info = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        assert info["task_id"] == "tennis-return-v0"
        assert observation.shape == (TENNIS_RETURN_V0.OBSERVATION_DIM,)
    finally:
        env.close()


def test_the_scripted_fixture_intercepts_and_returns_real_tennis_shots() -> None:
    """The fixture is a diagnostic, but a diagnostic that never returns is useless."""
    bank = ShotBank.from_resource(split="dev", task="tennis/return-v0")
    backend = MujocoShotBackend(sport="tennis")
    controller = scripted_controller_for("tennis")
    task = replace(TENNIS_RETURN_V0, split="dev")

    intercept = run_shots(
        backend, controller, bank.filter(level="L1")[:12], config=RunConfig.from_task_config(task)
    )
    ret = run_shots(
        backend, controller, bank.filter(level="L2")[:12], config=RunConfig.from_task_config(task)
    )

    assert all(result.incoming_valid for result in intercept.results)
    assert sum(result.hit for result in intercept.results) >= 11
    assert sum(result.valid_return for result in ret.results) >= 6
    outgoing = [r.outgoing_speed_mps for r in ret.results if r.outgoing_speed_mps is not None]
    # A returned tennis ball leaves at rally speeds, not at table-tennis speeds.
    assert np.median(outgoing) > 8.0


def test_a_sport_without_a_tuned_fixture_is_refused() -> None:
    with pytest.raises(ValueError, match="no scripted fixture is tuned"):
        scripted_controller_for("badminton")


def test_the_two_sports_do_not_share_a_shot_bank() -> None:
    tennis = ShotBank.from_resource(split="test", task="tennis/return-v0")
    table_tennis = ShotBank.from_resource(split="test")

    assert tennis.digest != table_tennis.digest
    assert all(shot.sport == "tennis" for shot in tennis)
    assert all(shot.position[0] > 10.0 for shot in tennis)
