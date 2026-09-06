from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from multisport_sim.benchmark.envs import TableTennisReturnEnv
from multisport_sim.benchmark.runner import RunConfig
from multisport_sim.benchmark.task_config import (
    CANONICAL_CONVENTION,
    TABLE_TENNIS_RETURN_PANDA_V1,
    TABLE_TENNIS_RETURN_V0,
    BallObservationLimits,
    CoordinateConvention,
    PaddleWorkspace,
    RewardWeights,
    TableTennisReturnTaskConfig,
    TaskFrame,
)
from multisport_sim.benchmark.types import BallState, EpisodeResult, SemanticContact
from multisport_sim.specs import CAMPUS_OFFSETS, Sport


def test_shared_config_matches_the_published_task_frame() -> None:
    config = TABLE_TENNIS_RETURN_V0

    assert config.env_id == "MultiSportRobot/TableTennisReturn-v0"
    assert config.sport == Sport.TABLE_TENNIS.value
    assert config.frame.is_identity
    assert config.convention == CANONICAL_CONVENTION
    assert config.convention.robot_side_sign == -1
    assert config.convention.opponent_side_sign == 1
    assert config.convention.quaternion_order == "wxyz"
    # The judge and the config must describe the same table, not two copies.
    assert config.table.net_plane_x_m == 0.0
    assert config.table.side_at(-1.0, 0.0) == "robot"
    assert config.table.side_at(1.0, 0.0) == "opponent"


def test_embodied_task_uses_the_statistically_sufficient_bank() -> None:
    assert TABLE_TENNIS_RETURN_PANDA_V1.bank_resource == "table_tennis/return-v1"
    assert TABLE_TENNIS_RETURN_V0.bank_resource == "table_tennis/return-v0"


def test_declared_space_bounds_have_task_dimensions() -> None:
    config = TABLE_TENNIS_RETURN_V0
    action_low, action_high = config.action_bounds()
    observation_low, observation_high = config.observation_bounds()

    assert len(action_low) == len(action_high) == config.ACTION_DIM
    assert len(observation_low) == len(observation_high) == config.OBSERVATION_DIM
    assert all(low < high for low, high in zip(action_low, action_high, strict=True))
    assert all(
        low < high for low, high in zip(observation_low, observation_high, strict=True)
    )
    # The paddle pose observation reuses exactly the action bounds.
    assert observation_low[9:] == action_low
    assert observation_high[9:] == action_high


def test_rate_helpers_derive_decimation_and_step_budget() -> None:
    config = replace(TABLE_TENNIS_RETURN_V0, control_hz=100.0, timeout_s=2.0)

    assert config.decimation(0.001) == 10
    assert config.control_dt(0.001) == pytest.approx(0.01)
    assert config.max_physics_steps(0.001) == 2002
    assert config.max_control_steps(0.001) == 201
    # A physics timestep slower than the control rate still advances one step.
    assert config.decimation(0.05) == 1


@pytest.mark.parametrize("physics_dt", [0.0, -0.001, float("inf"), float("nan")])
def test_rate_helpers_reject_invalid_timesteps(physics_dt: float) -> None:
    with pytest.raises(ValueError):
        TABLE_TENNIS_RETURN_V0.decimation(physics_dt)


def test_invalid_configuration_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        replace(TABLE_TENNIS_RETURN_V0, control_hz=0.0)
    with pytest.raises(ValueError):
        replace(TABLE_TENNIS_RETURN_V0, timeout_s=float("inf"))
    with pytest.raises(ValueError):
        replace(TABLE_TENNIS_RETURN_V0, split=" ")
    with pytest.raises(TypeError):
        replace(TABLE_TENNIS_RETURN_V0, table=object())
    with pytest.raises(ValueError):
        PaddleWorkspace(position_low=(0.0, -1.5, 0.5), position_high=(0.0, 1.5, 1.8))
    with pytest.raises(ValueError):
        BallObservationLimits(linear_velocity_limit=-1.0)
    with pytest.raises(ValueError):
        RewardWeights(valid_return=-1.0)
    with pytest.raises(ValueError):
        CoordinateConvention(length_axis="x", width_axis="x", up_axis="z")
    with pytest.raises(ValueError):
        CoordinateConvention(robot_side_sign=0)


def test_task_frame_round_trips_positions_and_leaves_vectors_alone() -> None:
    frame = TaskFrame.for_campus(Sport.TABLE_TENNIS)

    assert frame.origin_xyz == (*CAMPUS_OFFSETS[Sport.TABLE_TENNIS], 0.0)
    assert not frame.is_identity
    assert frame.to_task_position((-14.0, 48.5, 1.2)) == (1.0, 0.5, 1.2)
    assert frame.to_world_position(frame.to_task_position((-14.0, 48.5, 1.2))) == (
        -14.0,
        48.5,
        1.2,
    )
    # A translation cannot change a velocity or a spin.
    assert frame.to_task_vector((3.0, -1.0, 2.0)) == (3.0, -1.0, 2.0)


def test_task_frame_moves_ball_states_and_contact_samples() -> None:
    frame = TaskFrame.for_campus(Sport.TABLE_TENNIS)
    world_ball = BallState(
        position=(-14.0, 48.0, 1.0),
        linear_velocity=(-5.0, 0.0, 1.0),
        angular_velocity=(0.0, 120.0, 0.0),
    )

    task_ball = frame.to_task_ball(world_ball)
    assert task_ball.position == (1.0, 0.0, 1.0)
    assert task_ball.linear_velocity == world_ball.linear_velocity
    assert task_ball.angular_velocity == world_ball.angular_velocity
    assert frame.to_world_ball(task_ball) == world_ball

    contact = SemanticContact.between(
        "ball", "table", position=(-14.5, 48.2, 0.78), normal=(0.0, 0.0, 1.0)
    )
    task_contact = frame.to_task_contact(contact)
    assert task_contact.pair == contact.pair
    assert task_contact.position == pytest.approx((0.5, 0.2, 0.78))
    assert task_contact.normal == contact.normal


def test_identity_frame_returns_the_same_objects() -> None:
    frame = TaskFrame()
    ball = BallState(
        position=(0.5, 0.0, 1.0), linear_velocity=(1.0, 0.0, 0.0), angular_velocity=(0.0,) * 3
    )
    contact = SemanticContact.between("ball", "net", position=(0.0, 0.0, 0.9))

    assert frame.to_task_ball(ball) is ball
    assert frame.to_task_contact(contact) is contact


def test_rotated_task_frames_are_rejected_rather_than_mishandled() -> None:
    swapped = CoordinateConvention(length_axis="y", width_axis="x", up_axis="z")

    with pytest.raises(ValueError):
        TaskFrame(convention=swapped)


def test_reward_scores_each_event_once() -> None:
    config = TABLE_TENNIS_RETURN_V0
    hit_only = EpisodeResult(shot_id="s", hit=True)
    returned = EpisodeResult(shot_id="s", hit=True, valid_return=True)
    on_target = EpisodeResult(
        shot_id="s", hit=True, valid_return=True, target_hit=True, target_error_m=0.05
    )

    assert config.reward_for(EpisodeResult(shot_id="s")) == 0.0
    assert config.reward_for(hit_only) == 1.0
    assert config.reward_for(hit_only, previously_hit=True) == 0.0
    assert config.reward_for(returned, previously_hit=True) == 3.0
    assert config.reward_for(on_target, previously_hit=True) == 4.0
    with pytest.raises(TypeError):
        config.reward_for(object())  # type: ignore[arg-type]


def test_run_config_carries_the_shared_task_and_its_overrides() -> None:
    default = RunConfig()
    assert default.control_hz == TABLE_TENNIS_RETURN_V0.control_hz
    assert default.timeout_s == TABLE_TENNIS_RETURN_V0.timeout_s
    assert default.task is TABLE_TENNIS_RETURN_V0

    task = replace(TABLE_TENNIS_RETURN_V0, control_hz=50.0, timeout_s=1.5)
    from_task = RunConfig.from_task_config(task, seed=3)
    assert (from_task.seed, from_task.control_hz, from_task.timeout_s) == (3, 50.0, 1.5)

    # An explicit override wins over the task default and reaches the judge.
    overridden = RunConfig(control_hz=25.0, timeout_s=0.5, task=task).effective_task
    assert (overridden.control_hz, overridden.timeout_s) == (25.0, 0.5)
    assert overridden.table is task.table

    with pytest.raises(TypeError):
        RunConfig.from_task_config(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RunConfig(task=object())  # type: ignore[arg-type]


def test_environment_spaces_and_rates_come_from_the_shared_config() -> None:
    env = TableTennisReturnEnv()
    try:
        action_low, action_high = TABLE_TENNIS_RETURN_V0.action_bounds()
        observation_low, _ = TABLE_TENNIS_RETURN_V0.observation_bounds()

        assert env.config is TABLE_TENNIS_RETURN_V0
        assert np.allclose(env.action_space.low, np.array(action_low, dtype=np.float32))
        assert np.allclose(env.action_space.high, np.array(action_high, dtype=np.float32))
        assert np.allclose(
            env.observation_space.low, np.array(observation_low, dtype=np.float32)
        )
        assert env.observation_space.shape == (TABLE_TENNIS_RETURN_V0.OBSERVATION_DIM,)
        assert env.action_space.shape == (TABLE_TENNIS_RETURN_V0.ACTION_DIM,)
        assert env._control_decimation == TABLE_TENNIS_RETURN_V0.decimation(
            env.backend.timestep
        )
        assert env.backend.task_frame == TABLE_TENNIS_RETURN_V0.frame

        _, info = env.reset(seed=0)
        assert info["task_id"] == TABLE_TENNIS_RETURN_V0.task_id
    finally:
        env.close()


def test_environment_honours_config_overrides() -> None:
    env = TableTennisReturnEnv(control_hz=50.0, timeout_s=1.0, split="test")
    try:
        assert env.config.control_hz == 50.0
        assert env.config.timeout_s == 1.0
        assert env.config.split == "test"
        assert env.shot_bank.split == "test"
        assert env._control_decimation == env.config.decimation(env.backend.timestep)
    finally:
        env.close()

    with pytest.raises(ValueError):
        TableTennisReturnEnv(control_hz=0.0)
    with pytest.raises(TypeError):
        TableTennisReturnEnv(config=object())  # type: ignore[arg-type]


def test_serialized_config_is_json_friendly_and_complete() -> None:
    payload = TABLE_TENNIS_RETURN_V0.to_dict()

    assert payload["task_id"] == "table-tennis-return-v0"
    assert payload["action_dim"] == TableTennisReturnTaskConfig.ACTION_DIM
    assert payload["frame"]["origin_xyz"] == [0.0, 0.0, 0.0]
    assert payload["frame"]["convention"]["up_axis"] == "z"
    assert payload["reward"] == {"hit": 1.0, "valid_return": 3.0, "target_hit": 1.0}
    assert payload["table"]["top_height_m"] == pytest.approx(0.76)
