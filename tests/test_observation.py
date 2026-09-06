"""The observation contract: named fields, robot-dependent widths."""

from __future__ import annotations

import numpy as np
import pytest

from multisport_sim.benchmark.observation import (
    BALL_ANGULAR_VELOCITY,
    BALL_LINEAR_VELOCITY,
    BALL_POSITION,
    EFFECTOR_LINEAR_VELOCITY,
    EFFECTOR_POSITION,
    EFFECTOR_QUATERNION,
    JOINT_POSITION,
    JOINT_VELOCITY,
    ObservationField,
    ObservationLayout,
    ObservationLayoutError,
    joint_space_layout,
)
from multisport_sim.benchmark.task_config import TABLE_TENNIS_RETURN_PANDA_V1


def layout_for(dof: int) -> ObservationLayout:
    """A layout for an arm with ``dof`` joints and otherwise identical task."""
    return joint_space_layout(
        ball_position_low=(-3.0, -2.0, 0.0),
        ball_position_high=(3.0, 2.0, 3.0),
        linear_velocity_limit=30.0,
        angular_velocity_limit=400.0,
        joint_names=tuple(f"joint{index}" for index in range(dof)),
        joint_position_low=(-2.0,) * dof,
        joint_position_high=(2.0,) * dof,
        joint_velocity_limit=10.0,
        effector_position_low=(-3.0, -1.0, 0.3),
        effector_position_high=(-1.0, 1.0, 2.0),
    )


class TestTheFieldContract:
    def test_a_field_must_carry_one_bound_per_element(self) -> None:
        with pytest.raises(ObservationLayoutError, match="low must hold 3 values"):
            ObservationField("ball.position", 3, (0.0,), (1.0, 2.0, 3.0), "task")

    def test_a_field_may_not_be_inverted(self) -> None:
        with pytest.raises(ObservationLayoutError, match="low must not exceed high"):
            ObservationField("x", 2, (1.0, 0.0), (0.0, 1.0), "task")

    def test_element_names_must_match_the_size_when_present(self) -> None:
        with pytest.raises(ObservationLayoutError, match="element_names"):
            ObservationField(
                "robot.joint_position", 2, (0.0, 0.0), (1.0, 1.0), "robot", ("a",)
            )

    def test_a_layout_rejects_duplicate_fields(self) -> None:
        field = ObservationField("x", 1, (0.0,), (1.0,), "task")
        with pytest.raises(ObservationLayoutError, match="duplicate field"):
            ObservationLayout((field, field))


class TestRobotsOfDifferentWidths:
    def test_the_field_names_do_not_depend_on_the_robot(self) -> None:
        """The task fixes the vocabulary; only the widths move."""
        assert layout_for(6).names == layout_for(7).names == layout_for(12).names

    def test_only_the_joint_blocks_change_size(self) -> None:
        six, twelve = layout_for(6), layout_for(12)
        assert six.size == 6 * 2 + 19
        assert twelve.size == 12 * 2 + 19
        for name in (BALL_POSITION, EFFECTOR_POSITION, EFFECTOR_QUATERNION):
            assert six.field(name).size == twelve.field(name).size

    def test_a_robot_independent_policy_reads_the_same_meaning_on_both(self) -> None:
        """The property that makes a cross-robot comparison possible at all."""
        wanted = (BALL_POSITION, BALL_LINEAR_VELOCITY, EFFECTOR_POSITION)
        six, twelve = layout_for(6), layout_for(12)
        values_six = six.pack(_sample(six))
        values_twelve = twelve.pack(_sample(twelve))
        assert values_six[six.view(wanted)] == pytest.approx(
            values_twelve[twelve.view(wanted)]
        )

    def test_the_robot_independent_set_excludes_the_joint_blocks(self) -> None:
        independent = layout_for(7).robot_independent()
        assert JOINT_POSITION not in independent
        assert JOINT_VELOCITY not in independent
        assert BALL_POSITION in independent

    def test_joint_element_names_are_published_so_offsets_need_no_counting(self) -> None:
        field = layout_for(3).field(JOINT_POSITION)
        assert field.element_names == ("joint0", "joint1", "joint2")


class TestAskingForTheWrongThing:
    def test_an_unknown_field_names_what_is_available(self) -> None:
        layout = layout_for(7)
        with pytest.raises(ObservationLayoutError, match="does not publish"):
            layout.view(("robot.tactile",))

    def test_a_short_block_is_an_error_not_a_reshape(self) -> None:
        layout = layout_for(7)
        values = _sample(layout)
        values[JOINT_POSITION] = (0.0,) * 6
        with pytest.raises(ObservationLayoutError, match="expected 7 values, got 6"):
            layout.pack(values)

    def test_a_missing_block_is_named(self) -> None:
        layout = layout_for(7)
        values = _sample(layout)
        del values[BALL_POSITION]
        with pytest.raises(ObservationLayoutError, match="missing value for field"):
            layout.pack(values)

    def test_an_unknown_block_is_rejected_rather_than_ignored(self) -> None:
        layout = layout_for(7)
        values = _sample(layout)
        values["robot.tactile"] = (0.0,)
        with pytest.raises(ObservationLayoutError, match="unknown fields"):
            layout.pack(values)


class TestThePublishedPandaTask:
    def test_the_released_thirty_three_vector_is_unchanged(self) -> None:
        """The layout replaced a hand-written vector; scores must still mean it."""
        layout = TABLE_TENNIS_RETURN_PANDA_V1.observation_layout()
        assert layout.size == 33
        assert TABLE_TENNIS_RETURN_PANDA_V1.OBSERVATION_DIM == 33
        assert layout.names == (
            BALL_POSITION,
            BALL_LINEAR_VELOCITY,
            BALL_ANGULAR_VELOCITY,
            JOINT_POSITION,
            JOINT_VELOCITY,
            EFFECTOR_POSITION,
            EFFECTOR_QUATERNION,
            EFFECTOR_LINEAR_VELOCITY,
        )

    def test_the_flat_bounds_agree_with_the_declared_spaces(self) -> None:
        config = TABLE_TENNIS_RETURN_PANDA_V1
        low, high = config.observation_bounds()
        assert len(low) == len(high) == config.OBSERVATION_DIM
        joints = config.observation_layout().slice(JOINT_POSITION)
        assert tuple(low[joints]) == config.joint_action.low

    def test_the_action_width_follows_the_robot_not_a_constant(self) -> None:
        config = TABLE_TENNIS_RETURN_PANDA_V1
        assert config.ACTION_DIM == config.joint_action.dof == 7

    def test_the_layout_is_recorded_in_the_report_geometry(self) -> None:
        """A score is auditable only if the reader can see what was observed."""
        payload = TABLE_TENNIS_RETURN_PANDA_V1.rule_geometry()["observation_layout"]
        assert payload["size"] == 33
        names = [field["name"] for field in payload["fields"]]
        assert names[0] == BALL_POSITION
        assert JOINT_POSITION in names
        assert JOINT_POSITION not in payload["robot_independent_fields"]


class TestBuildingFromABackendObservation:
    def test_it_reads_the_duck_typed_state_the_backends_expose(self) -> None:
        layout = TABLE_TENNIS_RETURN_PANDA_V1.observation_layout()
        vector = layout.build(_FakeState(dof=7))
        assert vector.shape == (33,)
        assert vector.dtype == np.float32
        assert vector[layout.slice(BALL_POSITION)] == pytest.approx([0.1, 0.2, 0.3])

    def test_a_state_without_a_robot_is_refused(self) -> None:
        layout = TABLE_TENNIS_RETURN_PANDA_V1.observation_layout()
        with pytest.raises(ObservationLayoutError, match="'ball' and 'robot'"):
            layout.build(object())


def _sample(layout: ObservationLayout) -> dict[str, tuple[float, ...]]:
    """Deterministic per-field values that respect each field's declared size."""
    return {
        field.name: tuple(float(index) for index in range(field.size))
        for field in layout.fields
    }


class _FakeBall:
    position = (0.1, 0.2, 0.3)
    linear_velocity = (1.0, 0.0, 0.0)
    angular_velocity = (0.0, 5.0, 0.0)


class _FakeRobot:
    def __init__(self, dof: int) -> None:
        self.joint_positions = tuple(0.1 * index for index in range(dof))
        self.joint_velocities = tuple(0.0 for _ in range(dof))
        self.effector_position = (-1.5, 0.0, 1.0)
        self.effector_quaternion = (1.0, 0.0, 0.0, 0.0)
        self.effector_linear_velocity = (0.0, 0.0, 0.0)


class _FakeState:
    def __init__(self, *, dof: int) -> None:
        self.ball = _FakeBall()
        self.robot = _FakeRobot(dof)
