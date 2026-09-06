"""Contracts of the backend-neutral robot layer.

Nothing here imports a simulator or needs a robot asset: these are the types a
task is allowed to know about, and they must be testable without either.
"""

from __future__ import annotations

import pytest

from multisport_sim.benchmark.robot import (
    ControlMode,
    EffectorPoseCommand,
    JointCommand,
    JointLimits,
    RobotObservation,
    SafetyLimits,
    SafetyMonitor,
    SafetyViolation,
    UnsupportedControlMode,
    WorkspaceBox,
)

JOINTS = ("j1", "j2", "j3")


def limits(**overrides: object) -> JointLimits:
    values: dict[str, object] = {
        "joint_names": JOINTS,
        "position_low": (-1.0, -1.0, -1.0),
        "position_high": (1.0, 1.0, 1.0),
        "velocity_limit": (2.0, 2.0, 2.0),
        "torque_limit": (10.0, 10.0, 10.0),
    }
    values.update(overrides)
    return JointLimits(**values)  # type: ignore[arg-type]


def observation(**overrides: object) -> RobotObservation:
    values: dict[str, object] = {
        "time_s": 0.5,
        "joint_positions": (0.0, 0.0, 0.0),
        "joint_velocities": (0.0, 0.0, 0.0),
        "applied_torque": (0.0, 0.0, 0.0),
        "effector_position": (-1.5, 0.0, 1.0),
        "effector_quaternion": (1.0, 0.0, 0.0, 0.0),
    }
    values.update(overrides)
    return RobotObservation(**values)  # type: ignore[arg-type]


class TestCommands:
    def test_joint_command_rejects_an_effector_mode(self) -> None:
        with pytest.raises(ValueError, match="joint-space"):
            JointCommand((0.0, 0.0), ControlMode.EFFECTOR_POSE)

    def test_joint_command_rejects_non_finite_targets(self) -> None:
        with pytest.raises(ValueError):
            JointCommand((0.0, float("nan")))

    def test_effector_command_normalizes_its_quaternion(self) -> None:
        command = EffectorPoseCommand((0.0, 0.0, 1.0), (2.0, 0.0, 0.0, 0.0))
        assert command.quaternion == (1.0, 0.0, 0.0, 0.0)

    def test_effector_command_rejects_a_degenerate_quaternion(self) -> None:
        with pytest.raises(ValueError, match="non-zero norm"):
            EffectorPoseCommand((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0))

    def test_unsupported_mode_names_what_is_supported(self) -> None:
        error = UnsupportedControlMode(
            "arm", ControlMode.JOINT_TORQUE, {ControlMode.JOINT_POSITION}
        )
        assert "joint_torque" in str(error)
        assert "joint_position" in str(error)


class TestObservation:
    def test_vector_lengths_must_agree(self) -> None:
        with pytest.raises(ValueError, match="joint_velocities"):
            observation(joint_velocities=(0.0, 0.0))

    def test_mechanical_power_charges_for_braking_too(self) -> None:
        # Opposite signs: the joint is braking against its own motion, which
        # costs energy rather than returning it.
        state = observation(
            joint_velocities=(1.0, -2.0, 0.0), applied_torque=(3.0, 4.0, 5.0)
        )
        assert state.mechanical_power_w() == pytest.approx(3.0 + 8.0)


class TestJointLimits:
    def test_clamp_keeps_commands_inside_the_range(self) -> None:
        assert limits().clamp_positions((5.0, -5.0, 0.25)) == (1.0, -1.0, 0.25)

    def test_clamp_rejects_the_wrong_number_of_joints(self) -> None:
        with pytest.raises(ValueError, match="exactly 3"):
            limits().clamp_positions((0.0, 0.0))

    def test_velocity_and_torque_limits_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="velocity_limit"):
            limits(velocity_limit=(2.0, 0.0, 2.0))


class TestWorkspaceBox:
    def test_excess_is_zero_inside_and_positive_outside(self) -> None:
        box = WorkspaceBox((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0))
        assert box.excess((0.0, 0.0, 1.0)) == 0.0
        assert box.excess((1.25, 0.0, 1.0)) == pytest.approx(0.25)

    def test_excess_reports_the_worst_axis(self) -> None:
        box = WorkspaceBox((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0))
        assert box.excess((1.1, -1.4, 1.0)) == pytest.approx(0.4)


class TestSafetyLimits:
    """The four violation classes the benchmark specification requires."""

    def test_a_compliant_observation_reports_nothing(self) -> None:
        assert SafetyLimits(joints=limits()).violations(observation()) == ()

    def test_joint_position_violation(self) -> None:
        found = SafetyLimits(joints=limits()).violations(
            observation(joint_positions=(0.0, 1.5, 0.0))
        )
        assert [item.kind for item in found] == ["joint_position"]
        assert found[0].joint_index == 1

    def test_joint_velocity_violation(self) -> None:
        found = SafetyLimits(joints=limits()).violations(
            observation(joint_velocities=(0.0, 0.0, -3.0))
        )
        assert [item.kind for item in found] == ["joint_velocity"]
        assert found[0].value == pytest.approx(3.0)

    def test_joint_torque_violation(self) -> None:
        found = SafetyLimits(joints=limits()).violations(
            observation(applied_torque=(20.0, 0.0, 0.0))
        )
        assert [item.kind for item in found] == ["joint_torque"]

    def test_collision_violation_only_for_forbidden_labels(self) -> None:
        safety = SafetyLimits(joints=limits(), forbidden_contacts=frozenset({"table"}))
        assert safety.violations(observation(contacts=("ball",))) == ()
        found = safety.violations(observation(contacts=("table",)))
        assert [item.kind for item in found] == ["collision"]

    def test_workspace_violation(self) -> None:
        safety = SafetyLimits(
            joints=limits(), workspace=WorkspaceBox((-2.0, -1.0, 0.5), (-1.0, 1.0, 1.5))
        )
        found = safety.violations(observation(effector_position=(-0.5, 0.0, 1.0)))
        assert [item.kind for item in found] == ["workspace"]

    def test_tolerances_absorb_a_joint_resting_on_its_stop(self) -> None:
        """A simulator resolves a limit over several steps; that is not a breach."""
        safety = SafetyLimits(joints=limits(), position_tolerance_rad=0.02)
        assert safety.violations(observation(joint_positions=(1.01, 0.0, 0.0))) == ()
        assert safety.violations(observation(joint_positions=(1.05, 0.0, 0.0)))

    def test_every_class_is_reported_when_several_break_at_once(self) -> None:
        safety = SafetyLimits(
            joints=limits(), workspace=WorkspaceBox((-2.0, -1.0, 0.5), (-1.0, 1.0, 1.5))
        )
        found = safety.violations(
            observation(
                joint_positions=(2.0, 0.0, 0.0),
                joint_velocities=(9.0, 0.0, 0.0),
                applied_torque=(99.0, 0.0, 0.0),
                effector_position=(5.0, 0.0, 1.0),
                contacts=("table",),
            )
        )
        assert [item.kind for item in found] == [
            "joint_position",
            "joint_velocity",
            "joint_torque",
            "collision",
            "workspace",
        ]

    def test_a_mismatched_observation_is_an_error_not_a_pass(self) -> None:
        with pytest.raises(ValueError, match="declare"):
            SafetyLimits(joints=limits()).violations(
                observation(
                    joint_positions=(0.0, 0.0),
                    joint_velocities=(0.0, 0.0),
                    applied_torque=(0.0, 0.0),
                )
            )


class TestSafetyMonitor:
    def test_it_accumulates_across_steps_and_resets(self) -> None:
        monitor = SafetyMonitor(SafetyLimits(joints=limits()))
        assert not monitor.triggered
        monitor.check(observation(joint_velocities=(9.0, 0.0, 0.0)))
        monitor.check(observation(applied_torque=(99.0, 0.0, 0.0)))
        assert monitor.count == 2
        assert monitor.triggered
        monitor.reset()
        assert monitor.count == 0


class TestSafetyViolation:
    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown violation kind"):
            SafetyViolation(kind="vibes", detail="d", value=1.0, limit=0.0)

    def test_it_serializes_for_reports(self) -> None:
        payload = SafetyViolation(
            kind="collision", detail="robot touched table", value=1.0, limit=0.0
        ).to_dict()
        assert payload["kind"] == "collision"
        assert payload["joint_index"] is None
