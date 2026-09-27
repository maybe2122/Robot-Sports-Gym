"""The Unitree G1 embodiment: scene assembly, adapter contract and task wiring.

Every test here needs the Menagerie asset and skips without it.

The G1 is the benchmark's second embodiment, so most of what these tests pin is
not "the G1 works" but "the task did not have to change for it to".  Where a
test asserts something *about* the G1 specifically -- the fixed base, the ten
commanded joints, the shoulder bound -- it is asserting a declared scope
decision, and the docstring says which one.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from multisport_sim.benchmark.assets import ASSETS, UNITREE_G1, asset_available
from multisport_sim.benchmark.observation import JOINT_POSITION
from multisport_sim.benchmark.robot import ControlMode, JointCommand, RobotAdapter
from multisport_sim.benchmark.task_config import (
    TABLE_TENNIS_RETURN_G1_V1,
    TABLE_TENNIS_RETURN_PANDA_V1,
)

pytestmark = pytest.mark.skipif(
    not asset_available(UNITREE_G1),
    reason="the Unitree G1 asset is not installed; see docs/ROBOT_LAYER.md",
)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    from multisport_sim.benchmark.robots.g1 import build_g1_table_tennis_model

    return build_g1_table_tennis_model()


@pytest.fixture
def adapter(model: mujoco.MjModel):
    from multisport_sim.benchmark.robots.g1 import G1TableTennisAdapter

    data = mujoco.MjData(model)
    built = G1TableTennisAdapter(model, data)
    built.reset()
    mujoco.mj_forward(model, data)
    return built


class TestTheAssetRecord:
    def test_it_is_declared_with_its_licence(self) -> None:
        assert ASSETS[UNITREE_G1.asset_id] is UNITREE_G1
        assert UNITREE_G1.license_id == "BSD-3-Clause"

    def test_every_local_modification_is_listed(self) -> None:
        """The base and the keyframe are edits; a report has to carry them."""
        text = " ".join(UNITREE_G1.modifications)
        assert "free joint is deleted" in text
        assert "keyframe" in text
        assert "paddle" in text
        # The speed envelope is the one number that is not from the asset, and
        # saying so is the whole reason the entry is honest.
        assert "velocity" in text and "unverified" in text


class TestTheAssembledScene:
    def test_the_base_is_fixed_so_the_task_scores_a_return_not_balance(
        self, model: mujoco.MjModel
    ) -> None:
        free = [
            joint
            for joint in range(model.njnt)
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
            and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or "").startswith(
                "g1_"
            )
        ]
        assert not free

    def test_the_ball_still_has_its_own_free_joint(self, model: mujoco.MjModel) -> None:
        """Deleting the robot's base must not touch the thing being hit."""
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "table_tennis_ball_free")
        assert joint >= 0 or any(
            model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE
            for index in range(model.njnt)
        )

    def test_the_blade_is_part_of_the_kinematic_chain(self, model: mujoco.MjModel) -> None:
        from multisport_sim.benchmark.robots.g1 import PADDLE_GEOM, WRIST_BODY

        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PADDLE_GEOM)
        assert geom >= 0
        body = int(model.geom_bodyid[geom])
        parent = int(model.body_parentid[body])
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent) == f"g1_{WRIST_BODY}"

    def test_the_blade_carries_a_real_paddle_mass(self, model: mujoco.MjModel) -> None:
        from multisport_sim.benchmark.robots.g1 import PADDLE_GEOM

        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PADDLE_GEOM)
        assert float(model.body_mass[int(model.geom_bodyid[geom])]) == pytest.approx(0.17, abs=1e-3)

    def test_the_ball_blade_contact_matches_the_panda_exactly(
        self, model: mujoco.MjModel
    ) -> None:
        """Two robots whose blades bounced differently are not playing one task."""
        from multisport_sim.benchmark.robots import panda
        from multisport_sim.benchmark.robots.g1 import (
            BLADE_CONTACT_SOLIMP,
            BLADE_CONTACT_SOLREF,
        )

        assert BLADE_CONTACT_SOLREF == panda.BLADE_CONTACT_SOLREF
        assert BLADE_CONTACT_SOLIMP == panda.BLADE_CONTACT_SOLIMP
        pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, "robot_paddle_ball_contact")
        assert pair >= 0


class TestTheAdapterContract:
    def test_it_satisfies_the_backend_agnostic_protocol(self, adapter) -> None:
        assert isinstance(adapter, RobotAdapter)

    def test_it_commands_the_waist_and_the_right_arm_only(self, adapter) -> None:
        """A declared scope decision: nineteen held joints are not actions."""
        assert adapter.dof == 10
        assert [name.removeprefix("g1_") for name in adapter.joint_names] == [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        assert len(adapter.describe()["held_joints"]) == 19

    def test_the_torque_limits_come_from_the_asset(self, adapter) -> None:
        """25 N.m at the shoulder and elbow, 5 at the wrist, 88 at the waist."""
        limits = adapter.joint_limits.torque_limit
        assert limits[0] == pytest.approx(88.0)
        assert limits[3:8] == pytest.approx((25.0,) * 5)
        assert limits[8:] == pytest.approx((5.0, 5.0))

    def test_the_speed_envelope_is_labelled_as_unverified(self, adapter) -> None:
        """Unitree publishes torque but no per-joint speed; the report says so."""
        described = adapter.describe()
        assert described["velocity_limit_source"] == "unverified_placeholder"
        assert set(adapter.joint_limits.velocity_limit) == {
            described["velocity_limit_rad_s"]
        }

    def test_applied_torque_is_what_the_joint_received_not_what_the_servo_asked_for(
        self, model: mujoco.MjModel, adapter
    ) -> None:
        """Regression: the asset limits force on the JOINTS, not the actuators.

        MuJoCo enforces ``actuatorfrcrange`` in ``qfrc_actuator`` and leaves
        ``actuator_force`` unclamped.  Reading the latter reported wrist torques
        of 250 N.m against a 5 N.m limit and failed every episode on a safety
        violation that never physically happened.
        """
        low, high = TABLE_TENNIS_RETURN_G1_V1.action_bounds()
        target = np.clip(np.asarray(adapter.ready_qpos) + 0.5, low, high)
        adapter.apply(JointCommand(tuple(target), ControlMode.JOINT_POSITION))
        for _ in range(3):
            mujoco.mj_step(model, adapter.data)
        observed = np.abs(adapter.observe().applied_torque)
        limits = np.asarray(adapter.joint_limits.torque_limit)
        assert np.all(observed <= limits + 1e-6)

    def test_the_ready_pose_touches_nothing(self, adapter) -> None:
        """Self-contact is scored, so the pose it is scored against must be clean.

        If the frozen pose ever rests an arm on the torso, every episode would
        carry a permanent collision violation and this test is what says so.
        """
        assert adapter.observe().contacts == ()

    def test_a_command_of_the_wrong_width_is_refused(self, adapter) -> None:
        with pytest.raises(ValueError, match="10 joints"):
            adapter.apply(JointCommand((0.0,) * 7, ControlMode.JOINT_POSITION))


class TestTheShoulderBound:
    def test_the_solver_is_held_in_abduction(self, model: mujoco.MjModel) -> None:
        from multisport_sim.benchmark.robots.g1 import (
            JOINT_NAMES,
            SHOULDER_ROLL_SOLVER_MAX_RAD,
            solver_position_bounds,
        )

        _, high = solver_position_bounds(model)
        index = JOINT_NAMES.index("g1_right_shoulder_roll_joint")
        assert high[index] == pytest.approx(SHOULDER_ROLL_SOLVER_MAX_RAD)

    def test_the_published_action_space_is_not_narrowed(self) -> None:
        """The contract stays the hardware range; self-collision is scored.

        Narrowing the action space to keep a policy out of the folding branch
        would hide a failure mode the benchmark exists to measure.
        """
        from multisport_sim.benchmark.robots.g1 import SHOULDER_ROLL_SOLVER_MAX_RAD

        names = TABLE_TENNIS_RETURN_G1_V1.joint_action.joint_names
        index = names.index("g1_right_shoulder_roll_joint")
        published = TABLE_TENNIS_RETURN_G1_V1.joint_action.high[index]
        assert published > SHOULDER_ROLL_SOLVER_MAX_RAD
        assert published == pytest.approx(1.5882)

    def test_the_published_range_matches_the_compiled_model(
        self, model: mujoco.MjModel
    ) -> None:
        """The config restates the asset's limits; drift would be silent."""
        action = TABLE_TENNIS_RETURN_G1_V1.joint_action
        for index, name in enumerate(action.joint_names):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert joint >= 0, name
            assert float(model.jnt_range[joint][0]) == pytest.approx(
                action.low[index], abs=1e-4
            )
            assert float(model.jnt_range[joint][1]) == pytest.approx(
                action.high[index], abs=1e-4
            )


class TestTheTaskItPlays:
    def test_it_shares_the_bank_the_judge_and_the_thresholds_with_the_panda(self) -> None:
        panda, g1 = TABLE_TENNIS_RETURN_PANDA_V1, TABLE_TENNIS_RETURN_G1_V1
        assert g1.bank_resource == panda.bank_resource
        assert g1.sport == panda.sport
        assert g1.timeout_s == panda.timeout_s
        assert g1.control_hz == panda.control_hz
        assert g1.table == panda.table

    def test_it_is_a_separate_task_id_and_environment(self) -> None:
        assert TABLE_TENNIS_RETURN_G1_V1.task_id == "table-tennis-return-g1-v1"
        assert TABLE_TENNIS_RETURN_G1_V1.env_id.endswith("TableTennisReturn-G1-v1")
        assert TABLE_TENNIS_RETURN_G1_V1.task_id != TABLE_TENNIS_RETURN_PANDA_V1.task_id

    def test_it_registers_the_vision_track(self) -> None:
        import gymnasium as gym

        assert TABLE_TENNIS_RETURN_G1_V1.vision_env_id in gym.registry

    def test_the_observation_differs_only_in_the_widths_of_the_robot_blocks(self) -> None:
        """The claim the second embodiment exists to test."""
        panda = TABLE_TENNIS_RETURN_PANDA_V1.observation_layout()
        g1 = TABLE_TENNIS_RETURN_G1_V1.observation_layout()
        assert panda.names == g1.names
        assert panda.size == 33
        assert g1.size == 39
        assert g1.field(JOINT_POSITION).size == 10
        for name in panda.robot_independent():
            assert panda.field(name).size == g1.field(name).size

    def test_a_robot_independent_view_lines_up_across_both(self) -> None:
        wanted = TABLE_TENNIS_RETURN_PANDA_V1.observation_layout().robot_independent()
        panda_view = TABLE_TENNIS_RETURN_PANDA_V1.observation_layout().view(wanted)
        g1_view = TABLE_TENNIS_RETURN_G1_V1.observation_layout().view(wanted)
        assert len(panda_view) == len(g1_view)

    def test_it_is_registered_with_the_same_judge_as_the_panda(self) -> None:
        from multisport_sim.benchmark.registry import get_task

        entry = get_task("table-tennis-return-g1-v1")
        judge = entry.make_judge(TABLE_TENNIS_RETURN_G1_V1)
        assert judge.__class__ is get_task("table-tennis-return-panda-v1").make_judge(
            TABLE_TENNIS_RETURN_PANDA_V1
        ).__class__


class TestTheBaselines:
    def test_the_scripted_swing_is_named_after_the_robot_that_ran_it(self) -> None:
        from multisport_sim.benchmark.robot_controllers import (
            G1_SWING,
            ScriptedInterceptController,
        )

        control_dt = 1.0 / TABLE_TENNIS_RETURN_G1_V1.control_hz
        assert (
            ScriptedInterceptController(control_dt=control_dt, robot=G1_SWING).controller_id
            == "scripted-g1-intercept-v1"
        )
        assert (
            ScriptedInterceptController(control_dt=control_dt).controller_id
            == "scripted-panda-intercept-v1"
        )

    def test_the_vision_baseline_keeps_its_own_name(self) -> None:
        """Regression: a class attribute here is shadowed by the instance one."""
        from multisport_sim.benchmark.robot_controllers import VisionInterceptController

        controller = VisionInterceptController(
            control_dt=1.0 / TABLE_TENNIS_RETURN_PANDA_V1.control_hz
        )
        assert controller.controller_id == "vision-panda-intercept-v1"

    def test_this_robot_gets_its_own_setpoint_rate_margin(self) -> None:
        """A property of the servo, not of the swing: measured, not inherited."""
        from multisport_sim.benchmark.robot_controllers import G1_SWING, PANDA_SWING

        assert G1_SWING.rate_margin < PANDA_SWING.rate_margin


def test_the_environment_matches_the_shared_contract() -> None:
    """The whole point: a second robot needed no new environment logic."""
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env

    from multisport_sim.benchmark.envs import register_envs

    register_envs()
    env = gym.make(TABLE_TENNIS_RETURN_G1_V1.env_id).unwrapped
    try:
        assert env.action_space.shape == (10,)
        assert env.observation_space.shape == (39,)
        check_env(env, skip_render_check=True)
    finally:
        env.close()
