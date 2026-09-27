"""The Franka Panda adapter, its scene assembly and its physics calibration.

Every test here needs the Menagerie asset and skips without it.  The point of
these tests is not that MuJoCo integrates -- it is that the assembled scene has
the properties the benchmark depends on: real datasheet limits, a blade that is
part of the kinematic chain, a rubber contact that can actually return a ball,
and commands whose units retain their meaning across control-mode switches.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from multisport_sim.benchmark.assets import FRANKA_PANDA, asset_available
from multisport_sim.benchmark.robot import (
    ControlMode,
    EffectorPoseCommand,
    JointCommand,
    RobotAdapter,
)

pytestmark = pytest.mark.skipif(
    not asset_available(FRANKA_PANDA),
    reason="the Franka Panda asset is not installed; see docs/ROBOT_LAYER.md",
)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    from multisport_sim.benchmark.robots.panda import build_panda_table_tennis_model

    return build_panda_table_tennis_model()


@pytest.fixture
def adapter(model: mujoco.MjModel):
    from multisport_sim.benchmark.robots.panda import PandaTableTennisAdapter

    data = mujoco.MjData(model)
    instance = PandaTableTennisAdapter(model, data)
    instance.reset()
    mujoco.mj_forward(model, data)
    yield instance
    # The adapter owns actuator semantics on its compiled model. This test
    # module deliberately reuses one model for speed, so return it to servo
    # mode before another independent MjData/adapter pair sees it.
    instance.reset()


class TestSceneAssembly:
    def test_the_arm_and_the_ball_share_one_model(self, model: mujoco.MjModel) -> None:
        # Seven arm joints plus the ball's free joint.
        assert model.nq == 14
        assert model.nu == 7

    def test_upstream_joint_ranges_survive_the_unit_conversion(
        self, model: mujoco.MjModel
    ) -> None:
        """The scene is authored in degrees and the arm in radians.

        Attaching one spec to the other must not reinterpret a limit; joint 4's
        asymmetric range is the clearest witness.
        """
        from multisport_sim.benchmark.robots.panda import JOINT_NAMES

        joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES[3]))
        assert model.jnt_range[joint] == pytest.approx([-3.0718, -0.0698], abs=1e-4)

    def test_the_blade_is_part_of_the_kinematic_chain(
        self, model: mujoco.MjModel
    ) -> None:
        """Not a mocap body: it must be a descendant of the last link."""
        from multisport_sim.benchmark.robots.panda import PADDLE_GEOM

        geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PADDLE_GEOM))
        body = int(model.geom_bodyid[geom])
        assert model.body_mocapid[body] < 0
        assert model.body_dofnum[model.body_rootid[body]] == 0  # welded to world
        # Moving a joint must move the blade.
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        before = data.xpos[body].copy()
        data.qpos[model.jnt_qposadr[
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rb_joint1"))
        ]] = 0.5
        mujoco.mj_kinematics(model, data)
        assert not np.allclose(before, data.xpos[body])

    def test_the_blade_site_z_axis_is_the_strike_normal(
        self, model: mujoco.MjModel
    ) -> None:
        """Aiming the paddle must be a single-axis constraint for the solver."""
        from multisport_sim.benchmark.robots.panda import (
            BLADE_HALF_EXTENTS,
            PADDLE_GEOM,
            PADDLE_SITE,
        )

        geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PADDLE_GEOM))
        # The blade is thin along its own y axis.
        assert np.argmin(model.geom_size[geom][:3]) == 1
        assert model.geom_size[geom][:3] == pytest.approx(BLADE_HALF_EXTENTS)

        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        site = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PADDLE_SITE))
        body = int(model.geom_bodyid[geom])
        site_z = data.site_xmat[site].reshape(3, 3)[:, 2]
        body_y = data.xmat[body].reshape(3, 3)[:, 1]
        assert abs(float(site_z @ body_y)) == pytest.approx(1.0, abs=1e-6)


class TestBladeContact:
    def test_the_rubber_returns_a_ball_instead_of_absorbing_it(self) -> None:
        """A drop test on a fixed blade, against measured inverted rubber.

        Without a dedicated contact pair the blade inherits the scene's default
        damping ratio and returns the ball at e ~= 0.17 -- no achievable blade
        speed would then produce a legal return, so this number is load-bearing.
        """
        from multisport_sim.benchmark.robots.panda import (
            BLADE_CONTACT_FRICTION,
            BLADE_CONTACT_SOLIMP,
            BLADE_CONTACT_SOLREF,
        )

        friction = " ".join(str(value) for value in BLADE_CONTACT_FRICTION)
        solref = " ".join(str(value) for value in BLADE_CONTACT_SOLREF)
        solimp = " ".join(str(value) for value in BLADE_CONTACT_SOLIMP)
        xml = f"""<mujoco>
          <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"
                  cone="elliptic" iterations="80"/>
          <default><geom solref="0.006 0.7" solimp="0.94 0.99 0.001"
                         friction="0.7 0.01 0.001" condim="4"/></default>
          <worldbody>
            <geom name="blade" type="box" pos="0 0 0" size="0.2 0.2 0.01"/>
            <body name="ball" pos="0 0 0.31"><freejoint name="bj"/>
              <geom name="ball_geom" type="sphere" size="0.02" mass="0.0027"/></body>
          </worldbody>
          <contact><pair name="p" geom1="blade" geom2="ball_geom" condim="6"
                         friction="{friction}" solref="{solref}" solimp="{solimp}"/></contact>
        </mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        incoming: float | None = None
        previous = 0.0
        restitution: float | None = None
        for _ in range(6000):
            mujoco.mj_step(model, data)
            if data.ncon and incoming is None:
                incoming = abs(previous)
            if data.ncon == 0 and incoming is not None and data.qvel[2] > 0:
                restitution = float(data.qvel[2] / incoming)
                break
            previous = float(data.qvel[2])
        assert restitution is not None
        assert 0.78 <= restitution <= 0.92, restitution


class TestAdapter:
    def test_it_satisfies_the_backend_neutral_protocol(self, adapter) -> None:
        assert isinstance(adapter, RobotAdapter)
        assert adapter.dof == 7
        assert len(adapter.joint_names) == adapter.dof
        assert adapter.effector_name == "racket"

    def test_the_datasheet_limits_are_published_not_guessed(self, adapter) -> None:
        """MJCF carries no velocity limit; the benchmark must supply one."""
        from multisport_sim.benchmark.robots.panda import (
            PANDA_TORQUE_LIMIT,
            PANDA_VELOCITY_LIMIT,
        )

        limits = adapter.joint_limits
        assert limits.velocity_limit == PANDA_VELOCITY_LIMIT
        assert limits.torque_limit == PANDA_TORQUE_LIMIT

    def test_reset_is_deterministic_and_starts_at_rest(self, adapter, model) -> None:
        adapter.apply(JointCommand((0.0,) * 7, ControlMode.JOINT_POSITION))
        for _ in range(50):
            mujoco.mj_step(model, adapter.data)
        adapter.reset()
        mujoco.mj_forward(model, adapter.data)
        state = adapter.observe()
        assert state.joint_positions == pytest.approx(adapter.ready_qpos)
        assert state.joint_velocities == pytest.approx((0.0,) * 7)

    def test_the_ready_pose_is_not_already_in_collision(self, adapter, model) -> None:
        adapter.reset()
        mujoco.mj_forward(model, adapter.data)
        assert adapter.robot_contacts() == ()

    def test_a_position_command_is_clamped_to_the_joint_range(self, adapter) -> None:
        adapter.apply(JointCommand((99.0,) * 7, ControlMode.JOINT_POSITION))
        limits = adapter.joint_limits
        applied = adapter.data.ctrl[list(adapter._actuator_ids)]
        assert applied == pytest.approx(limits.position_high)

    def test_a_wrongly_sized_command_is_refused(self, adapter) -> None:
        with pytest.raises(ValueError, match="7 joints"):
            adapter.apply(JointCommand((0.0, 0.0), ControlMode.JOINT_POSITION))

    def test_a_torque_command_is_applied_in_nm_and_clamped(self, adapter, model) -> None:
        assert ControlMode.JOINT_TORQUE in adapter.control_modes
        requested = (20.0, -20.0, 5.0, -5.0, 99.0, -99.0, 3.0)

        adapter.apply(JointCommand(requested, ControlMode.JOINT_TORQUE))
        mujoco.mj_forward(model, adapter.data)

        expected = (20.0, -20.0, 5.0, -5.0, 12.0, -12.0, 3.0)
        assert adapter.data.ctrl[list(adapter._actuator_ids)] == pytest.approx(expected)
        assert adapter.observe().applied_torque == pytest.approx(expected)

    def test_switching_from_torque_restores_the_position_servo(self, adapter, model) -> None:
        actuator_ids = np.asarray(adapter._actuator_ids)
        original_gain = model.actuator_gainprm[actuator_ids].copy()
        original_bias = model.actuator_biasprm[actuator_ids].copy()
        original_range = model.actuator_ctrlrange[actuator_ids].copy()

        adapter.apply(JointCommand((1.0,) * 7, ControlMode.JOINT_TORQUE))
        assert model.actuator_gainprm[actuator_ids, 0] == pytest.approx((1.0,) * 7)
        assert model.actuator_biasprm[actuator_ids] == pytest.approx(0.0)

        adapter.apply(JointCommand(adapter.ready_qpos, ControlMode.JOINT_POSITION))
        assert model.actuator_gainprm[actuator_ids] == pytest.approx(original_gain)
        assert model.actuator_biasprm[actuator_ids] == pytest.approx(original_bias)
        assert model.actuator_ctrlrange[actuator_ids] == pytest.approx(original_range)

    def test_reset_leaves_torque_mode_and_holds_the_ready_pose(self, adapter, model) -> None:
        adapter.apply(JointCommand((1.0,) * 7, ControlMode.JOINT_TORQUE))

        adapter.reset()
        mujoco.mj_forward(model, adapter.data)

        assert adapter._active_control_mode is ControlMode.JOINT_POSITION
        assert adapter.observe().joint_positions == pytest.approx(adapter.ready_qpos)

    def test_a_velocity_command_cannot_exceed_the_datasheet_speed(
        self, adapter
    ) -> None:
        """The MJCF has position actuators only; the request must be bounded."""
        before = np.asarray(adapter.observe().joint_positions)
        adapter.apply(JointCommand((100.0,) * 7, ControlMode.JOINT_VELOCITY))
        applied = np.asarray(adapter.data.ctrl[list(adapter._actuator_ids)])
        limits = np.asarray(adapter.joint_limits.velocity_limit)
        timestep = float(adapter.model.opt.timestep)
        assert np.all(np.abs(applied - before) <= limits * timestep + 1e-9)

    def test_an_effector_pose_command_reaches_the_requested_point(
        self, adapter, model
    ) -> None:
        state = adapter.observe()
        target = np.asarray(state.effector_position) + np.array([0.0, 0.05, 0.05])
        adapter.apply(EffectorPoseCommand(tuple(target), state.effector_quaternion))
        for _ in range(400):
            mujoco.mj_step(model, adapter.data)
        reached = np.asarray(adapter.observe().effector_position)
        assert float(np.linalg.norm(reached - target)) < 0.03

    def test_a_foreign_command_type_is_refused(self, adapter) -> None:
        with pytest.raises(TypeError, match="JointCommand"):
            adapter.apply(object())

    def test_describe_carries_asset_provenance_into_reports(self, adapter) -> None:
        payload = adapter.describe()
        assert payload["asset"]["license"] == "Apache-2.0"
        assert payload["dof"] == 7
        assert "joint_position" in payload["control_modes"]
        assert payload["safety_limits"]["joints"]["velocity_limit"]


class TestReachability:
    """The properties the frozen constants were calibrated to hold."""

    @pytest.fixture(scope="class")
    def dev_crossings(self) -> list[tuple[float, float]]:
        """Where a sample of dev shots actually pass the strike plane."""
        from multisport_sim.benchmark.backends.mujoco import MujocoShotBackend
        from multisport_sim.benchmark.shot_bank import ShotBank
        from multisport_sim.benchmark.task_config import TABLE_TENNIS_RETURN_PANDA_V1

        config = TABLE_TENNIS_RETURN_PANDA_V1
        plane_x = config.strike_zone.plane_x_m
        backend = MujocoShotBackend()
        crossings: list[tuple[float, float]] = []
        shots = list(ShotBank.from_resource(split="dev", task=config.bank_resource))
        for shot in shots[::10]:
            backend.reset()
            backend.launch_ball(shot)
            previous_x = None
            for _ in range(4000):
                backend.step()
                x, y, z = backend.get_ball_state().position
                if previous_x is not None and previous_x >= plane_x > x and z > 0.55:
                    crossings.append((float(y), float(z)))
                    break
                previous_x = x
                if z < 0.2:
                    break
        assert len(crossings) >= 25
        return crossings

    def test_the_declared_zone_contains_the_measured_crossings(self, dev_crossings) -> None:
        from multisport_sim.benchmark.task_config import TABLE_TENNIS_RETURN_PANDA_V1

        zone = TABLE_TENNIS_RETURN_PANDA_V1.strike_zone
        outside = [
            (y, z)
            for y, z in dev_crossings
            if not (zone.y_low <= y <= zone.y_high and zone.z_low <= z <= zone.z_high)
        ]
        assert not outside

    def test_the_solver_covers_the_measured_crossings(self, model, dev_crossings) -> None:
        """IK reaches the balls themselves, not the corners of their bounding box."""
        from multisport_sim.benchmark.robots.kinematics import IKSolver
        from multisport_sim.benchmark.robots.panda import (
            IK_SETTINGS,
            JOINT_NAMES,
            PADDLE_SITE,
            PANDA_READY_QPOS,
        )
        from multisport_sim.benchmark.task_config import TABLE_TENNIS_RETURN_PANDA_V1

        plane_x = TABLE_TENNIS_RETURN_PANDA_V1.strike_zone.plane_x_m
        solver = IKSolver(
            model, site_name=PADDLE_SITE, joint_names=JOINT_NAMES, **IK_SETTINGS
        )
        ready = np.asarray(PANDA_READY_QPOS)
        converged = sum(
            solver.solve(
                (plane_x, y, z), target_axis=(1.0, 0.0, 0.0), initial_qpos=ready
            ).converged
            for y, z in dev_crossings
        )
        assert converged == len(dev_crossings)

    def test_an_unreachable_target_stops_at_its_fixed_point(self, model) -> None:
        """A stalled solve returns early with the configuration a full budget reaches."""
        from multisport_sim.benchmark.robots.kinematics import IKSolver
        from multisport_sim.benchmark.robots.panda import (
            IK_SETTINGS,
            JOINT_NAMES,
            PADDLE_SITE,
            PANDA_READY_QPOS,
        )

        target = (-0.80, 0.0, 1.10)  # past the table edge, out of the arm's reach
        ready = np.asarray(PANDA_READY_QPOS)
        kwargs = {"target_axis": (1.0, 0.0, 0.0), "initial_qpos": ready}
        early = IKSolver(
            model, site_name=PADDLE_SITE, joint_names=JOINT_NAMES, **IK_SETTINGS
        ).solve(target, **kwargs)
        full = IKSolver(
            model,
            site_name=PADDLE_SITE,
            joint_names=JOINT_NAMES,
            stall_step_rad=0.0,
            **IK_SETTINGS,
        ).solve(target, **kwargs)
        assert not early.converged and not full.converged
        assert full.iterations == IK_SETTINGS["max_iterations"]
        assert early.iterations < full.iterations
        assert np.max(np.abs(early.qpos - full.qpos)) < 1e-4

    def test_axis_only_aiming_costs_less_travel_than_a_full_pose(self, model) -> None:
        """Constraining the spin about the blade normal wastes the redundancy."""
        from multisport_sim.benchmark.robots.kinematics import IKSolver, blade_quaternion
        from multisport_sim.benchmark.robots.panda import (
            IK_SETTINGS,
            JOINT_NAMES,
            PADDLE_SITE,
            PANDA_READY_QPOS,
        )

        solver = IKSolver(
            model, site_name=PADDLE_SITE, joint_names=JOINT_NAMES, **IK_SETTINGS
        )
        ready = np.asarray(PANDA_READY_QPOS)
        target = (-1.55, -0.20, 1.05)
        axis = solver.solve(target, target_axis=(1.0, 0.0, 0.0), initial_qpos=ready)
        pose = solver.solve(
            target, blade_quaternion((1.0, 0.0, 0.0)), initial_qpos=ready
        )
        assert axis.converged
        assert axis.position_error_m <= max(pose.position_error_m, 2e-3)

    def test_the_solver_never_leaves_the_joint_range(self, model) -> None:
        from multisport_sim.benchmark.robots.kinematics import IKSolver
        from multisport_sim.benchmark.robots.panda import (
            IK_SETTINGS,
            JOINT_NAMES,
            PADDLE_SITE,
        )

        solver = IKSolver(
            model, site_name=PADDLE_SITE, joint_names=JOINT_NAMES, **IK_SETTINGS
        )
        # Deliberately unreachable: the answer must still be commandable.
        result = solver.solve((5.0, 5.0, 5.0), target_axis=(1.0, 0.0, 0.0))
        assert not result.converged
        assert np.all(result.qpos >= solver.lower - 1e-9)
        assert np.all(result.qpos <= solver.upper + 1e-9)
