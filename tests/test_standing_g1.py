"""Standing means a free pelvis supported by contacts, with failures scored."""

import mujoco
import numpy as np
import pytest

from multisport_sim.benchmark.assets import UNITREE_G1, asset_available
from multisport_sim.benchmark.backends.mujoco_standing import MujocoStandingG1TableTennisBackend
from multisport_sim.benchmark.vision import vision_sensors_for

pytestmark = pytest.mark.skipif(not asset_available(UNITREE_G1), reason="G1 asset not installed")

def test_stands_without_base_constraint_or_external_robot_wrench():
    backend = MujocoStandingG1TableTennisBackend()
    joint = backend.model.joint("g1_floating_base_joint")
    assert joint.type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert backend.model.neq == 0
    for _ in range(round(3 / backend.timestep)):
        backend.step()
        assert not backend.safety_violations()
    metrics = backend.balance_metrics
    assert metrics["min_pelvis_height_m"] > 0.85
    assert metrics["max_tilt_rad"] < 0.15
    robot_bodies = [
        i
        for i in range(backend.model.nbody)
        if (mujoco.mj_id2name(backend.model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("g1_")
    ]
    assert np.count_nonzero(backend.data.xfrc_applied[robot_bodies]) == 0
    assert backend.describe()["robot"]["base"].startswith("free pelvis")
    assert backend.mechanical_power_w() >= backend.robot_observation().mechanical_power_w()


def test_fallen_base_is_a_safety_failure_even_with_valid_arm_joints():
    backend = MujocoStandingG1TableTennisBackend()
    joint = backend.model.joint("g1_floating_base_joint")
    address = int(joint.qposadr[0])
    backend.data.qpos[address + 2] = 0.3
    mujoco.mj_forward(backend.model, backend.data)
    assert any(v.kind == "balance" for v in backend.safety_violations())
    backend.reset()
    backend.data.qpos[address + 3 : address + 7] = (np.cos(0.4), np.sin(0.4), 0, 0)
    mujoco.mj_forward(backend.model, backend.data)
    assert any(v.kind == "balance" for v in backend.safety_violations())


@pytest.mark.parametrize("perception", ["stereo", "rgbd"])
def test_standing_vision_environment_contract(perception):
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env

    env = gym.make("MultiSportRobot/TableTennisReturn-G1-Standing-Vision-v2", perception=perception)
    try:
        check_env(env.unwrapped, skip_render_check=True)
        obs, info = env.reset(seed=2)
        assert "ball" not in obs
        assert not info["privileged_ball_state"]
        assert len(env.unwrapped.backend.read_sensors()["balance_imu"].orientation) == 4
        assert env.action_space.shape == (10,)
    finally:
        env.close()


def test_standing_sensor_rig_has_its_own_imu():
    assert "balance_imu" in {s.name for s in vision_sensors_for("g1-standing")}
