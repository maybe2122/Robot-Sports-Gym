"""Free-pelvis G1 with a local ankle balance controller and explicit fall checks.

No base wrench, mocap constraint, or state teleportation is applied during a
rollout. Four ankle servos use ideal pelvis IMU feedback. The high-level policy
still commands the waist and racket arm. This controller stands; it does not walk.
"""

from dataclasses import replace
from math import acos, asin, atan2

import numpy as np

from ..assets import UNITREE_G1
from ..robot import JointLimits, SafetyLimits, SafetyMonitor, SafetyViolation
from ..robots import g1
from ..sensors import ImuSpec
from .mujoco_robot import MujocoG1TableTennisBackend
from .mujoco_sensors import MujocoSensorSuite

BALANCE_IMU = ImuSpec(name="balance_imu", rate_hz=1000.0, mount="g1_pelvis")
STANDING_SOURCE = replace(
    UNITREE_G1,
    modifications=(
        "G1 free pelvis retained; stand keyframe removed before scene attachment",
        "benchmark paddle attached to right wrist; upstream files unchanged",
        "robot stands on a 0.1 m high, 1.0 m diameter support platform",
        "four ankle position servos controlled by ideal pelvis IMU feedback",
        "other non-policy joints held at stand pose; no locomotion policy",
        "scene impratio retained; joint velocity limits remain unverified placeholders",
    ),
)


class StandingG1Adapter(g1.G1TableTennisAdapter):
    robot_id = "unitree-g1-standing-tabletennis-v2"

    def describe(self):
        result = super().describe()
        result["base"] = "free pelvis; ankle-feedback standing; no stepping"
        result["held_joints"] = [name for name in result["held_joints"] if "ankle" not in name]
        result["balance_joints"] = [
            f"g1_{side}_ankle_{axis}_joint"
            for side in ("left", "right")
            for axis in ("pitch", "roll")
        ]
        return result


class MujocoStandingG1TableTennisBackend(MujocoG1TableTennisBackend):
    backend_name = "mujoco-g1-standing"

    def __init__(self, **kwargs):
        kwargs.setdefault("mount", g1.G1Mount(pedestal_radius_m=0.5))
        self._balance_faults = 0
        self.balance_metrics = {
            "max_tilt_rad": 0.0,
            "min_pelvis_height_m": float("inf"),
            "physics_steps": 0,
        }
        super().__init__(**kwargs)
        self._pelvis = self.model.body("g1_pelvis").id
        self.balance_metrics["min_pelvis_height_m"] = float(self.data.xpos[self._pelvis, 2])
        self._imu = MujocoSensorSuite(self.model, self.data, (BALANCE_IMU,))
        self._ankles = [
            (self.model.actuator(f"g1_{side}_ankle_{axis}_joint").id, axis)
            for side in ("left", "right")
            for axis in ("pitch", "roll")
        ]
        names = [name for _, _, name in self.adapter._held_names()]
        joints = [self.model.joint(name).id for name in names]
        self._held_q = np.array([self.model.jnt_qposadr[j] for j in joints])
        self._held_v = np.array([self.model.jnt_dofadr[j] for j in joints])
        self._held_monitor = SafetyMonitor(
            SafetyLimits(
                joints=JointLimits(
                    joint_names=tuple(names),
                    position_low=tuple(self.model.jnt_range[j, 0] for j in joints),
                    position_high=tuple(self.model.jnt_range[j, 1] for j in joints),
                    velocity_limit=(g1.G1_VELOCITY_LIMIT_RAD_S,) * len(joints),
                    torque_limit=tuple(self.model.jnt_actfrcrange[j, 1] for j in joints),
                )
            )
        )
        self._all_v = np.concatenate((self._held_v, self.adapter._dof_adr))

    @staticmethod
    def build_model(*, mount, cameras):
        return g1.build_g1_table_tennis_model(mount=mount, cameras=cameras, fixed_base=False)

    @staticmethod
    def build_adapter(model, data, *, mount, workspace):
        return StandingG1Adapter(
            model, data, mount=mount, workspace=workspace, source=STANDING_SOURCE
        )

    def reset(self):
        super().reset()
        if hasattr(self, "_imu"):
            self._imu.reset()
            self._held_monitor.reset()
        self._balance_faults = 0

    def step(self, action=None):
        imu = self._imu.sample(self.time)[BALANCE_IMU.name]
        w, x, y, z = imu.orientation
        pitch = asin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        roll = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        for actuator, axis in self._ankles:
            angle, omega = (
                (pitch, imu.angular_velocity[1])
                if axis == "pitch"
                else (roll, imu.angular_velocity[0])
            )
            target = np.clip(0.5 * angle + 0.1 * omega, -0.4, 0.4)
            self.data.ctrl[actuator] = np.clip(target, *self.model.actuator_ctrlrange[actuator])
        super().step(action)
        height, tilt = self._balance_state()
        self.balance_metrics["min_pelvis_height_m"] = min(
            height, self.balance_metrics["min_pelvis_height_m"]
        )
        self.balance_metrics["max_tilt_rad"] = max(tilt, self.balance_metrics["max_tilt_rad"])
        self.balance_metrics["physics_steps"] += 1

    def _balance_state(self):
        w, _, _, z = self.data.xquat[self._pelvis]
        return (
            float(self.data.xpos[self._pelvis, 2]),
            acos(float(np.clip(2 * (w * w + z * z) - 1, -1.0, 1.0))),
        )

    def safety_violations(self):
        violations = list(super().safety_violations())
        held = replace(
            self.robot_observation(),
            joint_positions=tuple(self.data.qpos[self._held_q]),
            joint_velocities=tuple(self.data.qvel[self._held_v]),
            applied_torque=tuple(self.data.qfrc_actuator[self._held_v]),
            contacts=(),
        )
        violations.extend(self._held_monitor.check(held))
        height, tilt = self._balance_state()
        if height < self.mount.position[2] + 0.55:
            violations.append(
                SafetyViolation(
                    "balance",
                    "pelvis below standing envelope",
                    height,
                    self.mount.position[2] + 0.55,
                )
            )
        if tilt > 0.6:
            violations.append(
                SafetyViolation("balance", "pelvis tilt above standing envelope", tilt, 0.6)
            )
        self._balance_faults += sum(v.kind == "balance" for v in violations)
        return tuple(violations)

    def mechanical_power_w(self):
        return float(
            np.abs(self.data.qfrc_actuator[self._all_v] * self.data.qvel[self._all_v]).sum()
        )

    @property
    def safety_violation_count(self):
        return self.monitor.count + self._held_monitor.count + self._balance_faults

    def describe(self):
        result = super().describe()
        result["robot"]["balance_controller"] = {
            "id": "g1-ankle-imu-pd-v1",
            "kp": 0.5,
            "kd": 0.1,
            "sensor": BALANCE_IMU.to_dict(),
            "max_tilt_rad": 0.6,
            "min_pelvis_height_m": self.mount.position[2] + 0.55,
            "joint_limits": self._held_monitor.limits.to_dict(),
            "energy_scope": "all 29 robot actuators",
            "measured": dict(self.balance_metrics),
        }
        return result
