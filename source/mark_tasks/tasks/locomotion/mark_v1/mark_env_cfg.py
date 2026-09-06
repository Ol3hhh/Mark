"""Flat velocity-tracking env for mark_v1 

    Robot/Humanoid/Body/Body_upper       (rigid body)
    Robot/Humanoid/Body/Body_lower       (rigid body; joined to Body_upper via Spine)
    Robot/Humanoid/Left_leg/Left_hip     (rigid body)
    Robot/Humanoid/Left_leg/Left_shin    (rigid body)
    Robot/Humanoid/Left_leg/Left_foot    (rigid body)
    Robot/Humanoid/RIght_leg/Right_hip   (rigid body; note asset typo "RIght_leg")
    Robot/Humanoid/RIght_leg/Right_shin  (rigid body)
    Robot/Humanoid/RIght_leg/Right_foot  (rigid body)
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as base_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from mark_tasks.robots.mark_v1 import MARK_V1_CFG

from . import mdp as mark_mdp



@configclass
class MarkRewardsCfg(RewardsCfg):
    termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-50.0)
    lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=base_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_ang_vel_z_exp = None
    feet_air_time = RewTerm(
        func=mark_mdp.feet_air_time_positive_biped_split,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.6,
            "left_sensor_cfg": SceneEntityCfg("left_foot_contact"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_contact"),
        },
    )
    feet_slide_left = RewTerm(
        func=base_mdp.feet_slide,
        weight=-0.125,
        params={
            "sensor_cfg": SceneEntityCfg("left_foot_contact"),
            "asset_cfg": SceneEntityCfg("robot", body_names="Left_foot"),
        },
    )
    feet_slide_right = RewTerm(
        func=base_mdp.feet_slide,
        weight=-0.125,
        params={
            "sensor_cfg": SceneEntityCfg("right_foot_contact"),
            "asset_cfg": SceneEntityCfg("robot", body_names="Right_foot"),
        },
    )
    feet_slide = None
    dof_pos_limits = RewTerm(
        func=base_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*shin_foot")},
    )
    joint_deviation_torso = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="Spine")},
    )
    undesired_contacts = None


@configclass
class MarkEnvCfg(LocomotionVelocityRoughEnvCfg):
    """mark_v1 flat ground velocity tracking"""

    rewards: MarkRewardsCfg = MarkRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        self.scene.robot = MARK_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/Humanoid/Body/Body_lower"
        self.scene.left_foot_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Humanoid/Left_leg/Left_foot",
            history_length=3,
            track_air_time=True,
        )
        self.scene.right_foot_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Humanoid/RIght_leg/Right_foot",
            history_length=3,
            track_air_time=True,
        )

        self.scene.num_envs = 4096

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Body_lower"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0, 0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.terminations.base_contact.params["sensor_cfg"].body_names = "Body_lower"

        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class MarkEnvCfg_PLAY(MarkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, -1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
