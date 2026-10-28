import isaaclab_tasks.manager_based.locomotion.velocity.mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)

from mark_tasks.robots.mark_v2 import MARK_V2_CFG

from . import mdp as mark_v2_mdp

# Tunable scales (see rewards.py for the term definitions)
MAX_BASE_TILT_RAD = 0.8  # terminate when torso is tilted further than this from upright
HIP_ANTIPHASE_CAP = 0.1  # rad^2, saturation of the hip-product term
MOTION_SYNC_CAP = 0.2  # rad^2, saturation of the hip-arm product term


@configclass
class MarkRewardsCfg(RewardsCfg):
    termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-50.0)
    # vertical bobbing / one-leg hopping must not be free
    lin_vel_z_l2 = RewTerm(func=base_mdp.lin_vel_z_l2, weight=-0.5)
    track_lin_vel_xy_exp = RewTerm(
        func=base_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_ang_vel_z_exp = RewTerm(
        func=base_mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.3},
    )
    # command is zero yaw rate: explicit damping against walking in arcs
    yaw_rate_l2 = RewTerm(func=mark_v2_mdp.yaw_rate_l2, weight=-0.2)
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
    head_stabilisation = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Neck_body", "Head_neck"])
        },
    )

    joint_deviation_arms = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*body_shoulder", ".*upperarm_forearm"]
            )
        },
    )

    undesired_contacts = None

    feet_air_time = RewTerm(
        func=mark_v2_mdp.feet_air_time_positive_biped_split,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "threshold": 0.4,
            "left_sensor_cfg": SceneEntityCfg("left_foot_contact"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_contact"),
        },
    )

    feet_air_time_asymmetry = RewTerm(
        func=mark_v2_mdp.feet_air_time_asymmetry,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "left_sensor_cfg": SceneEntityCfg("left_foot_contact"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_contact"),
        },
    )

    motion_synchronisation = RewTerm(
        func=mark_v2_mdp.motion_synchronisation_impl,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "max_value": MOTION_SYNC_CAP,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Lbody_hip",
                    "Rbody_hip",
                    "Lshoulder_upperarm",
                    "Rshoulder_upperarm",
                ],
            ),
        },
    )
    hip_antiphase = RewTerm(
        func=mark_v2_mdp.hip_antiphase_impl,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "max_value": HIP_ANTIPHASE_CAP,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Lbody_hip", "Rbody_hip"],
            ),
        },
    )


@configclass
class MarkTerminationsCfg(TerminationsCfg):
    # Without this the torso could lean arbitrarily far; only Body_lower ground contact ended episodes.
    bad_orientation = DoneTerm(
        func=base_mdp.bad_orientation,
        params={"limit_angle": MAX_BASE_TILT_RAD},
    )


@configclass
class MarkEnvCfg(LocomotionVelocityRoughEnvCfg):
    """mark_v2 flat ground velocity tracking"""

    rewards: MarkRewardsCfg = MarkRewardsCfg()
    terminations: MarkTerminationsCfg = MarkTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        self.scene.robot = MARK_V2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.contact_forces.prim_path = (
            "{ENV_REGEX_NS}/Robot/Humanoid/Body/Body_lower"
        )
        self.scene.left_foot_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Humanoid/Left_leg/Left_foot",
            history_length=3,
            track_air_time=True,
        )
        self.scene.right_foot_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Humanoid/Right_leg/Right_foot",
            history_length=3,
            track_air_time=True,
        )

        self.scene.num_envs = 4096

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [
            "Body_lower"
        ]
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

        # sin^2(tilt): -1.0 cost only -0.25/step at 30 deg, far below the gait rewards
        self.rewards.flat_orientation_l2.weight = -5.0
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
