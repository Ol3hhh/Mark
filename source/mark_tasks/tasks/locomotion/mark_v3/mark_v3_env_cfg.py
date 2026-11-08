"""Manager-based stationary-balance task for the physically pedaled mark_v3."""

import math

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from mark_tasks.robots.mark_v3 import (
    MARK_V3_CFG,
    MARK_V3_PASSIVE_JOINTS,
    MARK_V3_POLICY_JOINTS,
)

from . import mdp as mark_v3_mdp

GROUND_COLLISION_PATH = "/World/ground/terrain/GroundPlane/CollisionPlane"


@configclass
class MarkV3SceneCfg(InteractiveSceneCfg):
    """Flat scene with filtered sensors for pedal and fall contacts."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot = MARK_V3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    left_foot_pedal_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Humanoid/Left_leg/Left_foot",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/Unicycle/Left_pedal",
        ],
        history_length=3,
    )
    right_foot_pedal_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Humanoid/Right_leg/Right_foot",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/Unicycle/Right_pedal",
        ],
        history_length=3,
    )
    lower_body_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Humanoid/Body/Body_lower",
        filter_prim_paths_expr=[GROUND_COLLISION_PATH],
        history_length=3,
    )
    upper_body_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Humanoid/Body/Body_upper",
        filter_prim_paths_expr=[GROUND_COLLISION_PATH],
        history_length=3,
    )
    head_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Humanoid/Head_/Head",
        filter_prim_paths_expr=[GROUND_COLLISION_PATH],
        history_length=3,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    """Position targets for humanoid joints only."""

    legs = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "Lbody_hip",
            "Lhip_shin",
            "Lshin_foot",
            "Rbody_hip",
            "Rhip_shin",
            "Rshin_foot",
        ],
        scale={
            ".*body_hip": 0.6,
            ".*hip_shin": 0.8,
            ".*shin_foot": 0.5,
        },
        use_default_offset=True,
        preserve_order=True,
    )
    upper_body = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "Spine",
            "Lbody_shoulder",
            "Lshoulder_upperarm",
            "Lupperarm_forearm",
            "Rbody_shoulder",
            "Rshoulder_upperarm",
            "Rupperarm_forearm",
        ],
        scale={
            "Spine": 0.15,
            ".*body_shoulder": 0.45,
            ".*shoulder_upperarm": 0.5,
            ".*upperarm_forearm": 0.6,
        },
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Policy observations for frame balance, posture, crank phase and contacts."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel, scale=0.5)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, scale=0.25)
        frame_projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        torso_projected_gravity = ObsTerm(
            func=base_mdp.body_projected_gravity_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["Body_lower"]
                )
            },
        )
        humanoid_joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=MARK_V3_POLICY_JOINTS,
                    preserve_order=True,
                )
            },
        )
        humanoid_joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=MARK_V3_POLICY_JOINTS,
                    preserve_order=True,
                )
            },
        )
        wheel_phase = ObsTerm(
            func=mark_v3_mdp.wheel_phase,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["wheel_axle"]
                )
            },
        )
        passive_joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=MARK_V3_PASSIVE_JOINTS,
                    preserve_order=True,
                )
            },
        )
        seat_position = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["seat_joint"]
                )
            },
        )
        feet_pedal_contact = ObsTerm(
            func=mark_v3_mdp.foot_pedal_contacts,
            params={
                "left_sensor_cfg": SceneEntityCfg(
                    "left_foot_pedal_contact"
                ),
                "right_sensor_cfg": SceneEntityCfg(
                    "right_foot_pedal_contact"
                ),
                "threshold": 5.0,
            },
        )
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Small reset perturbations that preserve the initial crank-foot alignment."""

    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )
    reset_humanoid_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.01, 0.01),
            "velocity_range": (-0.02, 0.02),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=MARK_V3_POLICY_JOINTS,
                preserve_order=True,
            ),
        },
    )
    reset_passive_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=MARK_V3_PASSIVE_JOINTS,
                preserve_order=True,
            ),
        },
    )


@configclass
class RewardsCfg:
    """Dense balance objective without any commanded wheel actuation."""

    alive = RewTerm(func=base_mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=base_mdp.is_terminated, weight=-20.0)
    frame_upright = RewTerm(func=base_mdp.flat_orientation_l2, weight=-5.0)
    torso_upright = RewTerm(
        func=mark_v3_mdp.body_upright_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["Body_lower"]
            )
        },
    )
    vertical_velocity = RewTerm(func=base_mdp.lin_vel_z_l2, weight=-0.5)
    angular_velocity = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.1)
    planar_drift = RewTerm(
        func=mark_v3_mdp.planar_velocity_l2,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    wheel_speed = RewTerm(
        func=mark_v3_mdp.wheel_speed_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["wheel_axle"]
            )
        },
    )
    feet_on_pedals = RewTerm(
        func=mark_v3_mdp.feet_pedal_contact_reward,
        weight=1.0,
        params={
            "left_sensor_cfg": SceneEntityCfg("left_foot_pedal_contact"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_pedal_contact"),
            "threshold": 5.0,
        },
    )
    feet_near_pedals = RewTerm(
        func=mark_v3_mdp.feet_pedal_distance_exp,
        weight=0.5,
        params={
            "std": 0.12,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "Left_foot",
                    "Right_foot",
                    "Left_pedal",
                    "Right_pedal",
                ],
                preserve_order=True,
            ),
        },
    )
    joint_limits = RewTerm(
        func=base_mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=MARK_V3_POLICY_JOINTS
            )
        },
    )
    joint_torques = RewTerm(
        func=base_mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=MARK_V3_POLICY_JOINTS
            )
        },
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    action_magnitude = RewTerm(func=base_mdp.action_l2, weight=-0.001)


@configclass
class TerminationsCfg:
    """End episodes after a fall, not after a brief loss of pedal contact."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    excessive_tilt = DoneTerm(
        func=base_mdp.bad_orientation,
        params={"limit_angle": math.radians(50.0)},
    )
    torso_too_low = DoneTerm(
        func=mark_v3_mdp.body_height_below_minimum,
        params={
            "minimum_height": 0.8,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["Body_lower"]
            ),
        },
    )
    lower_body_ground = DoneTerm(
        func=mark_v3_mdp.filtered_illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("lower_body_ground_contact"),
            "threshold": 1.0,
        },
    )
    upper_body_ground = DoneTerm(
        func=mark_v3_mdp.filtered_illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("upper_body_ground_contact"),
            "threshold": 1.0,
        },
    )
    head_ground = DoneTerm(
        func=mark_v3_mdp.filtered_illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("head_ground_contact"),
            "threshold": 1.0,
        },
    )


@configclass
class MarkEnvCfg(ManagerBasedRLEnvCfg):
    """Training configuration for upright physical pedaling."""

    scene: MarkV3SceneCfg = MarkV3SceneCfg(
        num_envs=1024,
        env_spacing=4.0,
        clone_in_fabric=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 0.9
        self.sim.physics_material.restitution = 0.0
        self.viewer.eye = (5.0, 5.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 1.2)


@configclass
class MarkEnvCfg_PLAY(MarkEnvCfg):
    """Deterministic single-environment configuration for evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 30.0
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_humanoid_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_humanoid_joints.params["velocity_range"] = (0.0, 0.0)
