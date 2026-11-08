"""Articulation configuration for the passively driven mark_v3 unicycle.

Only the humanoid joints are exposed to the policy.  The wheel, pedals, and
seat joint are configured as passive impedance elements: none of them can
produce a propulsive position target.
"""

from math import radians
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_MARK_V3_USD = str(Path(__file__).resolve().parents[3] / "assets" / "mark_v3.usd")

MARK_V3_POLICY_JOINTS = [
    "Spine",
    "Lbody_hip",
    "Lhip_shin",
    "Lshin_foot",
    "Rbody_hip",
    "Rhip_shin",
    "Rshin_foot",
    "Lbody_shoulder",
    "Lshoulder_upperarm",
    "Lupperarm_forearm",
    "Rbody_shoulder",
    "Rshoulder_upperarm",
    "Rupperarm_forearm",
]

MARK_V3_PASSIVE_JOINTS = [
    "wheel_axle",
    "left_pedal",
    "right_pedal",
    "seat_joint",
]


MARK_V3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_MARK_V3_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    articulation_root_prim_path="/Unicycle/Frame",
    init_state=ArticulationCfg.InitialStateCfg(
        # Frame origin is at the wheel axle (z = 0.372 m, equal to wheel radius);
        # 0.02 m clearance prevents initial ground penetration.
        pos=(0.0, 0.0, 0.392),
        joint_pos={
            "Spine": radians(0.0),
            "Neck_body": radians(0.0),
            "Head_neck": radians(0.0),
            "Lbody_hip": radians(21.51),
            "Lhip_shin": radians(-68.66),
            "Lshin_foot": radians(47.15),
            "Rbody_hip": radians(42.29),
            "Rhip_shin": radians(-71.44),
            "Rshin_foot": radians(29.71),
            "Lbody_shoulder": radians(-22.0),
            "Lshoulder_upperarm": radians(0.0),
            "Lupperarm_forearm": radians(20.0),
            "Rbody_shoulder": radians(22.0),
            "Rshoulder_upperarm": radians(0.0),
            "Rupperarm_forearm": radians(20.0),
            "wheel_axle": 0.0,
            "left_pedal": 0.0,
            "right_pedal": 0.0,
            "seat_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "Neck_body",
                "Head_neck",
            ],
            effort_limit_sim=20.0,
            stiffness={
                "Neck_body": 50.0,
                "Head_neck": 35.0,
            },
            damping={
                "Neck_body": 4.0,
                "Head_neck": 3.0,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*body_shoulder",
                ".*shoulder_upperarm",
                ".*upperarm_forearm",
            ],
            effort_limit_sim=40.0,
            stiffness={
                ".*body_shoulder": 30.0,
                ".*shoulder_upperarm": 30.0,
                ".*upperarm_forearm": 25.0,
            },
            damping={
                ".*body_shoulder": 3.0,
                ".*shoulder_upperarm": 3.0,
                ".*upperarm_forearm": 2.5,
            },
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*body_hip", ".*hip_shin", ".*shin_foot"],
            effort_limit_sim=150.0,
            stiffness={
                ".*body_hip": 150.0,
                ".*hip_shin": 150.0,
                ".*shin_foot": 80.0,
            },
            damping={
                ".*body_hip": 8.0,
                ".*hip_shin": 8.0,
                ".*shin_foot": 5.0,
            },
        ),
        "spine": ImplicitActuatorCfg(
            joint_names_expr=["Spine"],
            effort_limit_sim=40.0,
            stiffness=80.0,
            damping=5.0,
        ),
        # Passive mechanics.  Zero stiffness on the wheel and pedals guarantees
        # that the unicycle cannot drive itself; motion must come from foot contact.
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=["wheel_axle"],
            effort_limit_sim=200.0,
            stiffness=0.0,
            damping=0.05,
        ),
        "pedals": ImplicitActuatorCfg(
            joint_names_expr=["left_pedal", "right_pedal"],
            effort_limit_sim=50.0,
            stiffness=0.0,
            damping=2.0,
        ),
        "seat": ImplicitActuatorCfg(
            joint_names_expr=["seat_joint"],
            effort_limit_sim=80.0,
            stiffness=60.0,
            damping=5.0,
        ),
    },
)