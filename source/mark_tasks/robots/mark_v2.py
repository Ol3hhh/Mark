"""mark_v2 articulation config"""

from math import radians
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ARM_ABDUCTION_DEG = (
    15.0  # default sideways arm offset (hand ~0.18 m further out than at 0 deg)
)

_MARK_V2_USD = str(Path(__file__).resolve().parents[3] / "assets" / "mark_v2.usd")


MARK_V2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_MARK_V2_USD,
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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.7),
        joint_pos={
            "Spine": radians(0.0),
            "Lbody_hip": radians(0.0),
            "Lhip_shin": radians(-12.0),
            "Lshin_foot": radians(10.0),
            "Rbody_hip": radians(0.0),
            "Rhip_shin": radians(-12.0),
            "Rshin_foot": radians(10.0),
            "Head_neck": radians(0.0),
            "Neck_body": radians(0.0),
            # shoulder abduction (axis X): outward is negative on the left, positive on
            # the right (limits L(-130, 10) / R(-10, 130)); keeps the arms off the torso/thighs
            "Lbody_shoulder": radians(-ARM_ABDUCTION_DEG),
            "Lshoulder_upperarm": radians(0.0),
            "Lupperarm_forearm": radians(20.0),
            "Rbody_shoulder": radians(ARM_ABDUCTION_DEG),
            "Rshoulder_upperarm": radians(0.0),
            "Rupperarm_forearm": radians(20.0),
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
            joint_names_expr=[".*body_hip", ".*hip_shin"],
            effort_limit_sim=150.0,
            stiffness={
                ".*body_hip": 150.0,
                ".*hip_shin": 150.0,
            },
            damping={
                ".*body_hip": 5.0,
                ".*hip_shin": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*shin_foot"],
            effort_limit_sim=40.0,
            stiffness=20.0,
            damping=4.0,
        ),
        "spine": ImplicitActuatorCfg(
            joint_names_expr=["Spine"],
            effort_limit_sim=40.0,
            stiffness=80.0,
            damping=5.0,
        ),
    },
)
