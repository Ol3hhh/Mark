"""mark_v1 articulation config."""

from math import radians
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_MARK_V1_USD = str(Path(__file__).resolve().parents[3] / "assets" / "mark_v1.usd")


MARK_V1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_MARK_V1_USD,
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
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            "Spine": radians(0.0),
            "Lbody_hip": radians(0.0),
            "Lhip_shin": radians(-12.0),
            "Lshin_foot": radians(10.0),
            "Rbody_hip": radians(0.0),
            "Rhip_shin": radians(-12.0),
            "Rshin_foot": radians(10.0),
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
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
