"""Shared constants for mark_tasks unit tests."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MARK_V1_MDP = REPO_ROOT / "source" / "mark_tasks" / "tasks" / "locomotion" / "mark_v1" / "mdp"
ASSETS_DIR = REPO_ROOT / "assets"

JOINT_NAMES = [
    "Spine",
    "Lbody_hip",
    "Lhip_shin",
    "Lshin_foot",
    "Rbody_hip",
    "Rhip_shin",
    "Rshin_foot",
]

POLICY_TERM_NAMES = [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
]

POLICY_TERM_DIMS = [
    (3,),
    (3,),
    (3,),
    (3,),
    (len(JOINT_NAMES),),
    (len(JOINT_NAMES),),
    (len(JOINT_NAMES),),
]
