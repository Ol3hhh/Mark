"""Contract / config tests that do not launch Isaac Sim."""

from __future__ import annotations

import ast
import re

from helpers import ASSETS_DIR, JOINT_NAMES, REPO_ROOT

MARK_V1_ROBOT = REPO_ROOT / "source" / "mark_tasks" / "robots" / "mark_v1.py"
MARK_ENV_CFG = (
    REPO_ROOT
    / "source"
    / "mark_tasks"
    / "tasks"
    / "locomotion"
    / "mark_v1"
    / "mark_env_cfg.py"
)
MARK_INIT = (
    REPO_ROOT
    / "source"
    / "mark_tasks"
    / "tasks"
    / "locomotion"
    / "mark_v1"
    / "__init__.py"
)
SYMMETRY = (
    REPO_ROOT
    / "source"
    / "mark_tasks"
    / "tasks"
    / "locomotion"
    / "mark_v1"
    / "mdp"
    / "symmetry.py"
)
MARK_V2_ROBOT = REPO_ROOT / "source" / "mark_tasks" / "robots" / "mark_v2.py"
MARK_V2_ENV_CFG = (
    REPO_ROOT
    / "source"
    / "mark_tasks"
    / "tasks"
    / "locomotion"
    / "mark_v2"
    / "mark_v2_env_cfg.py"
)
MARK_V2_INIT = (
    REPO_ROOT
    / "source"
    / "mark_tasks"
    / "tasks"
    / "locomotion"
    / "mark_v2"
    / "__init__.py"
)
MARK_V2_SYMMETRY = (
    REPO_ROOT
    / "source"
    / "mark_tasks"
    / "tasks"
    / "locomotion"
    / "mark_v2"
    / "mdp"
    / "symmetry.py"
)

MARK_V2_JOINT_NAMES = [
    "Spine",
    "Neck_body",
    "Head_neck",
    "Lbody_hip",
    "Rbody_hip",
    "Lhip_shin",
    "Rhip_shin",
    "Lshin_foot",
    "Rshin_foot",
    "Lbody_shoulder",
    "Rbody_shoulder",
    "Lshoulder_upperarm",
    "Rshoulder_upperarm",
    "Lupperarm_forearm",
    "Rupperarm_forearm",
]


def test_mark_v1_usd_asset_exists():
    usd = ASSETS_DIR / "mark_v1.usd"
    assert usd.is_file(), f"Expected committed USD at {usd}"


def test_robot_cfg_usd_path_points_at_assets():
    text = MARK_V1_ROBOT.read_text(encoding="utf-8")
    assert 'parents[3] / "assets" / "mark_v1.usd"' in text


def test_robot_cfg_joint_names_match_expected_set():
    text = MARK_V1_ROBOT.read_text(encoding="utf-8")
    # joint_pos keys in init_state
    found = set(
        re.findall(
            r'"((?:Spine|Lbody_hip|Lhip_shin|Lshin_foot|Rbody_hip|Rhip_shin|Rshin_foot))"',
            text,
        )
    )
    assert found == set(JOINT_NAMES)


def test_symmetry_pairs_cover_leg_joints():
    text = SYMMETRY.read_text(encoding="utf-8")
    tree = ast.parse(text)
    pairs = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_MIRROR_JOINT_PAIRS":
                    pairs = ast.literal_eval(node.value)
    assert pairs is not None
    mirrored = {name for pair in pairs for name in pair}
    expected_legs = {n for n in JOINT_NAMES if n != "Spine"}
    assert mirrored == expected_legs


def test_gym_task_ids_registered_in_init():
    text = MARK_INIT.read_text(encoding="utf-8")
    for task_id in ("Mark-Flat-v1-Train", "Mark-Flat-v1-Play", "Mark-Flat-v1-RA"):
        assert f'id="{task_id}"' in text


def test_check_motion_task_registered_in_init():
    """The walk-eval gate task must reuse the PLAY env cfg (fixed velocity,
    num_envs=1) so check_motion.py evaluates the same conditions as -Play."""
    text = MARK_INIT.read_text(encoding="utf-8")
    assert 'id="Mark-Flat-v1-Check"' in text
    check_block = text.split('id="Mark-Flat-v1-Check"')[1]
    assert "MarkEnvCfg_PLAY" in check_block


def test_env_cfg_uses_split_foot_sensors():
    """Sensor prim paths must use Left_leg and Right_leg."""
    text = MARK_ENV_CFG.read_text(encoding="utf-8")
    assert "left_foot_contact" in text
    assert "right_foot_contact" in text
    assert "Left_leg/Left_foot" in text
    assert "Right_leg/Right_foot" in text
    assert "feet_air_time_positive_biped_split" in text


def test_play_cfg_uses_single_env():
    text = MARK_ENV_CFG.read_text(encoding="utf-8")
    # MarkEnvCfg_PLAY sets num_envs = 1
    assert "class MarkEnvCfg_PLAY" in text
    play_section = text.split("class MarkEnvCfg_PLAY")[1]
    assert "self.scene.num_envs = 1" in play_section


def test_cli_task_choices_match_registered_ids():
    cli = (REPO_ROOT / "scripts" / "cli.py").read_text(encoding="utf-8")
    for task_id in ("Mark-Flat-v1-Train", "Mark-Flat-v1-Play", "Mark-Flat-v1-RA"):
        assert task_id in cli


def test_cli_task_choices_include_check_motion():
    cli = (REPO_ROOT / "scripts" / "cli.py").read_text(encoding="utf-8")
    assert "Mark-Flat-v1-Check" in cli


def test_agent_dispatches_check_motion_task():
    agent = (REPO_ROOT / "scripts" / "agent.py").read_text(encoding="utf-8")
    assert "from check_motion import check_motion" in agent
    assert '"Mark-Flat-v1-Check"' in agent
    assert "sys.exit(0 if passed else 1)" in agent


def test_check_motion_script_defines_pass_fail_thresholds():
    """check_motion.py must exist and expose the walk-eval metrics/thresholds"""
    check_motion = (REPO_ROOT / "motion_tests" / "check_motion.py").read_text(
        encoding="utf-8"
    )
    assert "def check_motion(" in check_motion
    assert "min_survival_rate" in check_motion
    assert "max_mean_lin_vel_error" in check_motion
    assert "time_outs" in check_motion
    assert "return passed" in check_motion


def test_mark_v2_env_cfg_tracks_yaw_and_hip_antiphase():
    text = MARK_V2_ENV_CFG.read_text(encoding="utf-8")
    assert "track_ang_vel_z_world_exp" in text
    assert "track_ang_vel_z_exp = None" not in text
    assert "hip_antiphase_impl" in text


def test_mark_v2_usd_asset_exists():
    usd = ASSETS_DIR / "mark_v2.usd"
    assert usd.is_file(), f"Expected USD at {usd}"


def test_mark_v2_robot_cfg_usd_path_points_at_assets():
    text = MARK_V2_ROBOT.read_text(encoding="utf-8")
    assert 'parents[3] / "assets" / "mark_v2.usd"' in text


def test_mark_v2_robot_cfg_joint_names_match_expected_set():
    text = MARK_V2_ROBOT.read_text(encoding="utf-8")
    for name in MARK_V2_JOINT_NAMES:
        assert f'"{name}"' in text
    # Ensure fixed joints are NOT present
    assert "Lforearm_hand" not in text
    assert "Rforearm_hand" not in text
    assert ".*forearm_hand" not in text


def test_mark_v2_symmetry_pairs_cover_bilateral_joints():
    text = MARK_V2_SYMMETRY.read_text(encoding="utf-8")
    tree = ast.parse(text)
    pairs = None
    sign_flips = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "_MIRROR_JOINT_PAIRS":
                        pairs = ast.literal_eval(node.value)
                    elif target.id == "_SIGN_FLIP_JOINTS":
                        sign_flips = ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "_MIRROR_JOINT_PAIRS" and node.value is not None:
                pairs = ast.literal_eval(node.value)
            elif node.target.id == "_SIGN_FLIP_JOINTS" and node.value is not None:
                sign_flips = ast.literal_eval(node.value)
    assert pairs is not None
    assert sign_flips is not None

    mirrored = {name for pair in pairs for name in pair}
    expected_bilateral = {
        n for n in MARK_V2_JOINT_NAMES if n not in ("Spine", "Neck_body", "Head_neck")
    }
    assert mirrored == expected_bilateral


def test_mark_v2_gym_task_ids_registered_in_init():
    text = MARK_V2_INIT.read_text(encoding="utf-8")
    for task_id in (
        "Mark-Flat-v2-Train",
        "Mark-Flat-v2-Play",
        "Mark-Flat-v2-RA",
        "Mark-Flat-v2-Check",
    ):
        assert f'id="{task_id}"' in text


def test_cli_task_choices_include_mark_v2():
    cli = (REPO_ROOT / "scripts" / "cli.py").read_text(encoding="utf-8")
    for task_id in (
        "Mark-Flat-v2-Train",
        "Mark-Flat-v2-Play",
        "Mark-Flat-v2-RA",
        "Mark-Flat-v2-Check",
    ):
        assert task_id in cli


def test_mark_v2_symmetry_sign_flips_are_only_pseudo_vector_axes():
    """Pitch joints (legs, elbows, *_shoulder_upperarm) swap only; yaw/abduction flip."""
    text = MARK_V2_SYMMETRY.read_text(encoding="utf-8")
    tree = ast.parse(text)
    sign_flips = None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "_SIGN_FLIP_JOINTS" and node.value is not None:
                sign_flips = ast.literal_eval(node.value)
    assert sign_flips is not None
    assert set(sign_flips) == {
        "Head_neck",
        "Neck_body",
        "Lbody_shoulder",
        "Rbody_shoulder",
    }


def test_mark_v2_env_cfg_guards_against_lean_and_dominant_leg():
    text = MARK_V2_ENV_CFG.read_text(encoding="utf-8")
    assert "bad_orientation" in text
    assert "lin_vel_z_l2 = None" not in text
    assert "feet_air_time_asymmetry" in text
    assert "yaw_rate_l2" in text
