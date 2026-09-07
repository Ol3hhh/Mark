"""Contract / config tests that do not launch Isaac Sim."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from helpers import ASSETS_DIR, JOINT_NAMES, REPO_ROOT

MARK_V1_ROBOT = REPO_ROOT / "source" / "mark_tasks" / "robots" / "mark_v1.py"
MARK_ENV_CFG = REPO_ROOT / "source" / "mark_tasks" / "tasks" / "locomotion" / "mark_v1" / "mark_env_cfg.py"
MARK_INIT = REPO_ROOT / "source" / "mark_tasks" / "tasks" / "locomotion" / "mark_v1" / "__init__.py"
SYMMETRY = REPO_ROOT / "source" / "mark_tasks" / "tasks" / "locomotion" / "mark_v1" / "mdp" / "symmetry.py"


def test_mark_v1_usd_asset_exists():
    usd = ASSETS_DIR / "mark_v1.usd"
    assert usd.is_file(), f"Expected committed USD at {usd}"


def test_robot_cfg_usd_path_points_at_assets():
    text = MARK_V1_ROBOT.read_text(encoding="utf-8")
    assert 'parents[3] / "assets" / "mark_v1.usd"' in text


def test_robot_cfg_joint_names_match_expected_set():
    text = MARK_V1_ROBOT.read_text(encoding="utf-8")
    # joint_pos keys in init_state
    found = set(re.findall(r'"((?:Spine|Lbody_hip|Lhip_shin|Lshin_foot|Rbody_hip|Rhip_shin|Rshin_foot))"', text))
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


def test_env_cfg_uses_split_foot_sensors_and_right_leg_typo():
    """Asset typo RIght_leg must stay consistent in sensor prim paths."""
    text = MARK_ENV_CFG.read_text(encoding="utf-8")
    assert "left_foot_contact" in text
    assert "right_foot_contact" in text
    assert "Left_leg/Left_foot" in text
    assert "RIght_leg/Right_foot" in text
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
