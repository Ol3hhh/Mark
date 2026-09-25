# Launch Isaac Sim

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Agent script for Mark")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate"
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Key for agent configuration in gym registry (e.g. 'rsl_rl_cfg_entry_point').",
)
parser.add_argument(
    "--task",
    type=str,
    default="Mark-Flat-v1-Play",
    choices=[
        "Mark-Flat-v1-Play",
        "Mark-Flat-v1-Train",
        "Mark-Flat-v1-RA",
        "Mark-Flat-v1-Check",
    ],
    help="Name of the task",
)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Triggers mark_tasks/__init__.py's import_packages(), which recursively
# imports every tasks/locomotion/<robot>/__init__.py and runs their
# gym.register(...) calls. Without this, gym.spec()/gym.make() never see any
# "Mark-Flat-v*-*" task id (NameNotFound), because nothing else in this
# process ever imports mark_tasks. Must come after AppLauncher (mark_tasks'
# submodules import isaaclab/isaacsim types that only resolve once Kit is up).
import mark_tasks  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "motion_tests"))
