# Launch Isaac Sim 

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Agent script for Mark")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations"
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate")
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
    choices=["Mark-Flat-v1-Play", "Mark-Flat-v1-Train", "Mark-Flat-v1-RA"],
    help="Name of the task",
)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app