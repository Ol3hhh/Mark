"""Random-action / Train / Play / Check script for Mark tasks"""

# ruff: noqa: I001
# NOTE: the import order below is NOT alphabetical on purpose and must stay
# that way -- `cli` has to be imported first because it starts AppLauncher
# and puts motion_tests/ on sys.path. Every import after it (check_motion,
# isaaclab_tasks, play_agent, random_agent, train_agent) transitively pulls
# in isaaclab/isaacsim modules that only resolve once AppLauncher is running
# (see scripts/cli.py and .cursor/rules/rl-code-style.mdc). Do NOT let an
# auto-formatter/isort "fix" re-sort these -- that reintroduces
# `ModuleNotFoundError: No module named 'check_motion'` / `No module named 'pxr'`.
import sys

import gymnasium as gym

from cli import args_cli, simulation_app
from check_motion import check_motion
from isaaclab_tasks.utils.hydra import hydra_task_config
from play_agent import play_agent
from random_agent import random_agent
from train_agent import train_agent


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    if args_cli.task == "Mark-Flat-v1-Train":
        # Pass agent_cfg if train_agent expects it
        train_agent(simulation_app, env, args_cli.task, agent_cfg=agent_cfg)
    elif args_cli.task == "Mark-Flat-v1-Play":
        play_agent(simulation_app, env, args_cli.task)
    elif args_cli.task == "Mark-Flat-v1-Check":
        passed = check_motion(simulation_app, env, args_cli.task)
        env.close()
        simulation_app.close()
        sys.exit(0 if passed else 1)
    else:
        env.reset()
        random_agent(simulation_app, env, args_cli.task)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
