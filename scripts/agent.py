"""Random-action / Train / Play / Check script for Mark tasks"""

import sys

import gymnasium as gym
from check_motion import check_motion
from cli import args_cli, simulation_app
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

    if args_cli.task in ("Mark-Flat-v1-Train", "Mark-Flat-v2-Train"):
        train_agent(simulation_app, env, args_cli.task, agent_cfg=agent_cfg)
    elif args_cli.task in ("Mark-Flat-v1-Play", "Mark-Flat-v2-Play"):
        play_agent(simulation_app, env, args_cli.task)
    elif args_cli.task in ("Mark-Flat-v1-Check", "Mark-Flat-v2-Check"):
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
