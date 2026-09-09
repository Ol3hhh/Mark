"""Random-action / Train / Play script for Mark tasks"""

import gymnasium as gym
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

    if args_cli.task == "Mark-Flat-v1-Train":
        # Pass agent_cfg if train_agent expects it
        train_agent(simulation_app, env, args_cli.task, agent_cfg=agent_cfg)
    elif args_cli.task == "Mark-Flat-v1-Play":
        play_agent(simulation_app, env, args_cli.task)
    else:
        env.reset()
        random_agent(simulation_app, env, args_cli.task)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
