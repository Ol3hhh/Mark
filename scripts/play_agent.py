"""Random-action smoke test for Mark tasks"""

import os

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner


def play_agent(simulation_app, env, task_name):
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    log_root = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    resume_path = get_checkpoint_path(
        log_root, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _dones, _ = env.step(actions)
