"""Random-action smoke test for Mark tasks """

import os
from datetime import datetime

import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import isaaclab_tasks
import mark_tasks
from isaaclab_tasks.utils import parse_env_cfg, get_checkpoint_path
import importlib.metadata as metadata
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

def random_agent(simulation_app, env, task_name):
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            env.step(actions)
