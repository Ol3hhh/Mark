"""Mark training agent invoke"""

import os
from datetime import datetime, timezone

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner


def train_agent(simulation_app, env, task_name, agent_cfg=None):
    if agent_cfg is None:
        agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")

    log_root = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    log_dir = os.path.join(
        log_root, datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )
