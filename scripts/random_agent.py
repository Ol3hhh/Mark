"""Random-action smoke test for Mark tasks"""

import os

import torch


def random_agent(simulation_app, env, task_name):
    max_steps = int(os.environ.get("MARK_RANDOM_AGENT_STEPS", "0"))
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = (
                2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            )
            observations, rewards, *_ = env.step(actions)
            policy_observations = (
                observations["policy"]
                if isinstance(observations, dict)
                else observations
            )
            if not torch.isfinite(policy_observations).all():
                raise RuntimeError(f"{task_name}: non-finite policy observation")
            if not torch.isfinite(rewards).all():
                raise RuntimeError(f"{task_name}: non-finite reward")
        step += 1
        if max_steps > 0 and step >= max_steps:
            break
