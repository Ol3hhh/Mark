"""Random-action smoke test for Mark tasks"""

import torch


def random_agent(simulation_app, env, task_name):
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = (
                2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            )
            env.step(actions)
