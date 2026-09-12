from __future__ import annotations

import os

import torch
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

DEFAULT_NUM_EPISODES = 5
DEFAULT_MIN_SURVIVAL_RATE = 0.8
DEFAULT_MAX_MEAN_LIN_VEL_ERROR = 0.4

COMMAND_NAME = "base_velocity"


def check_motion(
    simulation_app,
    env,
    task_name,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    min_survival_rate: float = DEFAULT_MIN_SURVIVAL_RATE,
    max_mean_lin_vel_error: float = DEFAULT_MAX_MEAN_LIN_VEL_ERROR,
) -> bool:
    num_episodes = int(os.environ.get("CHECK_MOTION_NUM_EPISODES", num_episodes))
    min_survival_rate = float(
        os.environ.get("CHECK_MOTION_MIN_SURVIVAL_RATE", min_survival_rate)
    )
    max_mean_lin_vel_error = float(
        os.environ.get("CHECK_MOTION_MAX_LIN_VEL_ERROR", max_mean_lin_vel_error)
    )

    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    # Checkpoints are never committed to git (see rl-code-style.mdc), so on a
    # CI runner the checkout dir has no logs/ at all -- MARK_LOGS_ROOT lets a
    # self-hosted runner point at the real training logs dir on that machine
    # (set via the runner's own .env, same pattern as CONDA_SH_PATH). Defaults
    # to the plain relative "logs" dir for normal local/manual runs.
    logs_root = os.environ.get("MARK_LOGS_ROOT", "logs")
    log_root = os.path.abspath(
        os.path.join(logs_root, "rsl_rl", agent_cfg.experiment_name)
    )
    resume_path = get_checkpoint_path(
        log_root, agent_cfg.load_run, agent_cfg.load_checkpoint
    )

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(
        wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)

    base_env = wrapped_env.unwrapped
    robot = base_env.scene["robot"]

    fell_count = 0
    timeout_count = 0
    vel_error_sum = 0.0
    vel_error_steps = 0

    obs = wrapped_env.get_observations()
    while simulation_app.is_running() and (fell_count + timeout_count) < num_episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _rew, dones, extras = wrapped_env.step(actions)

            command_xy = base_env.command_manager.get_command(COMMAND_NAME)[:, :2]
            vel_yaw = quat_apply_inverse(
                yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3]
            )
            vel_error_sum += torch.norm(command_xy - vel_yaw[:, :2], dim=1).sum().item()
            vel_error_steps += command_xy.shape[0]

        done_mask = dones.bool()
        time_outs = extras.get("time_outs")
        if time_outs is not None:
            time_outs = time_outs.bool()
            fell_mask = done_mask & ~time_outs
            timeout_mask = done_mask & time_outs
        else:
            fell_mask = done_mask
            timeout_mask = torch.zeros_like(done_mask)

        fell_count += int(fell_mask.sum().item())
        timeout_count += int(timeout_mask.sum().item())

    total_episodes = fell_count + timeout_count
    survival_rate = (timeout_count / total_episodes) if total_episodes else 0.0
    mean_lin_vel_error = (
        vel_error_sum / vel_error_steps if vel_error_steps else float("inf")
    )

    passed = (
        total_episodes >= num_episodes
        and survival_rate >= min_survival_rate
        and mean_lin_vel_error <= max_mean_lin_vel_error
    )

    print(f"[check_motion] checkpoint: {resume_path}")
    print(
        f"[check_motion] episodes: {total_episodes}/{num_episodes} "
        f"(fell={fell_count}, timeout={timeout_count})"
    )
    print(
        f"[check_motion] survival_rate: {survival_rate:.3f} "
        f"(min required: {min_survival_rate:.3f})"
    )
    print(
        f"[check_motion] mean_lin_vel_error: {mean_lin_vel_error:.3f} "
        f"(max allowed: {max_mean_lin_vel_error:.3f})"
    )
    print(f"[check_motion] result: {'PASS' if passed else 'FAIL'}")

    return passed
