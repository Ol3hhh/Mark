"""Checkpoint evaluation gate for mark_v3 balance."""

from __future__ import annotations

import os

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

DEFAULT_NUM_EPISODES = 5
DEFAULT_MIN_SURVIVAL_RATE = 0.8
DEFAULT_MAX_MEAN_TILT_RAD = 0.35
DEFAULT_MIN_PEDAL_CONTACT_RATIO = 0.5


def _contact_state(force_matrix_w: torch.Tensor, threshold: float) -> torch.Tensor:
    force = torch.linalg.vector_norm(force_matrix_w, dim=-1)
    reduce_dims = tuple(range(1, force.ndim))
    return force.amax(dim=reduce_dims) > threshold


def check_balance(
    simulation_app,
    env,
    task_name,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    min_survival_rate: float = DEFAULT_MIN_SURVIVAL_RATE,
    max_mean_tilt_rad: float = DEFAULT_MAX_MEAN_TILT_RAD,
    min_pedal_contact_ratio: float = DEFAULT_MIN_PEDAL_CONTACT_RATIO,
) -> bool:
    """Evaluate survival, uprightness, and foot-pedal contact."""
    num_episodes = int(os.environ.get("CHECK_BALANCE_NUM_EPISODES", num_episodes))
    min_survival_rate = float(
        os.environ.get("CHECK_BALANCE_MIN_SURVIVAL_RATE", min_survival_rate)
    )
    max_mean_tilt_rad = float(
        os.environ.get("CHECK_BALANCE_MAX_MEAN_TILT_RAD", max_mean_tilt_rad)
    )
    min_pedal_contact_ratio = float(
        os.environ.get(
            "CHECK_BALANCE_MIN_PEDAL_CONTACT_RATIO",
            min_pedal_contact_ratio,
        )
    )

    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    logs_root = os.environ.get("MARK_LOGS_ROOT", "logs")
    log_root = os.path.abspath(
        os.path.join(logs_root, "rsl_rl", agent_cfg.experiment_name)
    )
    resume_path = get_checkpoint_path(
        log_root, agent_cfg.load_run, agent_cfg.load_checkpoint
    )

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(
        wrapped_env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)

    base_env = wrapped_env.unwrapped
    robot = base_env.scene["robot"]
    left_sensor = base_env.scene.sensors["left_foot_pedal_contact"]
    right_sensor = base_env.scene.sensors["right_foot_pedal_contact"]

    fell_count = 0
    timeout_count = 0
    tilt_sum = 0.0
    contact_sum = 0.0
    sample_count = 0

    obs = wrapped_env.get_observations()
    while simulation_app.is_running() and (fell_count + timeout_count) < num_episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _rew, dones, extras = wrapped_env.step(actions)

            tilt = torch.acos(
                torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
            )
            left_contact = _contact_state(left_sensor.data.force_matrix_w, 5.0)
            right_contact = _contact_state(right_sensor.data.force_matrix_w, 5.0)
            tilt_sum += tilt.sum().item()
            contact_sum += (
                0.5 * (left_contact.float() + right_contact.float())
            ).sum().item()
            sample_count += tilt.numel()

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
    survival_rate = timeout_count / total_episodes if total_episodes else 0.0
    mean_tilt_rad = tilt_sum / sample_count if sample_count else float("inf")
    pedal_contact_ratio = (
        contact_sum / sample_count if sample_count else 0.0
    )
    passed = (
        total_episodes >= num_episodes
        and survival_rate >= min_survival_rate
        and mean_tilt_rad <= max_mean_tilt_rad
        and pedal_contact_ratio >= min_pedal_contact_ratio
    )

    print(f"[check_balance] checkpoint: {resume_path}")
    print(
        f"[check_balance] episodes: {total_episodes}/{num_episodes} "
        f"(fell={fell_count}, timeout={timeout_count})"
    )
    print(
        f"[check_balance] survival_rate: {survival_rate:.3f} "
        f"(min: {min_survival_rate:.3f})"
    )
    print(
        f"[check_balance] mean_tilt_rad: {mean_tilt_rad:.3f} "
        f"(max: {max_mean_tilt_rad:.3f})"
    )
    print(
        f"[check_balance] pedal_contact_ratio: {pedal_contact_ratio:.3f} "
        f"(min: {min_pedal_contact_ratio:.3f})"
    )
    print(f"[check_balance] result: {'PASS' if passed else 'FAIL'}")
    return passed
