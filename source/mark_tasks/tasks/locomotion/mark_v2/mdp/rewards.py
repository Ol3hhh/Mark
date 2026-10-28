"""MDP rewards Mark_v2"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time_positive_biped_split(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Same reward as ``feet_air_time_positive_biped``, reading each foot from its own sensor."""
    left: ContactSensor = env.scene.sensors[left_sensor_cfg.name]
    right: ContactSensor = env.scene.sensors[right_sensor_cfg.name]

    air_time = torch.cat(
        [
            left.data.current_air_time[:, left_sensor_cfg.body_ids],
            right.data.current_air_time[:, right_sensor_cfg.body_ids],
        ],
        dim=1,
    )
    contact_time = torch.cat(
        [
            left.data.current_contact_time[:, left_sensor_cfg.body_ids],
            right.data.current_contact_time[:, right_sensor_cfg.body_ids],
        ],
        dim=1,
    )
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


def feet_air_time_asymmetry(
    env: ManagerBasedRLEnv,
    command_name: str,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    max_air_time: float = 1.0,
) -> torch.Tensor:
    """Squared difference between the last swing durations of the left and right foot.

    Meant to be used with a negative weight: it penalises a dominant-leg gait (one foot
    always swinging / hopping, the other dragged) that the single-stance air-time reward
    alone does not rule out. ``last_air_time`` is updated at touchdown, so both feet have
    to actually land to stop being penalised. Gated off for ~zero commands.
    """
    left: ContactSensor = env.scene.sensors[left_sensor_cfg.name]
    right: ContactSensor = env.scene.sensors[right_sensor_cfg.name]

    left_air = left.data.last_air_time[:, left_sensor_cfg.body_ids].reshape(
        left.data.last_air_time.shape[0], -1
    )
    right_air = right.data.last_air_time[:, right_sensor_cfg.body_ids].reshape(
        right.data.last_air_time.shape[0], -1
    )
    diff = torch.clamp(left_air, max=max_air_time) - torch.clamp(
        right_air, max=max_air_time
    )
    penalty = torch.sum(torch.square(diff), dim=1)
    moving = (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return penalty * moving


def yaw_rate_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared world-frame yaw rate. The command is a zero yaw rate, so this damps arcs."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_w[:, 2])


def motion_synchronisation_impl(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float | None = None,
) -> torch.Tensor:
    """Contralateral hip–arm swing: L_hip with R_arm, R_hip with L_arm.

    ``max_value`` caps the (otherwise unbounded, amplitude-growing) product so the
    term cannot dominate the return or be farmed with a static extreme pose.
    """
    asset = env.scene[asset_cfg.name]

    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    dq = q - q0  # [N, 4] = L_hip, R_hip, L_arm, R_arm
    coord = dq[:, 0] * dq[:, 3] + dq[:, 1] * dq[:, 2]
    moving = (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return torch.clamp(coord, min=0.0, max=max_value) * moving


def hip_antiphase_impl(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float | None = None,
) -> torch.Tensor:
    """Reward opposite hip deviations from default (left/right antiphase gait).

    ``max_value`` caps the (otherwise unbounded, amplitude-growing) product so a static
    scissor stance with the same leg always in front cannot be farmed for reward.
    """
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    dq = q - q0  # [N, 2] = L_hip, R_hip
    coord = -dq[:, 0] * dq[:, 1]
    moving = (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return torch.clamp(coord, min=0.0, max=max_value) * moving
