"""Reward terms for physical pedaling and unicycle balance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from .observations import foot_pedal_contacts

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def body_upright_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize tilt of a selected body using projected gravity."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    gravity_w = asset.data.GRAVITY_VEC_W.unsqueeze(1).expand_as(
        body_quat_w[..., :3]
    )
    projected_gravity = quat_apply_inverse(body_quat_w, gravity_w)
    return torch.sum(torch.square(projected_gravity[..., :2]), dim=(-2, -1))


def planar_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize excessive planar drift while allowing corrective movement."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=-1)


def wheel_speed_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize runaway wheel speed."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1
    )


def feet_pedal_contact_reward(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
) -> torch.Tensor:
    """Reward each foot that maintains contact with its pedal."""
    contacts = foot_pedal_contacts(
        env,
        left_sensor_cfg=left_sensor_cfg,
        right_sensor_cfg=right_sensor_cfg,
        threshold=threshold,
    )
    return contacts.mean(dim=-1)


def feet_pedal_distance_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense shaping reward that keeps each foot near its matching pedal."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    left_error = torch.sum(torch.square(body_pos[:, 0] - body_pos[:, 2]), dim=-1)
    right_error = torch.sum(torch.square(body_pos[:, 1] - body_pos[:, 3]), dim=-1)
    return 0.5 * (
        torch.exp(-left_error / std**2) + torch.exp(-right_error / std**2)
    )
