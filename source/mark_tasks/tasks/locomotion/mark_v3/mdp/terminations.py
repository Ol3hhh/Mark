"""Termination terms for mark_v3."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def body_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when any selected body drops below a world-frame height."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    return torch.any(body_height < minimum_height, dim=-1)


def filtered_illegal_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate on a filtered contact, such as torso against the ground."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if sensor.data.force_matrix_w is None:
        raise RuntimeError(
            f"Sensor '{sensor_cfg.name}' requires filter_prim_paths_expr."
        )
    force = torch.linalg.vector_norm(sensor.data.force_matrix_w, dim=-1)
    reduce_dims = tuple(range(1, force.ndim))
    return force.amax(dim=reduce_dims) > threshold
