"""Custom MDP reward terms specific to the mark_v1 USD hierarchy.

The Mark asset groups leg links under separate ``Left_leg`` / ``Right_leg`` Xforms
(unlike the flat, sibling-body hierarchy used by Isaac Lab's reference robots, e.g. H1).
Isaac Lab's ``ContactSensorCfg`` only resolves one literal parent prim plus a single
wildcard leaf level, so a single sensor cannot see both feet at once here. We instead
use one sensor per foot and recombine them below to reproduce the standard biped
"single stance" air-time reward from
``isaaclab_tasks.manager_based.locomotion.velocity.mdp.feet_air_time_positive_biped``.
"""

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
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
