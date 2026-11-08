"""Observation terms for the mark_v3 balance task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def phase_features(angle: torch.Tensor) -> torch.Tensor:
    """Encode an unbounded revolute-joint angle without a wrap discontinuity."""
    return torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)


def contact_state_from_force_matrix(
    force_matrix_w: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Return one binary contact state per environment from a filtered force matrix."""
    contact_force = torch.linalg.vector_norm(force_matrix_w, dim=-1)
    reduce_dims = tuple(range(1, contact_force.ndim))
    return (contact_force.amax(dim=reduce_dims) > threshold).float()


def wheel_phase(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return ``sin`` and ``cos`` of wheel angle."""
    asset: Articulation = env.scene[asset_cfg.name]
    return phase_features(asset.data.joint_pos[:, asset_cfg.joint_ids])


def foot_pedal_contacts(
    env: ManagerBasedEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    threshold: float = 5.0,
) -> torch.Tensor:
    """Return left/right foot-pedal contact indicators."""
    left_sensor: ContactSensor = env.scene.sensors[left_sensor_cfg.name]
    right_sensor: ContactSensor = env.scene.sensors[right_sensor_cfg.name]
    if (
        left_sensor.data.force_matrix_w is None
        or right_sensor.data.force_matrix_w is None
    ):
        raise RuntimeError(
            "Foot-pedal observations require filtered contact sensor force matrices."
        )

    left_contact = contact_state_from_force_matrix(
        left_sensor.data.force_matrix_w, threshold
    )
    right_contact = contact_state_from_force_matrix(
        right_sensor.data.force_matrix_w, threshold
    )
    return torch.stack((left_contact, right_contact), dim=-1)
