from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


_MIRROR_JOINT_PAIRS = [
    ("Lbody_hip", "Rbody_hip"),
    ("Lhip_shin", "Rhip_shin"),
    ("Lshin_foot", "Rshin_foot"),
]

_SIGN_FLIP_JOINTS: list[str] = []


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = _mirror_policy_obs(
            env.unwrapped, obs["policy"]
        )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(
            batch_size * 2, actions.shape[1], device=actions.device
        )
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _mirror_joint_vector(env.unwrapped, actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _mirror_policy_obs(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    term_names = env.observation_manager.active_terms["policy"]
    term_dims = env.observation_manager.group_obs_term_dim["policy"]

    offset = 0
    for name, dims in zip(term_names, term_dims):
        size = math.prod(dims)
        chunk = obs[:, offset : offset + size]
        mirror_fn = _TERM_MIRROR_FUNCS.get(name)
        if mirror_fn is not None:
            obs[:, offset : offset + size] = mirror_fn(env, chunk)
        offset += size

    return obs


def _mirror_sign(sign: list[float]):
    def _fn(env: ManagerBasedRLEnv, t: torch.Tensor) -> torch.Tensor:
        return t * torch.tensor(sign, device=t.device, dtype=t.dtype)

    return _fn


def _mirror_joints_obs(env: ManagerBasedRLEnv, t: torch.Tensor) -> torch.Tensor:
    return _mirror_joint_vector(env, t)


_TERM_MIRROR_FUNCS = {
    "base_lin_vel": _mirror_sign([1.0, -1.0, 1.0]),
    "base_ang_vel": _mirror_sign([-1.0, 1.0, -1.0]),
    "projected_gravity": _mirror_sign([1.0, -1.0, 1.0]),
    "velocity_commands": _mirror_sign([1.0, -1.0, -1.0]),
    "joint_pos": _mirror_joints_obs,
    "joint_vel": _mirror_joints_obs,
    "actions": _mirror_joints_obs,
}


def _get_joint_mirror_map(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    cache = getattr(env, "_mark_symmetry_joint_map", None)
    if cache is not None:
        return cache

    asset = env.scene["robot"]
    joint_names = list(asset.data.joint_names)
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    n = len(joint_names)

    perm = list(range(n))
    sign = [1.0] * n
    for left, right in _MIRROR_JOINT_PAIRS:
        li, ri = name_to_idx[left], name_to_idx[right]
        perm[li], perm[ri] = ri, li
    for name in _SIGN_FLIP_JOINTS:
        sign[name_to_idx[name]] = -1.0

    perm_t = torch.tensor(perm, device=env.device, dtype=torch.long)
    sign_t = torch.tensor(sign, device=env.device)
    env._mark_symmetry_joint_map = (perm_t, sign_t)
    return perm_t, sign_t


def _mirror_joint_vector(env: ManagerBasedRLEnv, data: torch.Tensor) -> torch.Tensor:
    perm, sign = _get_joint_mirror_map(env)
    return data[:, perm] * sign
