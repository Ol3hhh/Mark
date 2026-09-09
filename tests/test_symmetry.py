"""Unit tests for mark_v1 left-right observation/action symmetry."""

from __future__ import annotations

import math

import pytest
import torch
from helpers import JOINT_NAMES, POLICY_TERM_DIMS, POLICY_TERM_NAMES

tensordict = pytest.importorskip("tensordict")
TensorDict = tensordict.TensorDict


def _term_slices():
    offset = 0
    slices = {}
    for name, dims in zip(POLICY_TERM_NAMES, POLICY_TERM_DIMS):
        size = math.prod(dims)
        slices[name] = slice(offset, offset + size)
        offset += size
    return slices


SLICES = _term_slices()


def _make_policy_obs(batch: int, device: torch.device) -> torch.Tensor:
    """Distinct values so mirror swaps are easy to assert."""
    dim = sum(math.prod(d) for d in POLICY_TERM_DIMS)
    obs = torch.arange(batch * dim, device=device, dtype=torch.float32).reshape(
        batch, dim
    )
    return obs


def test_mirror_policy_sign_flips(
    symmetry_mod, fake_symmetry_env, device, policy_obs_dim
):
    obs = torch.ones(2, policy_obs_dim, device=device)
    # set unique vectors per term
    obs[:, SLICES["base_lin_vel"]] = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device
    )
    obs[:, SLICES["base_ang_vel"]] = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device
    )
    obs[:, SLICES["projected_gravity"]] = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device
    )
    obs[:, SLICES["velocity_commands"]] = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device
    )

    mirrored = symmetry_mod._mirror_policy_obs(fake_symmetry_env, obs)

    assert torch.allclose(
        mirrored[:, SLICES["base_lin_vel"]],
        obs[:, SLICES["base_lin_vel"]] * torch.tensor([1.0, -1.0, 1.0], device=device),
    )
    assert torch.allclose(
        mirrored[:, SLICES["base_ang_vel"]],
        obs[:, SLICES["base_ang_vel"]] * torch.tensor([-1.0, 1.0, -1.0], device=device),
    )
    assert torch.allclose(
        mirrored[:, SLICES["projected_gravity"]],
        obs[:, SLICES["projected_gravity"]]
        * torch.tensor([1.0, -1.0, 1.0], device=device),
    )
    assert torch.allclose(
        mirrored[:, SLICES["velocity_commands"]],
        obs[:, SLICES["velocity_commands"]]
        * torch.tensor([1.0, -1.0, -1.0], device=device),
    )


def test_mirror_swaps_left_right_joints(symmetry_mod, fake_symmetry_env, device):
    n = len(JOINT_NAMES)
    data = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(0)
    mirrored = symmetry_mod._mirror_joint_vector(fake_symmetry_env, data)

    name_to_idx = {n: i for i, n in enumerate(JOINT_NAMES)}
    for left, right in symmetry_mod._MIRROR_JOINT_PAIRS:
        li, ri = name_to_idx[left], name_to_idx[right]
        assert mirrored[0, li].item() == pytest.approx(data[0, ri].item())
        assert mirrored[0, ri].item() == pytest.approx(data[0, li].item())

    # unpaired (Spine) stays put
    assert mirrored[0, name_to_idx["Spine"]].item() == pytest.approx(
        data[0, name_to_idx["Spine"]].item()
    )


def test_mirror_joint_vector_is_involution(symmetry_mod, fake_symmetry_env, device):
    data = torch.randn(4, len(JOINT_NAMES), device=device)
    once = symmetry_mod._mirror_joint_vector(fake_symmetry_env, data)
    twice = symmetry_mod._mirror_joint_vector(fake_symmetry_env, once)
    assert torch.allclose(twice, data)


def test_mirror_policy_obs_is_involution(
    symmetry_mod, fake_symmetry_env, device, policy_obs_dim
):
    obs = _make_policy_obs(3, device)
    once = symmetry_mod._mirror_policy_obs(fake_symmetry_env, obs)
    twice = symmetry_mod._mirror_policy_obs(fake_symmetry_env, once)
    assert torch.allclose(twice, obs)


def test_compute_symmetric_states_augments_batch(
    symmetry_mod, fake_symmetry_env, device, policy_obs_dim
):
    batch = 2
    policy = _make_policy_obs(batch, device)
    obs = TensorDict({"policy": policy}, batch_size=[batch])
    actions = torch.randn(batch, len(JOINT_NAMES), device=device)

    obs_aug, actions_aug = symmetry_mod.compute_symmetric_states(
        fake_symmetry_env, obs, actions
    )

    assert obs_aug is not None and actions_aug is not None
    assert obs_aug.batch_size[0] == batch * 2
    assert actions_aug.shape == (batch * 2, len(JOINT_NAMES))
    assert torch.allclose(obs_aug["policy"][:batch], policy)
    assert torch.allclose(actions_aug[:batch], actions)
    assert torch.allclose(
        obs_aug["policy"][batch:],
        symmetry_mod._mirror_policy_obs(fake_symmetry_env, policy),
    )
    assert torch.allclose(
        actions_aug[batch:],
        symmetry_mod._mirror_joint_vector(fake_symmetry_env, actions),
    )


def test_compute_symmetric_states_none_branches(
    symmetry_mod, fake_symmetry_env, device, policy_obs_dim
):
    obs_aug, actions_aug = symmetry_mod.compute_symmetric_states(
        fake_symmetry_env, None, None
    )
    assert obs_aug is None and actions_aug is None

    policy = _make_policy_obs(1, device)
    obs = TensorDict({"policy": policy}, batch_size=[1])
    obs_aug, actions_aug = symmetry_mod.compute_symmetric_states(
        fake_symmetry_env, obs, None
    )
    assert obs_aug is not None and actions_aug is None

    actions = torch.zeros(1, len(JOINT_NAMES), device=device)
    obs_aug, actions_aug = symmetry_mod.compute_symmetric_states(
        fake_symmetry_env, None, actions
    )
    assert obs_aug is None and actions_aug is not None


def test_joint_mirror_map_caches(symmetry_mod, fake_symmetry_env):
    a = symmetry_mod._get_joint_mirror_map(fake_symmetry_env)
    b = symmetry_mod._get_joint_mirror_map(fake_symmetry_env)
    assert a[0] is b[0] and a[1] is b[1]


def test_unknown_mirror_joint_raises(symmetry_mod, device):
    from types import SimpleNamespace

    bad_robot = SimpleNamespace(
        data=SimpleNamespace(joint_names=["Spine", "Lbody_hip"])
    )
    env = SimpleNamespace(device=device, scene={"robot": bad_robot})
    with pytest.raises(KeyError):
        symmetry_mod._get_joint_mirror_map(env)
