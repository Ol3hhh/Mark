"""Unit tests for mark_v1 split-sensor feet air-time reward."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _sensor(air_time: torch.Tensor, contact_time: torch.Tensor):
    return SimpleNamespace(
        data=SimpleNamespace(
            current_air_time=air_time,
            current_contact_time=contact_time,
        )
    )


def _cfg(name: str, body_ids: list[int] | slice = slice(None)):
    return SimpleNamespace(name=name, body_ids=body_ids)


def _env(left, right, command_xy: torch.Tensor):
    command_manager = SimpleNamespace(
        get_command=lambda _name: command_xy,
    )
    return SimpleNamespace(
        scene=SimpleNamespace(
            sensors={"left_foot_contact": left, "right_foot_contact": right}
        ),
        command_manager=command_manager,
    )


def test_single_stance_rewards_mode_time(rewards_mod, device):
    # left in contact (contact_time=0.4), right in air (air_time=0.5) -> single stance
    left = _sensor(
        air_time=torch.tensor([[0.0]], device=device),
        contact_time=torch.tensor([[0.4]], device=device),
    )
    right = _sensor(
        air_time=torch.tensor([[0.5]], device=device),
        contact_time=torch.tensor([[0.0]], device=device),
    )
    cmd = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    env = _env(left, right, cmd)

    reward = rewards_mod.feet_air_time_positive_biped_split(
        env,
        command_name="base_velocity",
        threshold=0.6,
        left_sensor_cfg=_cfg("left_foot_contact"),
        right_sensor_cfg=_cfg("right_foot_contact"),
    )
    # min of in_mode_time on single stance = min(0.4, 0.5) = 0.4
    assert reward.shape == (1,)
    assert reward.item() == pytest.approx(0.4)


def test_double_contact_gives_zero(rewards_mod, device):
    left = _sensor(
        air_time=torch.tensor([[0.0]], device=device),
        contact_time=torch.tensor([[0.3]], device=device),
    )
    right = _sensor(
        air_time=torch.tensor([[0.0]], device=device),
        contact_time=torch.tensor([[0.2]], device=device),
    )
    cmd = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    env = _env(left, right, cmd)

    reward = rewards_mod.feet_air_time_positive_biped_split(
        env,
        command_name="base_velocity",
        threshold=0.6,
        left_sensor_cfg=_cfg("left_foot_contact"),
        right_sensor_cfg=_cfg("right_foot_contact"),
    )
    assert reward.item() == pytest.approx(0.0)


def test_zero_command_gates_reward(rewards_mod, device):
    left = _sensor(
        air_time=torch.tensor([[0.0]], device=device),
        contact_time=torch.tensor([[0.4]], device=device),
    )
    right = _sensor(
        air_time=torch.tensor([[0.5]], device=device),
        contact_time=torch.tensor([[0.0]], device=device),
    )
    cmd = torch.tensor([[0.05, 0.0, 0.0]], device=device)  # ||xy|| <= 0.1
    env = _env(left, right, cmd)

    reward = rewards_mod.feet_air_time_positive_biped_split(
        env,
        command_name="base_velocity",
        threshold=0.6,
        left_sensor_cfg=_cfg("left_foot_contact"),
        right_sensor_cfg=_cfg("right_foot_contact"),
    )
    assert reward.item() == pytest.approx(0.0)


def test_reward_clamped_to_threshold(rewards_mod, device):
    left = _sensor(
        air_time=torch.tensor([[0.0]], device=device),
        contact_time=torch.tensor([[2.0]], device=device),
    )
    right = _sensor(
        air_time=torch.tensor([[2.0]], device=device),
        contact_time=torch.tensor([[0.0]], device=device),
    )
    cmd = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    env = _env(left, right, cmd)

    reward = rewards_mod.feet_air_time_positive_biped_split(
        env,
        command_name="base_velocity",
        threshold=0.6,
        left_sensor_cfg=_cfg("left_foot_contact"),
        right_sensor_cfg=_cfg("right_foot_contact"),
    )
    assert reward.item() == pytest.approx(0.6)


def test_batch_of_envs(rewards_mod, device):
    left = _sensor(
        air_time=torch.tensor([[0.0], [0.0]], device=device),
        contact_time=torch.tensor([[0.4], [0.3]], device=device),
    )
    right = _sensor(
        air_time=torch.tensor([[0.5], [0.0]], device=device),
        contact_time=torch.tensor([[0.0], [0.2]], device=device),
    )
    # env0 moving (single stance), env1 moving but double contact
    cmd = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)
    env = _env(left, right, cmd)

    reward = rewards_mod.feet_air_time_positive_biped_split(
        env,
        command_name="base_velocity",
        threshold=0.6,
        left_sensor_cfg=_cfg("left_foot_contact"),
        right_sensor_cfg=_cfg("right_foot_contact"),
    )
    assert reward.shape == (2,)
    assert reward[0].item() == pytest.approx(0.4)
    assert reward[1].item() == pytest.approx(0.0)
