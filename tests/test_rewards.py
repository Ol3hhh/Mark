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


def _hip_env(joint_pos, default_pos, command_xy: torch.Tensor):
    asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=joint_pos,
            default_joint_pos=default_pos,
        )
    )
    return SimpleNamespace(
        scene={"robot": asset},
        command_manager=SimpleNamespace(get_command=lambda _name: command_xy),
    )


def test_hip_antiphase_rewards_opposite_hips(rewards_v2_mod, device):
    # L +0.2, R -0.3 → product negative → reward 0.06
    q = torch.tensor([[0.2, -0.3]], device=device)
    q0 = torch.zeros(1, 2, device=device)
    cmd = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    env = _hip_env(q, q0, cmd)
    cfg = SimpleNamespace(name="robot", joint_ids=slice(None))

    reward = rewards_v2_mod.hip_antiphase_impl(
        env, command_name="base_velocity", asset_cfg=cfg
    )
    assert reward.item() == pytest.approx(0.06)


def test_hip_antiphase_zero_when_in_phase(rewards_v2_mod, device):
    q = torch.tensor([[0.2, 0.3]], device=device)
    q0 = torch.zeros(1, 2, device=device)
    cmd = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    env = _hip_env(q, q0, cmd)
    cfg = SimpleNamespace(name="robot", joint_ids=slice(None))

    reward = rewards_v2_mod.hip_antiphase_impl(
        env, command_name="base_velocity", asset_cfg=cfg
    )
    assert reward.item() == pytest.approx(0.0)


def test_hip_antiphase_gated_by_command(rewards_v2_mod, device):
    q = torch.tensor([[0.2, -0.3]], device=device)
    q0 = torch.zeros(1, 2, device=device)
    cmd = torch.tensor([[0.05, 0.0, 0.0]], device=device)
    env = _hip_env(q, q0, cmd)
    cfg = SimpleNamespace(name="robot", joint_ids=slice(None))

    reward = rewards_v2_mod.hip_antiphase_impl(
        env, command_name="base_velocity", asset_cfg=cfg
    )
    assert reward.item() == pytest.approx(0.0)


def test_hip_antiphase_capped_by_max_value(rewards_v2_mod, device):
    # product 0.8 * 0.5 = 0.4, capped to 0.1
    q = torch.tensor([[0.8, -0.5]], device=device)
    q0 = torch.zeros(1, 2, device=device)
    cmd = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    env = _hip_env(q, q0, cmd)
    cfg = SimpleNamespace(name="robot", joint_ids=slice(None))

    reward = rewards_v2_mod.hip_antiphase_impl(
        env, command_name="base_velocity", asset_cfg=cfg, max_value=0.1
    )
    assert reward.item() == pytest.approx(0.1)


def test_motion_sync_contralateral_and_cap(rewards_v2_mod, device):
    # [L_hip, R_hip, L_arm, R_arm]: L_hip fwd with R_arm fwd, R_hip back with L_arm back
    q = torch.tensor([[0.5, -0.4, -0.3, 0.6]], device=device)
    q0 = torch.zeros(1, 4, device=device)
    cmd = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    env = _hip_env(q, q0, cmd)
    cfg = SimpleNamespace(name="robot", joint_ids=slice(None))

    raw = rewards_v2_mod.motion_synchronisation_impl(
        env, command_name="base_velocity", asset_cfg=cfg
    )
    assert raw.item() == pytest.approx(0.5 * 0.6 + (-0.4) * (-0.3))
    capped = rewards_v2_mod.motion_synchronisation_impl(
        env, command_name="base_velocity", asset_cfg=cfg, max_value=0.2
    )
    assert capped.item() == pytest.approx(0.2)


def _air_env(left_last, right_last, command_xy: torch.Tensor):
    left = SimpleNamespace(data=SimpleNamespace(last_air_time=left_last))
    right = SimpleNamespace(data=SimpleNamespace(last_air_time=right_last))
    return _env(left, right, command_xy)


def test_air_time_asymmetry_penalises_dominant_leg(rewards_v2_mod, device):
    env = _air_env(
        torch.tensor([[0.5], [0.4]], device=device),
        torch.tensor([[0.1], [0.4]], device=device),
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device),
    )
    out = rewards_v2_mod.feet_air_time_asymmetry(
        env,
        command_name="base_velocity",
        left_sensor_cfg=_cfg("left_foot_contact"),
        right_sensor_cfg=_cfg("right_foot_contact"),
    )
    assert out[0].item() == pytest.approx(0.16)
    assert out[1].item() == pytest.approx(0.0)


def test_air_time_asymmetry_clamped_and_gated(rewards_v2_mod, device):
    env = _air_env(
        torch.tensor([[5.0], [0.5]], device=device),
        torch.tensor([[0.0], [0.0]], device=device),
        torch.tensor([[1.0, 0.0, 0.0], [0.05, 0.0, 0.0]], device=device),
    )
    out = rewards_v2_mod.feet_air_time_asymmetry(
        env,
        command_name="base_velocity",
        left_sensor_cfg=_cfg("left_foot_contact"),
        right_sensor_cfg=_cfg("right_foot_contact"),
        max_air_time=1.0,
    )
    assert out[0].item() == pytest.approx(1.0)  # 5.0 clamped to 1.0
    assert out[1].item() == pytest.approx(0.0)  # zero command


def test_yaw_rate_l2(rewards_v2_mod, device):
    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_ang_vel_w=torch.tensor([[0.1, 0.2, -0.3]], device=device)
        )
    )
    env = SimpleNamespace(scene={"robot": asset})
    out = rewards_v2_mod.yaw_rate_l2(env, asset_cfg=SimpleNamespace(name="robot"))
    assert out.item() == pytest.approx(0.09)
