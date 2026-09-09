"""Shared fixtures and helpers for mark_tasks unit tests (no Isaac Sim)."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from helpers import JOINT_NAMES, MARK_V1_MDP, POLICY_TERM_DIMS, POLICY_TERM_NAMES


def _load_module(module_name: str, path: Path):
    """Load a .py file as a module without executing package __init__ chains."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def symmetry_mod():
    pytest.importorskip("tensordict")
    return _load_module("mark_mdp_symmetry", MARK_V1_MDP / "symmetry.py")


@pytest.fixture(scope="session")
def rewards_mod():
    # rewards.py imports isaaclab types; stub them so tests run without Isaac Lab.
    if "isaaclab.managers" not in sys.modules:
        managers = types.ModuleType("isaaclab.managers")
        managers.SceneEntityCfg = object  # type: ignore[attr-defined]
        sensors = types.ModuleType("isaaclab.sensors")
        sensors.ContactSensor = object  # type: ignore[attr-defined]
        isaaclab = types.ModuleType("isaaclab")
        sys.modules.setdefault("isaaclab", isaaclab)
        sys.modules["isaaclab.managers"] = managers
        sys.modules["isaaclab.sensors"] = sensors
    return _load_module("mark_mdp_rewards", MARK_V1_MDP / "rewards.py")


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def fake_symmetry_env(device):
    """Minimal env stub matching what compute_symmetric_states / mirror helpers need."""
    robot = SimpleNamespace(data=SimpleNamespace(joint_names=list(JOINT_NAMES)))
    observation_manager = SimpleNamespace(
        active_terms={"policy": list(POLICY_TERM_NAMES)},
        group_obs_term_dim={"policy": list(POLICY_TERM_DIMS)},
    )
    env = SimpleNamespace(
        device=device,
        scene={"robot": robot},
        observation_manager=observation_manager,
        unwrapped=None,
    )
    env.unwrapped = env
    return env


@pytest.fixture
def policy_obs_dim():
    return sum(math.prod(d) for d in POLICY_TERM_DIMS)
