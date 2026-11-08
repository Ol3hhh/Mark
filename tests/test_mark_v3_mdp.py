"""Pure tensor tests for mark_v3 observations."""

from __future__ import annotations

import math

import torch


def test_wheel_phase_is_periodic(observations_v3_mod):
    angles = torch.tensor([[0.0], [0.5 * math.pi], [math.pi], [2.0 * math.pi]])

    features = observations_v3_mod.phase_features(angles)

    expected = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ]
    )
    torch.testing.assert_close(features, expected, atol=1.0e-6, rtol=0.0)


def test_contact_state_reduces_filtered_force_matrix(observations_v3_mod):
    forces = torch.zeros(3, 1, 1, 3)
    forces[1, 0, 0, 2] = 4.9
    forces[2, 0, 0, 2] = 5.1

    contacts = observations_v3_mod.contact_state_from_force_matrix(
        forces, threshold=5.0
    )

    torch.testing.assert_close(contacts, torch.tensor([0.0, 0.0, 1.0]))
