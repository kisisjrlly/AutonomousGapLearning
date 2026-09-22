import pytest
import torch

from agl.sim.safety import replay_trace


def test_replay_separates_gate_from_observed_contacts():
    out = replay_trace(torch.tensor([[.8, .1, -.1], [.8, .4, .2]]),
                       torch.tensor([[1., 1., -1.], [0., 0., 0.]]),
                       torch.tensor(4.), .1)
    assert out['contacts'] == 1
    assert out['contact_allowed_count'] == 0
    assert out['noncontact_rejected_count'] == 1
    assert out['allowed'] == 4
    assert out['gate'].shape == (2, 3)


def test_replay_rejects_mismatched_traces():
    with pytest.raises(ValueError):
        replay_trace(torch.zeros(2), torch.zeros(3), torch.tensor(1.), .1)
    with pytest.raises(ValueError):
        replay_trace(torch.zeros(2, 2), torch.zeros(2, 2), torch.ones(3), .1)
