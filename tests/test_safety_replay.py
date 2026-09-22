import pytest
import torch

from agl.sim.safety import replay_trace


def test_replay_aligns_pre_action_gate_with_following_contact():
    clear_pre = torch.tensor([[.8, .1, .1], [.8, .4, .2]])
    speed_pre = torch.tensor([[1., 1., -1.], [0., 0., 0.]])
    contact_after = torch.tensor([[True, False, False], [False, False, False]])
    out = replay_trace(clear_pre, speed_pre, torch.tensor(4.), .1,
                       contact_after=contact_after)
    assert out['contacts'] == 1
    assert out['contact_allowed_count'] == 1
    assert out['noncontact_rejected_count'] == 1
    assert out['allowed'] == 5
    assert out['gate'].shape == (2, 3)


def test_replay_rejects_misaligned_transition_traces():
    with pytest.raises(ValueError):
        replay_trace(torch.zeros(2), torch.zeros(3), torch.tensor(1.), .1,
                     contact_after=torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError):
        replay_trace(torch.zeros(2), torch.zeros(2), torch.tensor(1.), .1,
                     contact_after=torch.zeros(3, dtype=torch.bool))
    with pytest.raises(ValueError):
        replay_trace(torch.zeros(2, 2), torch.zeros(2, 2), torch.ones(3), .1,
                     contact_after=torch.zeros(2, 2, dtype=torch.bool))
