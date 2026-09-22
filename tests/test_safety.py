import pytest
import torch

from agl.sim.safety import (braking_accel, retreat_gate, safe_to_continue,
                            stopping_distance)


def test_stopping_distance_and_gate_are_conservative():
    speed = torch.tensor([0., 1., 2., -1.])
    accel = torch.full_like(speed, 4.)
    d = stopping_distance(speed, accel, .1)
    assert torch.allclose(d, torch.tensor([0., .225, .7, .225]))
    clear = torch.tensor([.1, .3, .7, 0.])
    allowed = safe_to_continue(clear, speed, accel, .1)
    assert allowed.tolist() == [True, True, True, True]
    assert not retreat_gate(torch.tensor([0.]), torch.tensor([-2.]),
                        torch.tensor([1.]), .5).item()


def test_uncertainty_and_margin_can_only_reject():
    clear = torch.tensor([.5])
    speed = torch.tensor([1.])
    accel = torch.tensor([4.])
    assert safe_to_continue(clear, speed, accel, .1).item()
    assert not safe_to_continue(clear, speed, accel, .1,
                                uncertainty=torch.tensor([.3])).item()
    assert not safe_to_continue(clear, speed, accel, .1, margin=.3).item()
    with pytest.raises(ValueError):
        stopping_distance(speed, accel, -1.)
    with pytest.raises(ValueError):
        safe_to_continue(clear, speed, accel, .1, margin=-1.)


def test_braking_accel_uses_lower_bound():
    dyn = {'tmax': torch.tensor([9.81, 19.62]),
           'mass': torch.tensor([1., 1.])}
    assert torch.equal(braking_accel(dyn), torch.tensor([0., 9.81]))
