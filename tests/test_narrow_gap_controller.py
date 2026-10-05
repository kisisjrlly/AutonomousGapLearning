import math

import torch

from agl.attempt.executor import (
    ControllerConfig,
    _gap_entry_ready,
    _roll_schedule,
)
from agl.config import load_config
from agl.data.generate_attempt_dataset import _set_v0_fixed_geometry
from agl.sim.env import GapEnv


def test_roll_schedule_has_acquire_hold_and_release_phases():
    c = ControllerConfig(
        roll_target_rad=math.radians(65), roll_start_distance=1.0,
        roll_hold_distance=.25, roll_prebias_distance=.2,
        roll_release_distance=.4,
    )
    x = torch.tensor([1.0, 2.4, 2.8, 3.3, 3.8])
    out = _roll_schedule(x, torch.full_like(x, 3.0), torch.full_like(x, .15), c)
    assert abs(float(out[0])) < 1e-6
    assert 0.0 < float(out[1]) < c.roll_target_rad
    assert abs(float(out[2]) - c.roll_target_rad) < 1e-5
    assert abs(float(out[-1])) < 1e-6


def test_geometry_gate_accepts_centered_rolled_pose_only():
    cfg = load_config()
    cfg.sim.n_envs = 2
    cfg.sim.device = "cpu"
    env = GapEnv(cfg, "cpu", difficulty=1.0)
    _set_v0_fixed_geometry(env.task, cfg)
    env.state["p"][0] = torch.tensor([2.9, 0.0, 1.5])
    env.state["p"][1] = torch.tensor([2.9, 0.0, 1.5])
    half = math.radians(65) / 2
    env.state["q"][0] = torch.tensor([math.cos(half), math.sin(half), 0.0, 0.0])
    env.state["q"][1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    c = ControllerConfig(roll_target_rad=math.radians(65))
    ready = _gap_entry_ready(env, torch.tensor([math.radians(65), 0.0]), c)
    assert ready.tolist() == [True, False]
