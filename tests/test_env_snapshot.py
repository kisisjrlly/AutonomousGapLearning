import pytest
import torch

from agl.config import load_config
from agl.sim import render
from agl.sim.env import GapEnv


def test_snapshot_restore_replays_identical_observation_and_transition():
    cfg = load_config()
    cfg.sim.n_envs = 2
    cfg.sim.device = "cpu"
    env = GapEnv(cfg, "cpu", difficulty=1.)
    snap = env.snapshot()
    actions = [torch.tensor([[0., .1, -.2, .05], [0., -.1, .2, -.05]]) for _ in range(5)]
    first = []
    for action in actions:
        obs, rew, done, info = env.step(action)
        first.append((
            obs["img"].clone(), obs["vec"].clone(), rew.clone(), done.clone(),
            info["clearance"].clone(), env.state["p"].clone(),
            env.frame_delay_buf.clone(), torch.tensor(env.frame_buf_ptr),
        ))
    env.restore(snap)
    second = []
    for action in actions:
        obs, rew, done, info = env.step(action)
        second.append((
            obs["img"].clone(), obs["vec"].clone(), rew.clone(), done.clone(),
            info["clearance"].clone(), env.state["p"].clone(),
            env.frame_delay_buf.clone(), torch.tensor(env.frame_buf_ptr),
        ))
    for a, b in zip(first, second):
        for x, y in zip(a, b):
            assert torch.equal(x, y)


def test_snapshot_difficulty_and_config():
    cfg = load_config()
    cfg.sim.n_envs = 2
    cfg.sim.device = "cpu"
    env = GapEnv(cfg, "cpu", difficulty=.7)
    snap = env.snapshot()
    env.difficulty = .1
    env.restore(snap)
    assert env.difficulty == .7
    cfg.sim.dt_ctrl *= 2
    with pytest.raises(ValueError, match="config"):
        env.restore(snap)


def test_configurable_image_delay_zero_vs_one(monkeypatch):
    def fake_render(p, q, task, rays, sensor):
        # Encode x position in every pixel so temporal alignment is unambiguous.
        return p[:, 0, None, None, None].expand(
            -1, 3, sensor.img_h, sensor.img_w
        ).clone()

    monkeypatch.setattr(render, "render", fake_render)

    def make(delay):
        cfg = load_config()
        cfg.sim.n_envs = 1
        cfg.sim.device = "cpu"
        cfg.sensor.img_delay_steps = delay
        env = GapEnv(cfg, "cpu", difficulty=1.)
        env.state["v"].zero_()
        env.state["v"][:, 0] = 1.0
        env.task["dyn"]["wind_steady"].zero_()
        env.task["dyn"]["gust_sigma"].zero_()
        action = torch.zeros(1, 4)
        action[:, 0] = -1.0
        before = env.frame.clone()
        obs, *_ = env.step(action)
        return env, before, obs["img"]

    env0, before0, img0 = make(0)
    env1, before1, img1 = make(1)
    assert not torch.equal(img0, before0)
    assert torch.equal(img1, before1)
    assert env0.frame_delay_buf.shape[1] == 1
    assert env1.frame_delay_buf.shape[1] == 2


def test_negative_image_delay_rejected():
    cfg = load_config()
    cfg.sim.n_envs = 1
    cfg.sim.device = "cpu"
    cfg.sensor.img_delay_steps = -1
    with pytest.raises(ValueError, match="img_delay_steps"):
        GapEnv(cfg, "cpu")
