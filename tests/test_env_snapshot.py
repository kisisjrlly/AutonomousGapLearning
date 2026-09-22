import torch

from agl.config import load_config
from agl.sim.env import GapEnv


def test_snapshot_restore_replays_identical_observation_and_transition():
    cfg = load_config(); cfg.sim.n_envs = 2; cfg.sim.device = 'cpu'
    env = GapEnv(cfg, 'cpu', difficulty=1.)
    snap = env.snapshot()
    actions = [torch.tensor([[0., .1, -.2, .05], [0., -.1, .2, -.05]]) for _ in range(5)]
    first = []
    for action in actions:
        obs, rew, done, info = env.step(action)
        first.append((obs['img'].clone(), obs['vec'].clone(), rew.clone(), done.clone(),
                      info['clearance'].clone(), env.state['p'].clone()))
    env.restore(snap)
    second = []
    for action in actions:
        obs, rew, done, info = env.step(action)
        second.append((obs['img'].clone(), obs['vec'].clone(), rew.clone(), done.clone(),
                       info['clearance'].clone(), env.state['p'].clone()))
    for a, b in zip(first, second):
        for x, y in zip(a, b):
            assert torch.equal(x, y)
