"""Prepare matched same-state branches for history intervention studies.

This tool does not update weights and does not claim adaptation. It verifies
that paired GapEnv tasks can be branched from an identical physical snapshot;
the caller may later inject different history into a frozen policy.
"""
import argparse
import copy
import json

import torch

from ..config import load_config
from ..sim import dynamics, scene
from ..sim.env import GapEnv


def _clone_state(env):
    return {k: v.clone() for k, v in env.state.items()}


@torch.no_grad()
def prepare(n_pairs=4, steps=12, device='cpu', seed=20260922):
    if n_pairs <= 0 or steps <= 0:
        raise ValueError('n_pairs and steps must be positive')
    torch.manual_seed(seed)
    cfg = copy.deepcopy(load_config())
    cfg.sim.n_envs = 2 * n_pairs
    cfg.sim.device = device
    cfg.task.info_gate_enabled = True
    cfg.curriculum.enabled = False
    env = GapEnv(cfg, device, difficulty=1.)
    bank = scene.paired_information_tasks(n_pairs, cfg, 1., device)
    ids = torch.arange(2*n_pairs, device=device)
    env._reset_envs(ids, tasks=bank)
    env.task['dyn']['wind_steady'].zero_()
    env.task['dyn']['gust_sigma'].zero_()
    env.state['wind'].zero_()
    # Identical initial state and hover command for each pair.
    # Start just inside the probe zone so a short deterministic rollout
    # produces a measurable latent disturbance while the pair remains matched.
    env.state['p'][:, 0] = (env.task['wall_x'] - cfg.task.info_probe_distance
                            + cfg.task.info_probe_ramp + 0.02)
    env.state['p'][:, 1] = 0.0
    env.state['p'][:, 2] = 1.5
    env.state['v'].zero_(); env.state['w'].zero_()
    env.state['q'].zero_(); env.state['q'][:, 0] = 1.
    env.state['thrust'] = env.task['dyn']['mass'] * 9.81
    action = torch.zeros(2*n_pairs, 4, device=device)
    action[:, 0] = dynamics.hover_thrust_action(env.task['dyn'])
    env.prev_action = action.clone(); env.delay_buf[:] = action.unsqueeze(1)
    pre = _clone_state(env)
    probe_min_clear = torch.full((2*n_pairs,), float('inf'), device=device)
    probe_done = torch.zeros(2*n_pairs, dtype=torch.bool, device=device)
    probe_contact = torch.zeros_like(probe_done)
    for _ in range(steps):
        _, _, done, info = env.step(action)
        probe_min_clear = torch.minimum(probe_min_clear, info['clearance'])
        probe_done |= done
        probe_contact |= info['collision']
        if probe_done.any() or probe_contact.any():
            raise RuntimeError('probe was not safe: terminal/reset/contact occurred')
    post = _clone_state(env)
    pair_delta = {k: float((v[0::2] - v[1::2]).abs().max().item())
                  for k, v in post.items() if torch.is_tensor(v)}
    return {'scope': 'same_state_branch_preparation_NOT_adaptation_evidence',
            'n_pairs': n_pairs, 'steps': steps, 'seed': seed,
            'pre_state': pre, 'post_state': post,
            'probe_min_clearance': probe_min_clear.detach().cpu(),
            'probe_done': probe_done.detach().cpu(),
            'probe_contact': probe_contact.detach().cpu(),
            'post_pair_max_abs_delta': pair_delta,
            'probe_wind': env.task['dyn']['probe_wind'].detach().cpu(),
            'state_schema': sorted(pre)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=int, default=4)
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    result = prepare(a.pairs, a.steps, a.device)
    # Save tensors separately to avoid JSON silently losing precision.
    serial = {k: v for k, v in result.items() if not torch.is_tensor(v)
              and k not in ('pre_state', 'post_state')}
    torch.save({'pre_state': result['pre_state'], 'post_state': result['post_state'],
                'probe_wind': result['probe_wind']}, a.out + '.pt')
    with open(a.out + '.json', 'x') as f:
        json.dump(serial, f, indent=2)
    print(json.dumps(serial, indent=2))


if __name__ == '__main__':
    main()
