"""Safe-abort action sensitivity diagnostic, NOT a causal adaptation test."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from ..config import load_config
from ..models.policy import Policy
from ..sim.env import GapEnv, OUTCOME, PRIV_DIM
from .evaluate import make_bank


@torch.no_grad()
def run_verification(model, cfg, n_episodes, device):
    if n_episodes <= 0:
        raise ValueError("n must be positive")
    torch.manual_seed(9999)
    ecfg = copy.deepcopy(cfg)
    ecfg.sim.n_envs = n_episodes
    ecfg.curriculum.enabled = False
    env = GapEnv(ecfg, device, difficulty=1.)
    env._reset_envs(torch.arange(n_episodes, device=device),
                    tasks=make_bank(ecfg, 'id', n_episodes, device))
    h = model.init_hidden(n_episodes, device)
    obs = env.observe()
    priv = torch.zeros(n_episodes, PRIV_DIM, device=device)
    finished = torch.zeros(n_episodes, dtype=torch.bool, device=device)
    sampled = finished.clone()
    diffs, task_ids = [], []
    for _ in range(ecfg.sim.ep_len + 2):
        mean, _, _, h_new = model.step(obs['img'], obs['vec'], priv, h)
        obs, _, done, info = env.step(mean.clamp(-1, 1))
        priv = info['priv']
        eligible = (info['end_event'] & (info['end_outcome'] == OUTCOME['abort'])
                    & ~done & ~finished & ~sampled & ~info['collision'])
        idx = eligible.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel():
            img, vec = obs['img'][idx], obs['vec'][idx]
            remembered, _ = model.core(img, vec, h_new[idx])
            fresh, _ = model.core(img, vec, model.init_hidden(len(idx), device))
            a = model.actor(model.act_in(remembered)).clamp(-1, 1)
            b = model.actor(model.act_in(fresh)).clamp(-1, 1)
            diffs.extend((a-b).cpu().tolist())
            task_ids.extend(idx.cpu().tolist())
            sampled |= eligible
        finished |= done
        wipe = done.clone()
        if cfg.model.reset_between_attempts:
            wipe |= info['end_event']
        h = h_new * (~wipe).unsqueeze(-1)
        if finished.all():
            break
    return {'action_diffs': np.asarray(diffs).reshape(-1, 4),
            'task_ids': task_ids, 'n_tasks': n_episodes}


def analyze_results(results):
    norms = np.linalg.norm(results['action_diffs'], axis=1)
    return {'n_tasks': results['n_tasks'], 'n_eligible': len(norms),
            'mean_diff': float(norms.mean()) if len(norms) else None,
            'median_diff': float(np.median(norms)) if len(norms) else None,
            'conclusion': ('sensitivity_only_not_adaptation_evidence' if len(norms)
                           else 'insufficient_safe_abort_samples'),
            'action_dimensions': ['thrust', 'roll_rate', 'pitch_rate', 'yaw_rate']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    cfg = load_config(overrides=ck['cfg'])
    model = Policy(cfg).to(args.device)
    model.load_state_dict(ck['model'])
    model.eval()
    results = run_verification(model, cfg, args.n, args.device)
    stats = analyze_results(results)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'gru_verification_stats.json').open('x') as f:
        json.dump(stats, f, indent=2, allow_nan=False)
    with (args.out / 'gru_verification_data.npz').open('xb') as f:
        np.savez(f, **results)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
