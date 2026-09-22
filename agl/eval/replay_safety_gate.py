"""Replay the safety gate on an evaluation NPZ; no policy or training is run."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..sim.safety import replay_trace


def replay_npz(path, reaction_time=.05, margin=0.):
    data = np.load(path)
    required = ('rec_clear', 'rec_v')
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"missing fields: {missing}")
    clear = torch.from_numpy(data['rec_clear'])
    speed = torch.from_numpy(data['rec_v'])[..., 0]
    # Evaluation trajectories use per-task hidden dynamics when available.
    if 'task_mass' in data and 'task_twr' in data:
        accel = (torch.from_numpy(data['task_twr']) * 9.81 - 9.81)
        accel = accel.clamp_min(0.)
        while accel.ndim < clear.ndim:
            accel = accel.unsqueeze(0)
    else:
        raise KeyError('task_twr and task_mass required; no assumed braking authority')
    # A signed x speed is the forward component in this environment.
    out = replay_trace(clear, speed, accel, reaction_time, margin=margin)
    return {k: v for k, v in out.items() if not torch.is_tensor(v)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('npz', type=Path)
    p.add_argument('--reaction-time', type=float, default=.05)
    p.add_argument('--margin', type=float, default=0.)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    result = replay_npz(args.npz, args.reaction_time, args.margin)
    if args.out:
        with args.out.open('x') as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
