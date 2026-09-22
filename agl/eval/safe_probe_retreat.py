"""Scripted 6-DoF probe/stop/retreat acceptance rollout for GapEnv.

This is a conservative controller baseline, not a learned policy. It uses CTBR
actions and rejects any episode with contact, terminal/reset, or non-positive
clearance. A failed rollout is reported, never converted into a success.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from ..config import load_config
from ..sim import dynamics, scene
from ..sim.env import GapEnv


def _hover(env):
    a = torch.zeros(env.n, 4, device=env.dev)
    a[:, 0] = dynamics.hover_thrust_action(env.task['dyn'])
    return a


@torch.no_grad()
def run(n_pairs=4, probe_steps=12, stop_steps=8, retreat_steps=24, device='cpu', seed=0, record=None):
    if min(n_pairs, probe_steps, stop_steps, retreat_steps) <= 0:
        raise ValueError('all step counts must be positive')
    if record is not None and Path(record).exists():
        raise FileExistsError(record)
    torch.manual_seed(seed)
    cfg = load_config(); cfg.sim.n_envs = 2*n_pairs; cfg.sim.device = device
    cfg.task.info_gate_enabled = True; cfg.curriculum.enabled = False
    env = GapEnv(cfg, device, difficulty=1.)
    bank = scene.paired_information_tasks(n_pairs, cfg, 1., device)
    env._reset_envs(torch.arange(env.n, device=device), tasks=bank)
    env.task['dyn']['wind_steady'].zero_(); env.task['dyn']['gust_sigma'].zero_(); env.state['wind'].zero_()
    env.state['p'][:, 0] = env.task['wall_x'] - cfg.task.info_probe_distance + .02
    env.state['p'][:, 1:] = 0.; env.state['p'][:, 2] = 1.5
    env.state['v'].zero_(); env.state['w'].zero_(); env.state['q'].zero_(); env.state['q'][:,0]=1.
    env.state['thrust'] = env.task['dyn']['mass']*9.81
    hover = _hover(env); env.prev_action = hover.clone(); env.delay_buf[:] = hover.unsqueeze(1)
    minimum = torch.full((env.n,), float('inf'), device=device)
    contacts = torch.zeros(env.n, dtype=torch.bool, device=device); dones = contacts.clone()
    phases = []
    initial_p = env.state['p'].clone()
    task_record = {f'task_{k}': v.cpu().numpy().copy() for k, v in env.task.items()
                   if isinstance(v, torch.Tensor)}
    trace = {k: [] for k in ('p', 'q', 'v', 'act', 'clear', 'attempt_id', 'collision', 'done', 'phase')}
    phase_end = {}
    # Hover probe: latent wind is physically experienced, not read from task.
    schedule = [('probe', [hover]*probe_steps), ('stop', [hover]*stop_steps)]
    # Brake/retreat uses a conservative backward pitch command. The safety
    # acceptance below decides whether this candidate is admissible.
    retreat = hover.clone(); retreat[:, 2] = -0.12
    schedule.append(('retreat', [retreat]*retreat_steps))
    for phase, actions in schedule:
        for action in actions:
            # Record action-input state: env.step auto-resets terminal rows.
            for key in ('p', 'q', 'v'):
                trace[key].append(env.state[key].cpu().numpy().copy())
            trace['act'].append(action.cpu().numpy().copy())
            trace['phase'].append(np.full(env.n, phase))
            _, _, done, info = env.step(action)
            for key, value in [('clear', info['clearance']), ('attempt_id', info['attempt_id']),
                               ('collision', info['collision']), ('done', done)]:
                trace[key].append(value.cpu().numpy().copy())
            minimum = torch.minimum(minimum, info['clearance'])
            contacts |= info['collision']; dones |= done; phases.append(phase)
            if contacts.any() or dones.any():
                break
        if contacts.any() or dones.any(): break
        phase_end[phase] = {'speed_max': float(env.state['v'].norm(dim=-1).max()),
                            'x_displacement': (env.state['p'][:, 0]-initial_p[:, 0]).cpu().tolist()}
    success = ~(contacts | dones) & (minimum > 0)
    if record is not None:
        target = Path(record); target.parent.mkdir(parents=True, exist_ok=True)
        meta = {'scope': 'scripted CTBR schedule; NOT learned; stop/retreat NOT certified',
                'state_timing': 'pre_action; clearance/collision/done describe following action interval',
                'dt_ctrl': cfg.sim.dt_ctrl, 'seed': seed, 'info_gate_enabled': True,
                'info_probe_distance': cfg.task.info_probe_distance, 'info_probe_ramp': cfg.task.info_probe_ramp}
        with target.open('xb') as f:
            np.savez_compressed(f, **{f'rec_{k}': np.stack(v) for k,v in trace.items()},
                                **task_record, steps=np.full(env.n, len(phases)), meta=json.dumps(meta))
    return {'scope':'scripted_dynamics_probe_stop_retreat_NOT_learned_adaptation',
            'seed': seed, 'phase_end': phase_end, 'protocol_completed': False,
            'acceptance_scope': 'contact_free_only; stopping and recovered retry state not certified',
            'n_pairs':n_pairs, 'steps_executed':len(phases),
            'phase_at_stop': phases[-1] if phases else None,
            'safe_count':int(success.sum()), 'contact_count':int(contacts.sum()),
            'done_count':int(dones.sum()), 'unsafe_count':int((~success).sum()),
            'min_clearance':float(minimum.min().item()),
            'pair_post_y':None if dones.any() else (env.state['p'][0::2,1]-env.state['p'][1::2,1]).cpu().tolist()}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--pairs',type=int,default=4); p.add_argument('--out',required=True); p.add_argument('--device',default='cpu'); p.add_argument('--record'); p.add_argument('--seed',type=int,default=0); a=p.parse_args()
    if Path(a.out).exists(): raise FileExistsError(a.out)
    result=run(n_pairs=a.pairs,device=a.device,seed=a.seed,record=a.record)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out,'x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
