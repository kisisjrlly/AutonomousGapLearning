"""Privileged-state feedback baseline; neither learned adaptation nor a safety shield.

Approach from outside the information gate, hold, retreat, settle. Every phase
has measured completion conditions and a deadline. Rejected rollouts never
produce a recovery snapshot. Uses ground-truth state/mass/thrust calibration.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..config import load_config
from ..sim import collision, dynamics, render, scene
from ..sim.env import GapEnv
from ..sim.maths import body_z_world, quat_rotate_inv


def feedback(env, target):
    """Bounded position/velocity feedback, thrust-axis attitude feedback."""
    st = env.state
    acc = (2.0 * (target-st['p']) - 2.6*st['v']).clamp(-1.2, 1.2)
    force = acc.clone(); force[:, 2] += dynamics.G
    desired_z = force / force.norm(dim=-1, keepdim=True)
    axis_error = torch.cross(body_z_world(st['q']), desired_z, dim=-1)
    rates = 5.0 * quat_rotate_inv(st['q'], axis_error) - .25*st['w']
    action = torch.zeros(env.n, 4, device=env.dev)
    action[:, 0] = 2 * force.norm(dim=-1)*env.task['dyn']['mass']/env.task['dyn']['tmax'] - 1
    action[:, 1:] = rates / dynamics.OMEGA_MAX.to(env.dev)
    return action.clamp(-1, 1)


def ready(state, target):
    """All rows must stay within these bounds for a dwell period."""
    return ((state['p']-target).norm(dim=-1) < .08) & (state['v'].norm(dim=-1) < .08) & (state['w'].norm(dim=-1) < .12)


@torch.no_grad()
def run(n_pairs=2, seed=0, phase_limit=280, dwell_steps=12, out_dir=None):
    if min(n_pairs, phase_limit, dwell_steps) <= 0:
        raise ValueError('counts must be positive')
    dest = Path(out_dir) if out_dir else None
    if dest:
        dest.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    cfg = load_config(); cfg.sim.device = 'cpu'; cfg.sim.n_envs = 2*n_pairs
    cfg.sim.ep_len = 3*phase_limit+1
    cfg.task.info_gate_enabled = True; cfg.curriculum.enabled = False
    env = GapEnv(cfg, 'cpu', difficulty=1.)
    env._reset_envs(torch.arange(env.n), tasks=scene.paired_information_tasks(n_pairs, cfg, 1., 'cpu'))
    env.task['dyn']['wind_steady'].zero_(); env.task['dyn']['gust_sigma'].zero_()
    st = env.state
    st['p'][:, 0] = env.task['wall_x']-cfg.task.info_probe_distance-.4
    st['p'][:, 1] = 0.; st['p'][:, 2] = 1.5
    for k in ('v', 'w', 'wind', 'spec_force'): st[k].zero_()
    st['q'].zero_(); st['q'][:, 0] = 1
    st['thrust'] = env.task['dyn']['mass']*dynamics.G
    # Synchronize caches after controlled-state initialization.
    env.prev_x.copy_(st['p'][:, 0]); env.prev_dist.copy_((st['p']-env._target()).norm(dim=-1))
    env.frame.copy_(render.render(st['p'], st['q'], env.task, env.rays, cfg.sensor))
    home = st['p'].clone(); probe = home.clone()
    probe[:, 0] = env.task['wall_x']-cfg.task.info_probe_distance+.10
    hover = feedback(env, home); env.prev_action.copy_(hover); env.delay_buf[:] = hover.unsqueeze(1)
    task_data = {f'task_{k}': v.numpy().copy() for k,v in env.task.items() if torch.is_tensor(v)}
    task_data['task_probe_wind'] = env.task['dyn']['probe_wind'].numpy().copy()
    trace = {k: [] for k in ('p','q','v','act','clear_pre','clear','collision','done','attempt_id','phase')}
    stages = []; reason = None; minimum = float('inf')
    for phase, target in [('approach', probe), ('stop', probe), ('retreat', home)]:
        dwell = 0; complete = False
        for _ in range(phase_limit):
            clear_pre = collision.clearance(st['p'], st['q'], env.task, env.bpts, cfg.sim.arena_y, cfg.sim.arena_z)
            if not all(torch.isfinite(v).all() for v in st.values()) or not torch.isfinite(clear_pre).all():
                reason = 'nonfinite_state'; break
            if (clear_pre <= .15).any():
                reason = 'clearance_margin'; break
            action = feedback(env, target)
            for k in ('p','q','v'): trace[k].append(st[k].numpy().copy())
            trace['act'].append(action.numpy().copy()); trace['clear_pre'].append(clear_pre.numpy().copy())
            trace['phase'].append(np.full(env.n, phase))
            _, _, done, info = env.step(action)
            for k, value in [('clear',info['clearance']),('collision',info['collision']),('done',done),('attempt_id',info['attempt_id'])]:
                trace[k].append(value.numpy().copy())
            minimum = min(minimum, float(info['clearance'].min()))
            if done.any() or info['collision'].any():
                reason = 'terminal_or_contact'; break
            if not all(torch.isfinite(v).all() for v in st.values()) or not torch.isfinite(info['clearance']).all():
                reason = 'nonfinite_state'; break
            if (info['clearance'] <= .15).any():
                reason = 'clearance_margin'; break
            dwell = dwell+1 if ready(st, target).all() else 0
            if dwell >= dwell_steps:
                complete = True; break
        stages.append({'phase':phase, 'completed':complete, 'speed_max':float(st['v'].norm(dim=-1).max()) if reason != 'terminal_or_contact' else None})
        if not complete:
            reason = reason or 'phase_timeout'; break
    accepted = reason is None and len(stages) == 3 and all(s['completed'] for s in stages)
    summary = {'scope':'privileged_feedback_baseline_NOT_learned_NOT_safety_guarantee',
               'acceptance': {'position_error_m':.08, 'speed_mps':.08, 'body_rate_radps':.12,
                              'dwell_steps':dwell_steps, 'clearance_margin_m':.15},
               'seed':seed, 'n_envs':env.n, 'protocol_completed':accepted, 'failure_reason':reason,
               'stages':stages, 'steps':len(trace['p']), 'min_clearance':minimum if np.isfinite(minimum) else None,
               'state_feedback':'ground truth p/v/q/w; true mass and thrust calibration',
               'safety':'post-step rejection and pre-step margin; NOT certified action shield'}
    if dest:
        meta = {'scope':summary['scope'], 'dt_ctrl':cfg.sim.dt_ctrl,'seed':seed,'body_r':cfg.sim.body_r,
                'info_gate_enabled':True,'info_probe_distance':cfg.task.info_probe_distance,
                'info_probe_ramp':cfg.task.info_probe_ramp,'state_timing':'pre_action; clear is subsequent interval minimum'}
        if trace['p']:
            np.savez_compressed(dest/'trajectory.npz', **{f'rec_{k}':np.stack(v) for k,v in trace.items()},
                                **task_data, steps=np.full(env.n,len(trace['p'])),meta=json.dumps(meta))
        if accepted:
            torch.save({'env':env.snapshot(),'config':cfg,'acceptance':summary}, dest/'recovery.pt')
        (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True); p.add_argument('--seed',type=int,default=0); p.add_argument('--pairs',type=int,default=2)
    a=p.parse_args(); print(json.dumps(run(n_pairs=a.pairs,seed=a.seed,out_dir=a.out),indent=2))


if __name__ == '__main__': main()
