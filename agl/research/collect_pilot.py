"""Collect common-bank repetitions with existing GapEnv, preserving all failures.

This does not train a network, send real-flight commands, or choose safe margins.
The configuration's diagnostic thresholds are not deployment recommendations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..attempt.spec import AttemptSpec
from .pilot import file_sha256, write_json_exclusive


def collect(out_dir, *, config, tasks=12, repetitions=2, batch_tasks=3,
            seed=0, device='cpu', trace=False, generate_fn=None):
    for name, value in [('tasks',tasks),('repetitions',repetitions),('batch_tasks',batch_tasks),('seed',seed)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name=='seed' else 1):
            raise ValueError(f'invalid {name}')
    if tasks < 6 or repetitions < 2:
        raise ValueError('pilot requires >=6 physical tasks and >=2 repetitions')
    config = Path(config)
    bank = json.loads(config.read_text(encoding='utf-8'))
    if bank.get('version') != 'common_bank_v1':
        raise ValueError('unsupported candidate-bank version')
    probes = [AttemptSpec(**s) for s in bank['probes']]
    candidates = [AttemptSpec(**s) for s in bank['candidates']]
    if not probes or len(candidates) < 2:
        raise ValueError('at least one probe and two candidates required')
    if len({tuple(s.as_vector()) for s in candidates}) != len(candidates):
        raise ValueError('duplicate candidate specifications')
    if generate_fn is None:
        from ..data.generate_attempt_dataset import generate
        generate_fn = generate
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'version':'research_pilot_collection_v1', 'config':bank,
                'config_sha256':file_sha256(config),'tasks':tasks,'repetitions':repetitions,
                'batch_tasks':batch_tasks,'task_seed':seed,'device':device,
                'scope':'independent_fixture_simulation_NOT_autonomous_retries', 'files':[]}
    try:
        for r in range(repetitions):
            path = out_dir / f'repeat_{r:02d}.npz'
            rollout_seed = seed + 1000003 * (r+1)
            result = generate_fn(path, num_tasks=tasks, attempts_per_task=len(probes)+len(candidates),
                probes_per_task=len(probes), batch_tasks=batch_tasks, seed=seed, device=device,
                fixed_specs=probes+candidates, rollout_seed=rollout_seed,
                executor_kwargs=bank.get('executor_options',{}),
                trace_dir=out_dir/'trace' if trace and r==0 else None)
            manifest['files'].append({'file':path.name,'sha256':file_sha256(path),
                                      'rollout_seed':rollout_seed,'summary':result})
    except Exception as exc:
        manifest['status']='incomplete';manifest['error']=f'{type(exc).__name__}: {exc}'
        write_json_exclusive(out_dir/'manifest.incomplete.json',manifest)
        raise
    manifest['status']='complete'
    write_json_exclusive(out_dir/'manifest.json',manifest)
    return manifest


def main():
    default = Path(__file__).resolve().parents[2]/'configs'/'research_pilot.json'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--config',default=str(default))
    p.add_argument('--tasks',type=int,default=12);p.add_argument('--repetitions',type=int,default=2)
    p.add_argument('--batch-tasks',type=int,default=3);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cpu');p.add_argument('--trace',action='store_true')
    a=p.parse_args()
    result=collect(a.out,config=a.config,tasks=a.tasks,repetitions=a.repetitions,
                   batch_tasks=a.batch_tasks,seed=a.seed,device=a.device,trace=a.trace)
    print(json.dumps({'status':result['status'],'files':result['files']},indent=2))


if __name__=='__main__': main()
