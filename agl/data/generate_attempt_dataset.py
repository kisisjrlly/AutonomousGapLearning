"""Generate sensor-evidence V1 trials. No training or real-flight commands.

Physical settings are shared across independent, fixture-reset trials; these
are NOT a flown sequence of retries. Default V0 geometry is fixed. Varying
geometry is blocked until measured geometry/uncertainty are model inputs.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from ..config import load_config
from ..sim import scene
from ..sim.env import GapEnv
from ..attempt.spec import AttemptRanges, AttemptSpec, SPEC_FIELDS, SPEC_VERSION, sample_specs
from ..attempt.outcome import TARGET_FIELDS
from ..attempt.executor import execute_attempt_batch, ControllerConfig, CONTROLLER_VERSION
from .attempt_dataset import save_attempt_dataset, SCHEMA


def _tree_plain(x):
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    if isinstance(x,dict): return {k:_tree_plain(v) for k,v in x.items()}
    return x


def _join(trees):
    first=trees[0]
    if isinstance(first,dict): return {k:_join([t[k] for t in trees]) for k in first}
    if torch.is_tensor(first): return torch.cat(trees,0)
    if any(t!=first for t in trees): raise ValueError('task scalar mismatch')
    return first


def _repeat_tree(x,count):
    if isinstance(x,dict): return {k:_repeat_tree(v,count) for k,v in x.items()}
    if torch.is_tensor(x): return x.repeat_interleave(count,0)
    return x


def _set_fixed_geometry(task,cfg):
    for k,v in dict(gap_w=cfg.task.narrow_gap_width,gap_h=cfg.task.narrow_gap_height,
                    gap_roll=0.,wall_x=3.,thick=.15,gap_cy=0.,gap_cz=1.5).items():
        task[k].fill_(v)
    task['feasible'],task['geo_margin']=scene.feasibility(task['gap_w'],task['gap_h'],cfg.sim.body_r,
        cfg.sim.body_hh,cfg.task.feas_margin,cfg.task.feas_roll_max_deg,task['gap_w'].device)


def _set_v0_fixed_geometry(task, cfg):
    """Compatibility alias for archived V0 callers; uses current 0.30 m gap."""
    return _set_fixed_geometry(task, cfg)


def _task(cfg,seed,task_number,device,distribution,c):
    gen=torch.Generator(device=device).manual_seed(int(seed)+1009+104729*task_number)
    t=scene.sample_tasks(1,cfg,1.,device,gen)
    _set_fixed_geometry(t,cfg)
    d=t['dyn']
    # Motor rating does NOT increase automatically with payload. Otherwise
    # normalized thrust plus TWR-constant randomization cancels mass from motion.
    d['tmax'].fill_(c.nominal_tmax)
    d['wind_steady'].zero_()
    d['probe_wind'].zero_();d['gust_sigma'].zero_()
    family=('nominal','payload','response')[task_number%3] if distribution=='separate' else 'combined'
    if family in ('nominal','response'): d['mass'].fill_(c.nominal_mass)
    else: d['mass'][:]=c.nominal_mass*(.90+.20*torch.rand(1,device=device,generator=gen))
    if family not in ('response','combined'):
        d['delay'].zero_();d['tau_rate'].fill_(.05);d['tau_thrust'].fill_(.045)
    return t,family


def _provenance():
    root=Path(__file__).resolve().parents[2]
    hashes={}
    for pattern in ('agl/attempt/*.py','agl/data/*.py','agl/sim/*.py','agl/config.py'):
        for path in sorted(root.glob(pattern)):
            hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True))
    except (OSError,subprocess.CalledProcessError): commit=None;dirty=None
    return {'git_commit':commit,'git_dirty':dirty,'source_sha256':hashes,
            'torch_version':torch.__version__,'numpy_version':np.__version__}


def save_viewer_trace(path,trace):
    """Existing Rerun/GIF format, with authoritative per-row episode lengths."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    names=np.array(['APPROACH','RETREAT','FINISHED'])
    arrays={f'rec_{k}':v for k,v in trace.items() if isinstance(v,np.ndarray) and k!='steps'}
    if arrays['rec_phase'].size: arrays['rec_phase']=names[arrays['rec_phase']]
    arrays.update({f'task_{k}':v for k,v in trace['task'].items()})
    arrays['steps']=trace['steps'];arrays['meta']=json.dumps(trace['meta'],allow_nan=False)
    # Exclusive write prevents accidental replacement of previous experiment files.
    with path.open('xb') as f: np.savez_compressed(f,**arrays)


@torch.no_grad()
def generate(out,*,num_tasks=16,attempts_per_task=8,batch_tasks=4,seed=0,device='cpu',
             difficulty=1.,disable_gust=True,vary_geometry=False,ranges=None,executor_kwargs=None,
             probes_per_task=2,distribution='separate',trace_dir=None,
             fixed_specs=None,rollout_seed=None):
    for v in (num_tasks,attempts_per_task,batch_tasks):
        if isinstance(v,bool) or not isinstance(v,int) or v<1: raise ValueError('positive integer counts required')
    if not disable_gust: raise ValueError('wind is disabled; gusts cannot be enabled')
    if vary_geometry: raise ValueError('geometry variation requires measured geometry inputs; V1 is fixed-geometry diagnostic')
    if difficulty!=1.: raise ValueError('V1 diagnostic distribution is explicitly difficulty=1')
    if not 0<=probes_per_task<attempts_per_task: raise ValueError('need at least one candidate per task')
    if distribution not in ('separate','combined'): raise ValueError('unknown distribution')
    # Common-bank pilots vary physical conditions, not which candidates exist.
    if fixed_specs is not None:
        fixed_specs=tuple(fixed_specs)
        if len(fixed_specs)!=attempts_per_task or not all(isinstance(x,AttemptSpec) for x in fixed_specs):
            raise ValueError('fixed_specs must contain one valid AttemptSpec per trial')
    if rollout_seed is not None and (isinstance(rollout_seed,bool) or not isinstance(rollout_seed,int) or rollout_seed<0):
        raise ValueError('rollout_seed must be a nonnegative integer')
    out=Path(out)
    if out.suffix!='.npz': raise ValueError('output must end in .npz')
    if out.exists(): raise FileExistsError(out)
    options=dict(executor_kwargs or {})
    if set(options)&{'record','return_batch','probe_only','controller'}: raise ValueError('reserved executor options')
    c=ControllerConfig(); ranges=ranges or AttemptRanges()
    cfg=load_config();cfg.sim.device=device;cfg.curriculum.enabled=False
    cfg.sim.ep_len=options.get('max_approach_steps',720)+options.get('max_retreat_steps',600)+2
    rows={k:[] for k in ('task_id','group_id','attempt_index','spec','initial_observation','evidence',
                        'evidence_mask','context_eligible','target','target_mask','status','steps','role')}
    audit_tasks=[];families=[];audit_ids=[]
    for start in range(0,num_tasks,batch_tasks):
        count=min(batch_tasks,num_tasks-start)
        cfg.sim.n_envs=count*attempts_per_task
        tasks=[];group_ids=[];specs=[]
        for tid in range(start,start+count):
            task,family=_task(cfg,seed,tid,device,distribution,c)
            payload=json.dumps(_tree_plain(task),sort_keys=True,separators=(',',':'),allow_nan=False)
            group_ids.append(hashlib.sha256(payload.encode()).hexdigest())
            audit_tasks.append(payload);audit_ids.append(tid);families.append(family)
            tasks.append(task)
            rng=np.random.default_rng(np.random.SeedSequence([seed,tid,29]))
            specs+=list(fixed_specs) if fixed_specs is not None else sample_specs(attempts_per_task,rng,ranges)
        base=_join(tasks);bank=_repeat_tree(base,attempts_per_task)
        # Same settings reproduce exactly; batch-layout invariance of the global
        # simulator RNG is NOT claimed. The layout is recorded in the manifest.
        torch.manual_seed((seed if rollout_seed is None else rollout_seed)+5003+start)
        env=GapEnv(cfg,device,difficulty=1.)
        env._reset_envs(torch.arange(env.n,device=env.dev),tasks=bank)
        for j in range(0,env.n,attempts_per_task):
            env.v_bias[j:j+attempts_per_task]=env.v_bias[j].clone()
            env.z_bias[j:j+attempts_per_task]=env.z_bias[j].clone()
        is_probe=np.tile(np.arange(attempts_per_task)<probes_per_task,count)
        result=execute_attempt_batch(env,specs,probe_only=is_probe,return_batch=True,
                                     record=trace_dir is not None and start==0,controller=c,**options)
        outcomes=result.outcomes
        data={'task_id':np.repeat(np.arange(start,start+count),attempts_per_task),
              'group_id':np.repeat(np.array(group_ids),attempts_per_task),
              'attempt_index':np.tile(np.arange(attempts_per_task),count),
              'spec':np.stack([s.as_vector() for s in specs]),
              'initial_observation':result.initial_observation,'evidence':result.evidence,
              'evidence_mask':result.evidence_mask,
              'context_eligible':np.array([o.context_eligible for o in outcomes]) & result.evidence_mask.any(1),
              'target':np.stack([o.as_target_vector() for o in outcomes]),
              'target_mask':np.stack([o.as_target_mask() for o in outcomes]),
              'status':np.array([o.status for o in outcomes]),'steps':np.array([o.steps for o in outcomes]),
              'role':np.where(is_probe,'probe','candidate')}
        for k,v in data.items(): rows[k].append(v)
        if result.trace is not None: save_viewer_trace(Path(trace_dir)/'batch_0000.npz',result.trace)
    arrays={k:np.concatenate(v) for k,v in rows.items()}
    meta={'schema':SCHEMA,'spec_version':SPEC_VERSION,'spec_fields':list(SPEC_FIELDS),'target_fields':list(TARGET_FIELDS),
          'rollout_seed':seed if rollout_seed is None else rollout_seed,
          'fixed_specs':None if fixed_specs is None else [s.as_vector().tolist() for s in fixed_specs],
          'seed':seed,'num_tasks':num_tasks,'attempts_per_task':attempts_per_task,'batch_tasks':batch_tasks,
          'probes_per_task':probes_per_task,'geometry_varied':False,'distribution':distribution,
          'wind_model':'disabled','gust_disabled':True,
          'task_mode':'no_wind_narrow_gap_v1','controller_version':CONTROLLER_VERSION,'controller':asdict(c),
          'ranges':asdict(ranges),'executor_options':options,'sim_config':cfg.to_dict(),
          'sampling_mode':'independent_fixture_trials_NOT_continuous_retries',
          'evidence_rule':'o_pre[:17],requested_command,o_post[:17],dt,time; no query labels or terminal reset obs',
          'labels_rule':'observed controller-conditional trial outcomes with applicability masks; NOT unexecuted feasibility',
          'privileges':'executor uses true pose/fixture geometry; fixed nominal dynamics; predictor receives simulated sensor evidence',
          'split_rule':'group_id physical fingerprints; task_id is local metadata only',**_provenance()}
    save_attempt_dataset(out,meta=meta,audit={'audit_task_json':np.array(audit_tasks),
                         'audit_task_id':np.array(audit_ids),'audit_family':np.array(families)},**arrays)
    unique,counts=np.unique(arrays['status'],return_counts=True)
    return {'out':str(out),'num_records':len(arrays['task_id']),'num_tasks':num_tasks,
            'status_counts':dict(zip(unique.tolist(),counts.tolist())),
            'context_eligible_records':int(arrays['context_eligible'].sum()),
            'candidate_records':int((arrays['role']=='candidate').sum()),
            'valid_target_fraction':arrays['target_mask'].mean(0).tolist(),
            'scope':'data_pipeline_validation_NOT_prediction_gain_NOT_real_flight'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--tasks',type=int,default=16)
    p.add_argument('--attempts-per-task',type=int,default=8);p.add_argument('--batch-tasks',type=int,default=4)
    p.add_argument('--probes-per-task',type=int,default=2);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cpu')
    p.add_argument('--distribution',choices=['separate','combined'],default='separate')
    p.add_argument('--trace-dir');p.add_argument('--vary-geometry',action='store_true')
    a=p.parse_args()
    print(json.dumps(generate(a.out,num_tasks=a.tasks,attempts_per_task=a.attempts_per_task,
       batch_tasks=a.batch_tasks,probes_per_task=a.probes_per_task,seed=a.seed,device=a.device,
       distribution=a.distribution,trace_dir=a.trace_dir,
       vary_geometry=a.vary_geometry),indent=2,allow_nan=False))

if __name__=='__main__': main()
