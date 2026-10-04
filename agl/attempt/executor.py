"""Auditable independent simulation trials; NOT autonomous retry or a safety shield.

The controller uses ground-truth pose and known fixture geometry (declared V0
privilege), but fixed nominal thrust calibration, never hidden mass/tmax/wind.
Only simulated onboard observations and requested commands enter evidence.
Terminal labels use pre-reset transition state. Every finished row is masked.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import torch
from .outcome import AttemptOutcome
from .spec import AttemptSpec
from .reference import path_reference, speed_reference
from ..sim import dynamics, collision, render
from ..sim.maths import body_z_world, quat_rotate_inv

CONTROLLER_VERSION = "nominal_feedback_hermite_v1"
OBS_DIM = 17  # exclude simulator-only thrust/tmax channel at index 17
EVIDENCE_DIM = 2*OBS_DIM + 4 + 2  # o_pre, requested action, o_post, dt, elapsed time


@dataclass
class AttemptBatch:
    outcomes: list[AttemptOutcome]
    initial_observation: np.ndarray
    evidence: np.ndarray
    evidence_mask: np.ndarray
    trace: dict | None


@dataclass(frozen=True)
class ControllerConfig:
    nominal_mass: float = .775
    nominal_tmax: float = .775 * 9.81 * 2.8
    accel_limit: float = 2.0
    switch_distance: float = .8
    probe_distance: float = .65

    def __post_init__(self):
        if any(not np.isfinite(x) or x <= 0 for x in asdict(self).values()):
            raise ValueError("controller constants must be positive and finite")


def _action_from_accel(env, accel, heading, c):
    # Pose feedback is privileged in V0; dynamics calibration is not task-specific.
    st = env.state
    force = accel.clamp(-c.accel_limit, c.accel_limit).clone()
    force[:, 2] += dynamics.G
    desired_z = force / force.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    axis = torch.cross(body_z_world(st['q']), desired_z, dim=-1)
    rates = 5.0 * quat_rotate_inv(st['q'], axis) - .3*st['w']
    w, x, y, z = st['q'].unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    error = torch.atan2(torch.sin(heading-yaw), torch.cos(heading-yaw))
    rates[:, 2] += 2.0*error
    action = torch.zeros(env.n, 4, device=env.dev)
    action[:, 0] = 2*force.norm(dim=-1)*c.nominal_mass/c.nominal_tmax - 1
    action[:, 1:] = rates / dynamics.OMEGA_MAX.to(env.dev)
    return action.clamp(-1, 1)


def _ready(st, target):
    return ((st['p']-target).norm(dim=-1)<.08) & (st['v'].norm(dim=-1)<.08) & (st['w'].norm(dim=-1)<.12)


def _finite_rows(st):
    out = torch.ones(st['p'].shape[0], dtype=torch.bool, device=st['p'].device)
    for v in st.values():
        out &= torch.isfinite(v).reshape(len(out), -1).all(-1)
    return out


def _initialize(env, specs, c):
    st, cfg = env.state, env.cfg
    home = torch.stack([torch.full_like(env.task['wall_x'], cfg.sim.retry_x-.25),
                        env.task['gap_cy']+specs[:, 0], env.task['gap_cz']+specs[:, 1]], -1)
    st['p'].copy_(home)
    for key in ('v','w','wind','spec_force'):
        st[key].zero_()
    st['q'].zero_(); st['q'][:, 0]=1
    # Explicit simulation fixture, not inferred task-specific hover calibration.
    st['thrust'].fill_(c.nominal_mass*dynamics.G)
    for key in ('t_step','attempts','in_attempt','attempt_depth','retry_dwell'):
        getattr(env,key).zero_()
    env.ep_min_clear.fill_(10)
    env.prev_x.copy_(home[:, 0]); env.prev_dist.copy_((home-env._target()).norm(dim=-1))
    a = _action_from_accel(env, torch.zeros_like(home), torch.zeros(env.n,device=env.dev), c)
    env.prev_action.copy_(a); env.delay_buf.copy_(a[:,None].expand_as(env.delay_buf)); env.buf_ptr=0
    fresh=render.render(st['p'],st['q'],env.task,env.rays,cfg.sensor)
    env.frame.copy_(fresh); env.frame_delay_buf.copy_(fresh[:,None].expand_as(env.frame_delay_buf))
    env.frame_buf_ptr=0
    return home


@torch.no_grad()
def execute_attempt_batch(env, specs, *, abort_clearance=.04, settle_steps=12,
                          max_approach_steps=720, max_retreat_steps=600,
                          evidence_steps=32, probe_only=None, record=False,
                          return_batch=False, controller=None):
    """Run independent fixture-initialized trials and preserve all outcomes.

    Two budgets are independent: a row aborted early still gets exactly the
    requested recovery budget, not the unused approach budget. Stop distance is
    forward overshoot up to the first non-positive vx, valid only if observed
    before contact. Clearance is a sampled simulator diagnostic, not a bound.
    """
    if len(specs)!=env.n or not all(isinstance(s,AttemptSpec) for s in specs):
        raise ValueError("one valid AttemptSpec is required per row")
    for value in (settle_steps,max_approach_steps,max_retreat_steps,evidence_steps):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError("step budgets must be positive integers")
    if not np.isfinite(abort_clearance) or abort_clearance<=0:
        raise ValueError("abort_clearance must be positive finite")
    c=controller or ControllerConfig()
    cfg, st, dev, n=env.cfg,env.state,env.dev,env.n
    xi=torch.tensor(np.stack([s.as_vector() for s in specs]),device=dev)
    home=_initialize(env,xi,c)
    refs={k:env.task[k].clone() for k in ('wall_x','gap_cy','gap_cz','thick','gap_w','gap_h','gap_roll')}
    probe_wind_ref=env.task['dyn'].get('probe_wind',torch.zeros(n,3,device=dev)).clone()
    probe=torch.zeros(n,dtype=torch.bool,device=dev) if probe_only is None else torch.as_tensor(probe_only,device=dev,dtype=torch.bool)
    if probe.shape!=(n,): raise ValueError("probe_only shape mismatch")
    obs=env.observe(); initial=obs['vec'][:,:OBS_DIM].cpu().numpy().copy()
    phase=torch.zeros(n,dtype=torch.long,device=dev)  # approach=0, recovery=1, finished=2
    phase_count=torch.zeros_like(phase); steps=torch.zeros_like(phase); dwell=torch.zeros_like(phase)
    aborted=torch.zeros(n,dtype=torch.bool,device=dev)
    status=['']*n; reasons=[None]*n
    minimum=torch.full((n,),float('inf'),device=dev)
    tiltmax=torch.zeros(n,device=dev); peakx=st['p'][:,0].clone()
    peaky=st['p'][:,1].clone(); peakz=st['p'][:,2].clone()
    abortx=torch.full((n,),float('nan'),device=dev)
    overshoot=torch.zeros(n,device=dev); stopped=torch.zeros(n,dtype=torch.bool,device=dev)
    terminal_speed=torch.full((n,),float('nan'),device=dev)
    integral=torch.zeros_like(home); vcmd=torch.minimum(xi[:,3]*.65,torch.full((n,),.08,device=dev))
    evidence_rows=[[] for _ in range(n)]
    trace={k:[] for k in ('p','v','q','p_post','v_post','act','applied_act','clear_pre','clear','collision',
                          'success','done','attempt_id','phase','obs_vec','valid','frames')} if record else None

    def finish(mask, why, speed=None):
        for i in mask.nonzero().flatten().tolist(): status[i]=why
        if speed is not None: terminal_speed[mask]=speed[mask]
        phase[mask]=2

    for tick in range(max_approach_steps+max_retreat_steps):
        active=phase<2
        if not active.any(): break
        clear=collision.clearance(st['p'],st['q'],env.task,env.bpts,cfg.sim.arena_y,cfg.sim.arena_z)
        bad=active & (~_finite_rows(st) | ~torch.isfinite(clear) | ~torch.isfinite(obs['vec'][:,:OBS_DIM]).all(-1))
        finish(bad,'invalid_state')
        if tick==0:
            finish((phase<2)&(clear<=0),'invalid_start')
        active=phase<2
        if not active.any(): break
        # Do not send NaNs from a rejected row through the shared simulator.
        if bad.any(): env._reset_envs(bad.nonzero().flatten())
        minimum[active]=torch.minimum(minimum[active],clear[active])
        approach=phase==0
        forced=probe & (st['p'][:,0]>=refs['wall_x']-c.probe_distance)
        switch=approach & ((clear<=abort_clearance)|forced|(phase_count>=max_approach_steps))
        for i in switch.nonzero().flatten().tolist():
            reasons[i]='probe_end' if bool(forced[i]) else ('clearance' if float(clear[i])<=abort_clearance else 'approach_budget')
        aborted[switch]=True; abortx[switch]=st['p'][switch,0]
        phase[switch]=1;phase_count[switch]=0;dwell[switch]=0;integral[switch]=0
        # Already non-positive vx means zero measured forward stopping distance.
        stopped[switch & (st['v'][:,0]<=0)]=True
        finish((phase==1)&(phase_count>=max_retreat_steps),'recovery_timeout',st['v'].norm(dim=-1))
        active=phase<2
        if not active.any(): break
        vcmd=speed_reference(vcmd,st['p'][:,0],refs['wall_x'],xi[:,3],xi[:,4],xi[:,5],cfg.sim.dt_ctrl,c.switch_distance)
        yz,dydx=path_reference(st['p'][:,0],home,refs['wall_x'],refs['gap_cy'],refs['gap_cz'],xi[:,2])
        a=torch.zeros_like(home)
        a[:,0]=2.6*(vcmd-st['v'][:,0])
        a[:,1:]=2.2*(yz-st['p'][:,1:])+2.6*(vcmd[:,None]*dydx-st['v'][:,1:])
        integral[:,2]=(integral[:,2]+cfg.sim.dt_ctrl*(yz[:,1]-st['p'][:,2])).clamp(-1,1)
        a[:,2]+=.8*integral[:,2]
        recovery=phase==1
        integral[recovery]=(integral[recovery]+cfg.sim.dt_ctrl*(home[recovery]-st['p'][recovery])).clamp(-1,1)
        # Do not integrate long-distance x errors into a saturated retreat.
        integral[:,0]=0
        back=(2.2*(home-st['p'])-2.8*st['v']+.8*integral)
        a[recovery]=back[recovery]
        heading=torch.atan(dydx[:,0]);heading[recovery]=0
        action=_action_from_accel(env,a,heading,c)
        action[~active]=0
        before={k:st[k].clone() for k in ('p','v','q')}
        preobs=obs['vec'][:,:OBS_DIM].clone()
        image=obs['img'] if record else None
        command_phase=phase.clone()
        steps[active]+=1;phase_count[active]+=1
        obs,_,done,info=env.step(action,capture_transition=True)
        post=info['transition']['state']
        # The terminal state above is from BEFORE reset, never env.state after done.
        finite=_finite_rows(post)&torch.isfinite(info['clearance'])
        bad=active & ~finite
        finish(bad,'invalid_state')
        valid=active & finite
        minimum[valid]=torch.minimum(minimum[valid],info['clearance'][valid])
        tilt=torch.acos(body_z_world(post['q'])[:,2].clamp(-1,1))
        tiltmax[valid]=torch.maximum(tiltmax[valid],tilt[valid])
        advancing=valid & (post['p'][:,0]>peakx)
        peakx[advancing]=post['p'][advancing,0]
        peaky[advancing]=post['p'][advancing,1];peakz[advancing]=post['p'][advancing,2]
        braking=valid & aborted & ~stopped
        overshoot[braking]=torch.maximum(overshoot[braking],(post['p'][braking,0]-abortx[braking]).clamp_min(0))
        stopped |= braking & (post['v'][:,0]<=0) & ~info['collision']
        speed=post['v'].norm(dim=-1)
        hit=valid&info['collision'];passed=valid&info['success']&~hit
        finish(hit,'contact',speed);finish(passed,'success',speed)
        finish(valid&done&~hit&~passed,'environment_terminal',speed)
        # A returned observation at done belongs to the next task: never store it.
        evidence_valid=valid&~done&torch.isfinite(obs['vec'][:,:OBS_DIM]).all(-1)
        for i in evidence_valid.nonzero().flatten().tolist():
            token=torch.cat([preobs[i],action[i],obs['vec'][i,:OBS_DIM],
                             torch.tensor([cfg.sim.dt_ctrl,tick*cfg.sim.dt_ctrl],device=dev)])
            evidence_rows[i].append(token.cpu().numpy())
        retry_ready=(phase==1)&~done&_ready(post,home)&(post['p'][:,0]<cfg.sim.retry_x)&~env.in_attempt
        dwell[phase==1]=torch.where(retry_ready[phase==1],dwell[phase==1]+1,0)
        finish((phase==1)&(dwell>=settle_steps),'recovered',speed)
        if record:
            data={**before,'p_post':post['p'],'v_post':post['v'],'act':action,
                  'applied_act':info['transition']['applied_action'],'clear_pre':clear,
                  'clear':info['clearance'],'collision':info['collision'],'success':info['success'],
                  'done':done,'attempt_id':info['attempt_id'],'phase':command_phase,
                  'obs_vec':preobs,'valid':valid,'frames':(image.clamp(0,1)*255).to(torch.uint8)}
            for k,v in data.items(): trace[k].append(v.cpu().numpy().copy())
    unfinished=phase<2
    finish(unfinished,'recovery_timeout',st['v'].norm(dim=-1))
    evidence=np.zeros((n,evidence_steps,EVIDENCE_DIM),np.float32)
    evidence_mask=np.zeros((n,evidence_steps),bool)
    for i,rows in enumerate(evidence_rows):
        if rows:
            idx=np.linspace(0,len(rows)-1,min(evidence_steps,len(rows)),dtype=int)
            evidence[i,:len(idx)]=np.asarray(rows)[idx];evidence_mask[i,:len(idx)]=True
    def scalar(x,i):
        v=float(x[i]);return v if np.isfinite(v) else None
    outcomes=[]
    for i in range(n):
        invalid=status[i] in ('invalid_start','invalid_state')
        outcomes.append(AttemptOutcome(status=status[i],success=status[i]=='success',recovered=status[i]=='recovered',
          contact=status[i]=='contact',aborted=bool(aborted[i]),
          lateral_drift=None if invalid else float(peaky[i]-refs['gap_cy'][i]),
          vertical_drift=None if invalid else float(peakz[i]-refs['gap_cz'][i]),
          min_clearance=None if invalid else scalar(minimum,i),
          stopping_distance=float(overshoot[i]) if bool(aborted[i]&stopped[i]) and not invalid else None,
          max_tilt=None if invalid else scalar(tiltmax,i),terminal_speed=None if invalid else scalar(terminal_speed,i),
          abort_x=scalar(abortx,i),steps=int(steps[i]),abort_reason=reasons[i]))
    if trace is not None:
        trace={k:np.stack(v) if v else np.empty((0,n)) for k,v in trace.items()}
        trace['steps']=steps.cpu().numpy();trace['task']={k:v.cpu().numpy() for k,v in refs.items()}
        trace['task']['probe_wind']=probe_wind_ref.cpu().numpy()
        trace['dt_ctrl']=cfg.sim.dt_ctrl
        trace['meta']={'dt_ctrl':cfg.sim.dt_ctrl,'body_r':cfg.sim.body_r,'body_hh':cfg.sim.body_hh,
                       'retry_x':cfg.sim.retry_x,'succ_margin':cfg.sim.succ_margin,
                       'info_gate_enabled':cfg.task.info_gate_enabled,
                       'info_probe_distance':cfg.task.info_probe_distance,'info_probe_ramp':cfg.task.info_probe_ramp,
                       'scope':'independent_simulation_trials_NOT_learned_NOT_safety_guarantee',
                       'timing':'pre-action state; p_post/v_post before reset; commands are not motor forces'}
    result=AttemptBatch(outcomes,initial,evidence,evidence_mask,trace)
    return result if return_batch else outcomes
