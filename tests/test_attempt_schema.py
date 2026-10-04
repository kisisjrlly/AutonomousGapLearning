import json
import numpy as np
import pytest
import torch
from agl.attempt import AttemptSpec, AttemptRanges, sample_specs
from agl.attempt.outcome import AttemptOutcome, TARGET_FIELDS
from agl.attempt.executor import execute_attempt_batch, ControllerConfig, _action_from_accel
from agl.attempt.reference import path_reference, speed_reference
from agl.config import load_config
from agl.sim.env import GapEnv
from agl.data.generate_attempt_dataset import _set_v0_fixed_geometry


def fixture(n=2):
    torch.manual_seed(37)
    cfg=load_config();cfg.sim.n_envs=n;cfg.sim.device='cpu';cfg.sim.ep_len=2000
    env=GapEnv(cfg,'cpu',difficulty=1.)
    _set_v0_fixed_geometry(env.task,cfg)
    d=env.task['dyn'];d['mass'].fill_(.775);d['tmax'].fill_(.775*9.81*2.8)
    d['wind_steady'].zero_();d['gust_sigma'].zero_();d['delay'].zero_()
    d['tau_rate'].fill_(.05);d['tau_thrust'].fill_(.045)
    d['kd_lin'].fill_(.1);d['kd_quad'].fill_(.01)
    return env


def spec(**kw):
    d=dict(start_y_offset=0.,start_z_offset=0.,entry_yaw=0.,entry_speed=.6,accel_early=.6,accel_late=.3)
    d.update(kw);return AttemptSpec(**d)


@pytest.mark.parametrize('field,value', [('entry_speed',0),('entry_speed',float('nan')),('entry_yaw',1.),
 ('accel_early',-1.),('start_y_offset',2.),('accel_late',True)])
def test_spec_rejects_invalid(field,value):
    with pytest.raises(ValueError): spec(**{field:value})


def test_schema_roundtrip_and_deterministic_sampler():
    s=spec();np.testing.assert_allclose(AttemptSpec.from_vector(s.as_vector()).as_vector(),s.as_vector())
    a=sample_specs(3,np.random.default_rng(2));b=sample_specs(3,np.random.default_rng(2))
    assert a==b
    with pytest.raises(ValueError): AttemptRanges(entry_speed=(0.,1.))
    with pytest.raises(ValueError): AttemptSpec.from_vector([0.,1.])


def test_unobserved_stopping_is_masked_not_zero_label():
    o=AttemptOutcome('success',True,False,False,False,0.,0.,.1,None,.1,.5,None,12)
    j=TARGET_FIELDS.index('stopping_distance');r=TARGET_FIELDS.index('recovered')
    assert o.as_target_vector()[j]==0 and not o.as_target_mask()[j] and not o.as_target_mask()[r]
    assert json.loads(json.dumps(o.to_dict(),allow_nan=False))['stopping_distance'] is None
    with pytest.raises(ValueError):
        AttemptOutcome('success',True,False,True,False,0.,0.,.1,None,.1,.5,None,12)


def test_path_angle_changes_tangent_not_only_initial_camera():
    start=torch.tensor([[.95,.2,1.6],[.95,.2,1.6]])
    wall=torch.tensor([3.,3.]);ys=torch.zeros(2);zs=torch.full((2,),1.5);ang=torch.tensor([-.2,.2])
    yz,slope=path_reference(wall,start,wall,ys,zs,ang)
    assert torch.allclose(yz[:,0],ys)
    assert torch.allclose(slope[:,0],torch.tan(ang))
    before,_=path_reference(wall-.3,start,wall,ys,zs,ang)
    assert abs(float(before[0,0]-before[1,0]))>.01


def test_positive_late_acceleration_has_headroom():
    v=torch.tensor([.39,.39]);x=torch.tensor([2.3,2.3]);wall=torch.tensor([3.,3.])
    cap=torch.tensor([.6,.6]);early=torch.tensor([.6,.6]);late=torch.tensor([.1,.4])
    out=speed_reference(v,x,wall,cap,early,late,.1)
    assert out[1]>out[0] and out[1]<cap[1]


def test_nominal_controller_does_not_read_hidden_mass_tmax():
    env=fixture();env.state['q'].zero_();env.state['q'][:,0]=1;env.state['w'].zero_()
    zero=torch.zeros(2,3);heading=torch.zeros(2)
    a=_action_from_accel(env,zero,heading,ControllerConfig())
    env.task['dyn']['mass']*=1.8;env.task['dyn']['tmax']*=.7
    b=_action_from_accel(env,zero,heading,ControllerConfig())
    assert torch.equal(a,b)


def test_transition_capture_is_before_reset_and_does_not_change_rollout():
    env=fixture();env.state['p'][:,0]=env.task['wall_x']+2
    snap=env.snapshot();a=torch.zeros(2,4)
    obs1,r1,d1,i1=env.step(a,capture_transition=True)
    assert d1.all()
    assert (i1['transition']['state']['p'][:,0]>4).all()
    assert (env.state['p'][:,0]<1).all()
    env.restore(snap);obs2,r2,d2,i2=env.step(a)
    assert torch.equal(obs1['img'],obs2['img']) and torch.equal(obs1['vec'],obs2['vec'])
    assert torch.equal(r1,r2) and torch.equal(d1,d2) and 'transition' not in i2


def test_executor_timeout_masks_unobserved_stop_and_preserves_evidence():
    r=execute_attempt_batch(fixture(),[spec(),spec(start_y_offset=.1)],
       max_approach_steps=4,max_retreat_steps=3,return_batch=True,record=True)
    assert all(o.status=='recovery_timeout' for o in r.outcomes)
    assert all(o.steps==7 for o in r.outcomes)
    assert r.evidence.shape==(2,32,40)
    assert r.initial_observation.shape==(2,17)
    assert all(not o.as_target_mask()[3] for o in r.outcomes)
    assert r.trace['p_post'].shape==r.trace['p'].shape
    assert np.isfinite(r.evidence).all()


def test_contact_cannot_be_saved_as_recovered_and_no_reset_sensor_transition(monkeypatch):
    env=fixture();original=env.step
    def contact(action,**kw):
        obs,reward,done,info=original(action,**kw)
        info['collision'][0]=True;info['success'][0]=False;done[0]=True
        info['transition']['state']['v'][0]=torch.tensor([3.,4.,0.])
        env._reset_envs(torch.tensor([0]));obs['vec'][0]=9999
        return obs,reward,done,info
    monkeypatch.setattr(env,'step',contact)
    r=execute_attempt_batch(env,[spec(),spec()],max_approach_steps=2,max_retreat_steps=2,
                             return_batch=True,record=True)
    assert r.outcomes[0].status=='contact' and r.outcomes[0].steps==1
    assert r.outcomes[0].terminal_speed==5.0
    assert not r.outcomes[0].context_eligible and not r.evidence_mask[0].any()
    assert (r.evidence[0]==0).all()
    assert r.outcomes[1].steps==4


def test_nonfinite_is_invalid_not_fabricated_safe_label(monkeypatch):
    env=fixture();original=env.step
    def corrupt(action,**kw):
        data=original(action,**kw);data[3]['transition']['state']['v'][0,0]=float('nan');return data
    monkeypatch.setattr(env,'step',corrupt)
    out=execute_attempt_batch(env,[spec(),spec()],max_approach_steps=2,max_retreat_steps=2)
    assert out[0].status=='invalid_state' and not out[0].as_target_mask().any()
    assert out[0].min_clearance is None


def test_each_early_abort_gets_its_own_recovery_budget():
    # Huge diagnostic margin forces an immediate abort. The unused approach
    # budget must NOT be added to the requested three recovery steps.
    r=execute_attempt_batch(fixture(),[spec(),spec()],abort_clearance=10.,
                            max_approach_steps=100,max_retreat_steps=3,settle_steps=10)
    assert all(o.steps==3 for o in r)


def test_payload_changes_motion_when_motor_rating_is_fixed():
    env=fixture();env.task['dyn']['mass'][1]*=1.1
    r=execute_attempt_batch(env,[spec(),spec()],max_approach_steps=6,max_retreat_steps=2,
                             return_batch=True,record=True)
    assert abs(float(r.trace['p_post'][4,0,2]-r.trace['p_post'][4,1,2]))>1e-4
