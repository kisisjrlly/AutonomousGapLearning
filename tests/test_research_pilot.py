"""Synthetic analytical fixtures test evaluation logic, NOT drone capability."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from agl.attempt.outcome import AttemptOutcome, TARGET_FIELDS
from agl.attempt.spec import AttemptSpec, SPEC_FIELDS, SPEC_VERSION
from agl.data.attempt_dataset import AttemptDataset, SCHEMA, save_attempt_dataset
from agl.research.collect_pilot import collect
from agl.research.pilot import (check_common_bank, decision_inputs, design_matrix,
    evidence_summary, evaluate, grouped_interval, MaskedRidge, observed_utility,
    write_json_exclusive)


def make_fixture(path, *, rep=0, groups=30, mode='informative', failed_probes=False):
    bank = [AttemptSpec(0.,0.,0.,.35,.5,.1), AttemptSpec(-.2,0.,0.,.45,.6,.2),
            AttemptSpec(.2,0.,0.,.45,.6,.2)]
    n=groups*3
    a={'task_id':np.repeat(np.arange(groups),3),'group_id':np.repeat([f'group_{g:03d}' for g in range(groups)],3),
       'attempt_index':np.tile(np.arange(3),groups),'spec':np.tile(np.stack([s.as_vector() for s in bank]),(groups,1)),
       'initial_observation':np.zeros((n,17),np.float32),'evidence':np.zeros((n,4,40),np.float32),
       'evidence_mask':np.ones((n,4),bool),'context_eligible':np.zeros(n,bool),
       'target':np.zeros((n,len(TARGET_FIELDS)),np.float32),'target_mask':np.zeros((n,len(TARGET_FIELDS)),bool),
       'status':np.full(n,'recovered',dtype='U30'),'steps':np.full(n,40,dtype=np.int64),
       'role':np.tile(np.array(['probe','candidate','candidate']),groups)}
    rng=np.random.default_rng(200+rep)
    for g in range(groups):
        latent=-1. if g%2==0 else 1.
        for k in range(3):
            i=g*3+k
            # Information occurs before recovery, while the last observation is
            # identical; otherwise a current-only baseline already has the answer.
            z=a['evidence'][i]
            z[:,38]=.025;z[:,39]=np.arange(4)*.025
            z[:3,18]=.1
            signal=latent if mode!='noise' else float(rng.choice([-1,1]))
            z[:3,31]=.025*signal
            good=(k==1 if mode=='flat' else (k==1 and latent<0) or (k==2 and latent>0))
            if mode=='unstable' and rep%2: good=not good
            status=('contact' if failed_probes else 'recovered') if k==0 else ('success' if good else 'contact')
            out=AttemptOutcome(status,status=='success',status=='recovered',status=='contact',k==0,
                .1*latent*(float(bank[k].start_y_offset)+1),0.,.1 if status!='contact' else -.01,
                .1 if k==0 else None,.05,.05,None,40)
            a['status'][i]=status;a['context_eligible'][i]=k==0 and status=='recovered'
            a['target'][i]=out.as_target_vector();a['target_mask'][i]=out.as_target_mask()
    meta={'schema':SCHEMA,'spec_version':SPEC_VERSION,'spec_fields':list(SPEC_FIELDS),'target_fields':list(TARGET_FIELDS),
          'fixed_specs':[s.as_vector().tolist() for s in bank],'probes_per_task':1,
          'rollout_seed':200+rep,'controller_version':'synthetic_test_fixture','controller':{},
          'executor_options':{},'distribution':'synthetic','sampling_mode':'synthetic_NOT_drone_result',
          'batch_tasks':groups,'sim_config':{'sim':{'dt_ctrl':.025}},'scope':'UNIT_TEST_ONLY'}
    save_attempt_dataset(path,meta=meta,**a)
    return a,meta


def pair(tmp_path, **kwargs):
    paths=[tmp_path/f'r{r}.npz' for r in range(2)]
    for r,p in enumerate(paths): make_fixture(p,rep=r,**kwargs)
    return paths


def test_pilot_finds_signal_in_analytical_fixture(tmp_path):
    result=evaluate(pair(tmp_path),penalties=(1.,10.),bootstrap_draws=50)
    assert result['flight_approval'] is False
    assert result['question_1_strategy_necessity']['cross_repeat_choice_gain']['mean']>0
    assert result['question_2_sensor_value']['normalized_mse_reduction']['mean']>0
    assert result['question_3_decision_value']['context_minus_no_context_utility']['mean']>0
    assert result['question_3_decision_value']['net_benefit_status'].startswith('NOT_ESTABLISHED')
    json.dumps(result,allow_nan=False)


def test_flat_fixture_does_not_invent_need_for_adaptation(tmp_path):
    r=evaluate(pair(tmp_path,mode='flat'),penalties=(1.,),bootstrap_draws=30)
    assert r['question_1_strategy_necessity']['cross_repeat_choice_gain']['mean']==0
    assert r['question_3_decision_value']['context_minus_no_context_utility']['mean']==0


def test_same_repeat_hindsight_is_not_reported_as_gain(tmp_path):
    r=evaluate(pair(tmp_path,mode='unstable'),penalties=(1.,),bootstrap_draws=30)
    assert r['question_1_strategy_necessity']['cross_repeat_choice_gain']['mean']<0


def test_failed_probes_are_counted_and_block_later_fixture_history(tmp_path):
    r=evaluate(pair(tmp_path,failed_probes=True),penalties=(1.,),bootstrap_draws=30)
    assert r['eligible_decisions']==0
    assert len(r['blocked_decisions'])==60
    assert 'unavailable_reason' in r['question_2_sensor_value']
    assert r['status_counts']['contact']>0


def test_shared_inputs_ignore_all_candidate_observations_and_labels(tmp_path):
    p=tmp_path/'a.npz';make_fixture(p)
    ds=AttemptDataset(p)
    c,obs,z=decision_inputs(ds,'group_000')
    ds.initial_observation[c]=10000
    ds.evidence[c]=-50000
    ds.target[c]=555
    c2,obs2,z2=decision_inputs(ds,'group_000')
    np.testing.assert_array_equal(c,c2)
    np.testing.assert_array_equal(obs,obs2)
    np.testing.assert_array_equal(z,z2)
    assert np.array_equal(obs,np.zeros(17))


def test_context_only_uses_past_probes(tmp_path):
    p=tmp_path/'a.npz';make_fixture(p)
    ds=AttemptDataset(p)
    ds.context_eligible[0]=False
    ds.context_eligible[1:3]=True
    assert decision_inputs(ds,'group_000') is None


def test_test_labels_cannot_change_fitted_choices(tmp_path):
    paths=pair(tmp_path)
    r1=evaluate(paths,penalties=(1.,),bootstrap_draws=10)
    testgroups=set(r1['split_groups']['test'])
    other=[]
    for i,p in enumerate(paths):
        ds=AttemptDataset(p)
        which=np.isin(ds.group_id,list(testgroups)) & (ds.role=='candidate')
        # Change only held-out labels, not the sensor inputs or bank.
        ds.arrays['target'][which,0]+=500
        q=tmp_path/f'changed{i}.npz'
        save_attempt_dataset(q,meta=ds.meta,**ds.arrays);other.append(q)
    r2=evaluate(other,penalties=(1.,),bootstrap_draws=10)
    ch=lambda r:[[m['candidate'] for m in x['methods'].values()] for x in r['question_3_decision_value']['choices']]
    assert ch(r1)==ch(r2)
    assert r1['question_2_sensor_value']['sensor_context_fit']==r2['question_2_sensor_value']['sensor_context_fit']


def test_candidate_specific_initial_readings_do_not_improve_decisions(tmp_path):
    paths=pair(tmp_path)
    r1=evaluate(paths,penalties=(1.,),bootstrap_draws=10)
    changed=[]
    for i,p in enumerate(paths):
        ds=AttemptDataset(p)
        ds.arrays['initial_observation'][ds.role=='candidate']=999
        q=tmp_path/f'peek{i}.npz';save_attempt_dataset(q,meta=ds.meta,**ds.arrays);changed.append(q)
    r2=evaluate(changed,penalties=(1.,),bootstrap_draws=10)
    assert r1['question_3_decision_value']==r2['question_3_decision_value']


def test_duplicate_repetitions_rejected(tmp_path):
    p=tmp_path/'a.npz';make_fixture(p)
    with pytest.raises(ValueError,match='duplicate'):
        evaluate([p,p])


def test_random_bank_and_protocol_mix_rejected(tmp_path):
    paths=pair(tmp_path)
    ds=[AttemptDataset(p) for p in paths]
    ds[0].meta['fixed_specs']=None
    with pytest.raises(ValueError,match='common-bank'): check_common_bank(ds)
    ds=[AttemptDataset(p) for p in paths]
    ds[1].meta['executor_options']={'abort_clearance':.2}
    with pytest.raises(ValueError,match='protocol'): check_common_bank(ds)
    ds=[AttemptDataset(p) for p in paths]
    ds[1].spec[5,0]+=.01
    with pytest.raises(ValueError,match='identical'): check_common_bank(ds)


def test_group_bootstrap_does_not_count_candidates_as_independent():
    a=grouped_interval([1,1,1,0],['a','a','a','b'],draws=50)
    assert a['physical_groups']==2
    assert a['mean']==.5


def test_complete_transition_shuffle_is_invariant_for_moment_baseline():
    rng=np.random.default_rng(1);x=rng.normal(size=(5,40));x[:,38]=.025
    np.testing.assert_allclose(evidence_summary(x,np.ones(5,bool)),evidence_summary(x[::-1],np.ones(5,bool)))
    x[:,38]=0
    with pytest.raises(ValueError,match='dt'): evidence_summary(x,np.ones(5,bool))


def test_no_context_has_current_observation_interactions():
    x=design_matrix(np.array([[.1,0,0,.4,.5,.1]]),np.ones(17))
    assert x.shape[1]==6+6+17+17+6*17


def test_ridge_normalization_is_train_only():
    x=np.arange(20).reshape(10,2).astype(float);y=x[:,:1]*2
    m=MaskedRidge.fit(x,y,np.ones_like(y,bool),np.repeat(np.arange(5),2),1)
    old=m.x_mean.copy();m.predict(x+1000)
    np.testing.assert_array_equal(m.x_mean,old)


def test_invalid_candidate_is_reported_not_imputed(tmp_path):
    paths=pair(tmp_path)
    ds=AttemptDataset(paths[0]);ds.arrays['status'][1]='invalid_state'
    ds.arrays['target'][1]=0;ds.arrays['target_mask'][1]=False
    p=tmp_path/'bad.npz';save_attempt_dataset(p,meta=ds.meta,**ds.arrays)
    r=evaluate([p,paths[1]],penalties=(1.,),bootstrap_draws=10)
    assert 'group_000' in r['question_1_strategy_necessity']['incomplete_groups']
    assert any(b['reason']=='invalid_candidate_matrix' for b in r['blocked_decisions'])


def test_atomic_report_preserves_existing_file(tmp_path):
    p=tmp_path/'report.json';write_json_exclusive(p,{'a':1})
    with pytest.raises(FileExistsError):write_json_exclusive(p,{'a':2})
    assert json.loads(p.read_text())=={'a':1}
    with pytest.raises(ValueError):write_json_exclusive(tmp_path/'nan.json',{'a':float('nan')})


def test_collector_keeps_task_seed_and_changes_only_rollout_seed(tmp_path):
    calls=[]
    def fake(path,**kwargs):
        calls.append(kwargs);Path(path).write_bytes(str(kwargs['rollout_seed']).encode())
        return {'scope':'UNIT_TEST_ONLY'}
    cfg=Path(__file__).resolve().parents[1]/'configs/research_pilot.json'
    result=collect(tmp_path/'out',config=cfg,tasks=6,repetitions=2,generate_fn=fake)
    assert result['status']=='complete'
    assert calls[0]['seed']==calls[1]['seed']
    assert calls[0]['rollout_seed']!=calls[1]['rollout_seed']
    assert calls[0]['fixed_specs']==calls[1]['fixed_specs']
    assert calls[0]['probes_per_task']==2
    with pytest.raises(FileExistsError):collect(tmp_path/'out',config=cfg,tasks=6,generate_fn=fake)


def test_collector_failure_never_publishes_complete_manifest(tmp_path):
    def fail(*a,**kw):raise RuntimeError('fixture fail')
    cfg=Path(__file__).resolve().parents[1]/'configs/research_pilot.json'
    with pytest.raises(RuntimeError):collect(tmp_path/'out',config=cfg,tasks=6,generate_fn=fail)
    assert not (tmp_path/'out/manifest.json').exists()
    assert json.loads((tmp_path/'out/manifest.incomplete.json').read_text())['status']=='incomplete'
