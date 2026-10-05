import copy
import json
import numpy as np
import pytest
from agl.data.attempt_dataset import AttemptDataset, save_attempt_dataset, split_task_ids, SCHEMA
from agl.data.generate_attempt_dataset import generate
from agl.attempt.spec import SPEC_FIELDS,SPEC_VERSION
from agl.attempt.outcome import TARGET_FIELDS


def sample():
    n=12
    spec=np.tile([0,0,0,.5,.5,.2],(n,1)).astype('float32')
    target=np.zeros((n,10),np.float32);target[:,7]=1;target[:,9]=1
    a=dict(task_id=np.repeat(np.arange(6),2),group_id=np.repeat(np.array([f'physical-{i}' for i in range(6)]),2),
       attempt_index=np.tile(np.arange(2),6),spec=spec,initial_observation=np.zeros((n,17),np.float32),
       evidence=np.arange(n*4*40,dtype=np.float32).reshape(n,4,40),evidence_mask=np.ones((n,4),bool),
       context_eligible=np.ones(n,bool),target=target,target_mask=np.ones((n,10),bool),
       status=np.full(n,'recovered'),steps=np.full(n,20),role=np.tile(['probe','candidate'],6))
    m=dict(schema=SCHEMA,spec_version=SPEC_VERSION,spec_fields=list(SPEC_FIELDS),target_fields=list(TARGET_FIELDS))
    return a,m


def load(tmp_path):
    a,m=sample();p=tmp_path/'data.npz';save_attempt_dataset(p,meta=m,audit={'audit_secret':np.array([123])},**a)
    return AttemptDataset(p)


def test_query_never_contains_true_targets_ids_or_audit(tmp_path):
    ds=load(tmp_path);q=ds.make_query(1,[0])
    assert set(q)=={'inputs','targets','target_mask'}
    assert set(q['inputs'])=={'candidate','initial_observation','protocol','context_spec','context_evidence','context_mask','context_protocol'}
    before=copy.deepcopy(q['inputs']);ds.target[0,:]=98765
    after=ds.make_query(1,[0])['inputs']
    for k in before:np.testing.assert_array_equal(before[k],after[k])
    assert ds.audit['audit_secret'][0]==123


@pytest.mark.parametrize('history',[[1],[2],[0,0],[-1]])
def test_query_rejects_self_future_other_task_or_duplicates(tmp_path,history):
    with pytest.raises(ValueError): load(tmp_path).make_query(1,history)


def test_unrecovered_context_rejected(tmp_path):
    ds=load(tmp_path);ds.context_eligible[0]=False
    with pytest.raises(ValueError,match='unrecovered'):ds.make_query(1,[0])


def test_grouped_split_and_empty_context(tmp_path):
    ds=load(tmp_path);s=ds.split(seed=1)
    sets=[set(ds.group_id[i]) for i in s.values()]
    assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i))
    assert ds.make_query(1,[])['inputs']['context_evidence'].shape==(0,4,40)
    with pytest.raises(ValueError):split_task_ids(ds.group_id,val_frac=0)


@pytest.mark.parametrize('kind',['nan','mask','duplicate','fractional_id','bad_order'])
def test_writer_rejects_invalid_before_creating_output(tmp_path,kind):
    a,m=sample()
    if kind=='nan':a['target'][0,0]=np.nan
    if kind=='mask':a['target_mask'][0,7]=False
    if kind=='duplicate':a['attempt_index'][1]=0
    if kind=='fractional_id':a['task_id']=a['task_id'].astype(float)+.1
    if kind=='bad_order':m['target_fields']=list(reversed(TARGET_FIELDS))
    p=tmp_path/'invalid.npz'
    with pytest.raises(ValueError):save_attempt_dataset(p,meta=m,**a)
    assert not p.exists()


def test_no_clobber_and_no_suffix_surprise(tmp_path):
    a,m=sample();p=tmp_path/'x.npz';save_attempt_dataset(p,meta=m,**a);b=p.read_bytes()
    with pytest.raises(FileExistsError):save_attempt_dataset(p,meta=m,**a)
    assert p.read_bytes()==b
    with pytest.raises(ValueError):save_attempt_dataset(tmp_path/'not_npz',meta=m,**a)
    assert not list(tmp_path.glob('.*.npz.*'))


def test_v0_explicitly_rejected(tmp_path):
    p=tmp_path/'old.npz';np.savez(p,meta=json.dumps({'schema':'attempt_dataset_v0'}))
    with pytest.raises(ValueError,match='regenerate'):AttemptDataset(p)


def test_generator_deterministic_and_fixed_geometry(tmp_path):
    paths=[tmp_path/'a.npz',tmp_path/'b.npz']
    for p in paths:
        generate(p,num_tasks=3,attempts_per_task=3,batch_tasks=3,probes_per_task=1,seed=4,
                 executor_kwargs={'max_approach_steps':3,'max_retreat_steps':3,'evidence_steps':4})
    a,b=map(AttemptDataset,paths)
    for k in a.arrays:np.testing.assert_array_equal(a.arrays[k],b.arrays[k])
    assert a.meta['sampling_mode']=='independent_fixture_trials_NOT_continuous_retries'
    assert a.meta['geometry_varied'] is False
    assert 'source_sha256' in a.meta and len(a.meta['source_sha256'])>3
    assert len(a.candidate_indices)==6
    tasks=[json.loads(t) for t in a.audit['audit_task_json']]
    assert all(t['gap_w']==pytest.approx([.30]) for t in tasks)
    assert all(t['dyn']['tmax']==pytest.approx([.775*9.81*2.8]) for t in tasks)
    with pytest.raises(ValueError,match='geometry'):
        generate(tmp_path/'wrong.npz',vary_geometry=True)
