"""V1 separates sensor evidence, query inputs, supervision, and audit metadata.

V0 files are deliberately rejected: their labels cannot reconstruct missing
sensor histories or censored outcomes. Regenerate rather than rename fields.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from ..attempt.spec import SPEC_FIELDS, SPEC_VERSION
from ..attempt.outcome import TARGET_FIELDS, STATUSES

SCHEMA = 'attempt_dataset_v1_evidence'
OBS_DIM, EVIDENCE_DIM = 17, 40
REQUIRED = ('task_id','group_id','attempt_index','spec','initial_observation','evidence',
            'evidence_mask','context_eligible','target','target_mask','status','steps','role')


def split_task_ids(task_ids, *, train_frac=.8, val_frac=.1, seed=0):
    """Group split: pass physical group fingerprints, not file-local row IDs."""
    ids=np.unique(np.asarray(task_ids))
    if len(ids)<3: raise ValueError('at least three physical groups required')
    if not (np.isfinite(train_frac) and np.isfinite(val_frac) and 0<train_frac<1 and 0<val_frac<1 and train_frac+val_frac<1):
        raise ValueError('strictly positive train/val/test fractions required')
    ids=np.random.default_rng(seed).permutation(ids)
    nt=max(1,min(len(ids)-2,int(len(ids)*train_frac)))
    nv=max(1,min(len(ids)-nt-1,int(len(ids)*val_frac)))
    return {'train':ids[:nt],'val':ids[nt:nt+nv],'test':ids[nt+nv:]}


def validate_arrays(a, meta):
    if meta.get('schema')!=SCHEMA or meta.get('spec_version')!=SPEC_VERSION:
        raise ValueError('obsolete/unknown schema; regenerate V1 sensor-evidence data')
    if meta.get('spec_fields')!=list(SPEC_FIELDS) or meta.get('target_fields')!=list(TARGET_FIELDS):
        raise ValueError('field order differs from the versioned schema')
    missing=set(REQUIRED)-set(a)
    if missing: raise ValueError(f'missing arrays: {sorted(missing)}')
    n=len(a['task_id'])
    if n==0: raise ValueError('empty dataset')
    for k in REQUIRED:
        if len(a[k])!=n or a[k].dtype.hasobject: raise ValueError(f'invalid length/dtype: {k}')
    for k in ('task_id','attempt_index','steps'):
        if a[k].shape!=(n,) or a[k].dtype.kind not in 'iu' or (a[k]<0).any():
            raise ValueError(f'{k} must contain nonnegative integer metadata')
    for k in ('group_id','status','role'):
        if a[k].shape!=(n,) or a[k].dtype.kind not in 'US': raise ValueError(f'invalid {k}')
    for k in ('evidence_mask','target_mask','context_eligible'):
        if a[k].dtype!=np.bool_: raise ValueError(f'{k} must be bool')
    shapes={'spec':(n,len(SPEC_FIELDS)),'target':(n,len(TARGET_FIELDS)),
            'target_mask':(n,len(TARGET_FIELDS)),'initial_observation':(n,OBS_DIM),
            'context_eligible':(n,)}
    for k,shape in shapes.items():
        if a[k].shape!=shape: raise ValueError(f'invalid shape: {k}')
    if a['evidence'].ndim!=3 or a['evidence'].shape[2]!=EVIDENCE_DIM or a['evidence'].shape[1]<1:
        raise ValueError('evidence must be N x L x 40')
    if a['evidence_mask'].shape!=a['evidence'].shape[:2]: raise ValueError('evidence mask shape')
    for k in ('spec','target','initial_observation','evidence'):
        if not np.isfinite(a[k]).all(): raise ValueError(f'nonfinite values: {k}')
    if (a['target'][~a['target_mask']]!=0).any(): raise ValueError('unobserved targets require zero + false mask')
    if (a['evidence'][~a['evidence_mask']]!=0).any(): raise ValueError('padded evidence requires zero + false mask')
    if not set(a['status'].tolist())<=set(STATUSES): raise ValueError('unknown termination status')
    if not set(a['role'].tolist())<={'probe','candidate'}: raise ValueError('unknown role')
    for j in range(6,len(TARGET_FIELDS)):
        v=a['target'][a['target_mask'][:,j],j]
        if not np.isin(v,[0.,1.]).all(): raise ValueError('observed binary labels must be 0/1')
    from ..attempt.spec import AttemptSpec
    for row in a['spec']: AttemptSpec.from_vector(row)
    # Local IDs must map to exactly one physical group; groups may span files.
    for tid in np.unique(a['task_id']):
        ii=a['task_id']==tid
        if len(np.unique(a['group_id'][ii]))!=1: raise ValueError('one task ID maps to multiple physical groups')
    pairs=list(zip(a['group_id'].tolist(),a['attempt_index'].tolist()))
    if len(set(pairs))!=n: raise ValueError('duplicate physical-group/attempt identity')
    eligible=a['context_eligible']
    if np.any(eligible & (a['status']!='recovered')) or np.any(eligible & ~a['evidence_mask'].any(1)):
        raise ValueError('retry context must be a recovered trial with actual evidence')
    for name, expected in [('success','success'),('recovered','recovered'),('contact','contact')]:
        j=TARGET_FIELDS.index(name)
        known=a['target_mask'][:,j]
        if not np.array_equal(a['target'][known,j].astype(bool),a['status'][known]==expected):
            raise ValueError('termination status conflicts with observed binary label')
    if np.any(a['context_eligible'] & (a['target'][:,TARGET_FIELDS.index('contact')]>0)):
        raise ValueError('contact cannot be retry context')
    invalid=np.isin(a['status'],['invalid_start','invalid_state'])
    if a['target_mask'][invalid].any(): raise ValueError('invalid trials cannot have valid targets')


def save_attempt_dataset(path, *, meta, audit=None, **arrays):
    path=Path(path)
    if path.suffix!='.npz': raise ValueError('output must end in .npz; no implicit suffix changes')
    a={k:np.asarray(v) for k,v in arrays.items()}
    validate_arrays(a,meta)
    for k,v in (audit or {}).items():
        if not k.startswith('audit_'): raise ValueError('audit namespace required')
        if np.asarray(v).dtype.hasobject: raise ValueError('pickle/object audit data forbidden')
        a[k]=np.asarray(v)
    encoded=json.dumps(meta,ensure_ascii=False,allow_nan=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez_compressed(f,meta=encoded,**a);f.flush();os.fsync(f.fileno())
        # Atomic no-clobber publication. A racing writer cannot overwrite a file.
        os.link(tmp,path)
    finally:
        os.unlink(tmp)
    return path


class AttemptDataset:
    def __init__(self,path):
        self.path=Path(path)
        with np.load(path,allow_pickle=False) as f:
            self.meta=json.loads(str(f['meta'].item())) if 'meta' in f else {}
            if self.meta.get('schema')!=SCHEMA:
                raise ValueError('V0/unknown dataset: regenerate; missing evidence cannot be inferred from labels')
            self.arrays={k:f[k].copy() for k in REQUIRED if k in f}
            self._audit={k:f[k].copy() for k in f.files if k.startswith('audit_')}
        validate_arrays(self.arrays,self.meta)
        for k in REQUIRED: setattr(self,k,self.arrays[k])

    def __len__(self): return len(self.task_id)

    def __getitem__(self,index):
        # Neither a query outcome nor task identifiers belong inside inputs.
        return {'inputs':{'candidate':self.spec[index].copy(),
                          'initial_observation':self.initial_observation[index].copy(),
                          'protocol':np.array([float(self.role[index]=='probe')],np.float32)},
                'targets':self.target[index].copy(),'target_mask':self.target_mask[index].copy()}

    def record_metadata(self,index):
        return {k:self.arrays[k][index].item() for k in
                ('task_id','group_id','attempt_index','status','steps','role','context_eligible')}

    @property
    def audit(self): return self._audit

    @property
    def candidate_indices(self): return np.flatnonzero(self.role=='candidate')

    def task_indices(self,task_id): return np.flatnonzero(self.task_id==task_id)

    def split(self,*,train_frac=.8,val_frac=.1,seed=0):
        parts=split_task_ids(self.group_id,train_frac=train_frac,val_frac=val_frac,seed=seed)
        return {k:np.flatnonzero(np.isin(self.group_id,v)) for k,v in parts.items()}

    def make_query(self,index,history_indices):
        history=np.asarray(history_indices)
        if history.size==0: history=np.empty(0,dtype=np.int64)
        if history.ndim!=1 or history.dtype.kind not in 'iu': raise ValueError('history indices must be integer vector')
        if np.any(history<0) or np.any(history>=len(self)) or len(np.unique(history))!=len(history):
            raise ValueError('invalid/duplicate history indices')
        if (index in history or np.any(self.attempt_index[history]>=self.attempt_index[index])
            or np.any(self.group_id[history]!=self.group_id[index])):
            raise ValueError('query/future/cross-task history leakage')
        if not self.context_eligible[history].all(): raise ValueError('unsafe/unrecovered trial cannot be retry context')
        sample=self[index]
        sample['inputs'].update(context_spec=self.spec[history].copy(),
                                context_evidence=self.evidence[history].copy(),
                                context_mask=self.evidence_mask[history].copy(),
                                context_protocol=(self.role[history]=='probe').astype(np.float32))
        return sample
