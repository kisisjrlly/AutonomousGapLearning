"""Validate dataset structure and report label/evidence coverage, not efficacy."""
import argparse
import json
import numpy as np
from .attempt_dataset import AttemptDataset
from ..attempt.outcome import TARGET_FIELDS


def audit(path):
    ds=AttemptDataset(path)
    candidates=ds.candidate_indices
    statuses,counts=np.unique(ds.status,return_counts=True)
    warnings=[]
    if not ds.context_eligible.any(): warnings.append('No recovered sensor histories: context training is not ready.')
    for field in ('success','contact'):
        j=TARGET_FIELDS.index(field)
        ii=candidates[ds.target_mask[candidates,j]]
        if len(ii)==0 or len(np.unique(ds.target[ii,j]))<2:
            warnings.append(f'Candidate {field} label has fewer than two observed classes.')
    groups=np.unique(ds.group_id)
    if len(groups)<3: warnings.append('Fewer than three groups: no train/validation/test split.')
    lengths=[len(ds.task_indices(t)) for t in np.unique(ds.task_id)]
    return {'schema_valid':True,'records':len(ds),'physical_groups':len(groups),'attempts_per_task':lengths,
            'candidate_records':len(candidates),'context_eligible_records':int(ds.context_eligible.sum()),
            'status_counts':dict(zip(statuses.tolist(),counts.tolist())),
            'target_observed_counts':dict(zip(TARGET_FIELDS,ds.target_mask.sum(0).tolist())),
            'warnings':warnings,'scope':'data-quality report; NOT a training result or safety certificate'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True)
    args=p.parse_args();print(json.dumps(audit(args.input),indent=2,allow_nan=False))

if __name__=='__main__':main()
