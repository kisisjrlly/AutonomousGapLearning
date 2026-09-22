from agl.eval.animate_eval_3d import load_record


def test_load_record_from_existing_eval():
    rec, task = load_record('results/full_s1/eval_id.npz', 0)
    assert rec['p'].shape[1] == 3
    assert len(rec['p']) == len(rec['clear'])
    assert task['gap_w'] > 0
    assert 'gap_roll' in task
