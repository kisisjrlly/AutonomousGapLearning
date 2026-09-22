from agl.eval.safe_probe_retreat import run
import numpy as np
from agl.viz.episode import load_eval_episode


def test_scripted_probe_retreat_reports_real_acceptance():
    out = run(n_pairs=2, probe_steps=2, stop_steps=2, retreat_steps=3)
    assert out['steps_executed'] > 0
    assert out['safe_count'] + out['unsafe_count'] == 4
    assert out['min_clearance'] == out['min_clearance']


def test_record_reproducible_and_viewer_compatible(tmp_path):
    paths = [tmp_path / 'a.npz', tmp_path / 'b.npz']
    results = [run(n_pairs=1, record=p, seed=7) for p in paths]
    assert results[0] == results[1]
    a, b = [load_eval_episode(p, 0) for p in paths]
    np.testing.assert_array_equal(a.rec['p'], b.rec['p'])
    assert a.phase(0) == 'PROBE (COMMAND)'
    assert a.phase(12) == 'STOP (COMMAND)'
    assert a.phase(20) == 'RETREAT (COMMAND)'
    assert a.meta['state_timing'].startswith('pre_action')
    assert not results[0]['protocol_completed']
