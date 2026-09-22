from agl.eval.safe_probe_retreat import run


def test_scripted_probe_retreat_reports_real_acceptance():
    out = run(n_pairs=2, probe_steps=2, stop_steps=2, retreat_steps=3)
    assert out['steps_executed'] > 0
    assert out['safe_count'] + out['contact_count'] + out['done_count'] == 4
    assert out['min_clearance'] == out['min_clearance']
