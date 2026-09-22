from agl.eval.sweep_safe_probe import sweep


def test_safety_sweep_has_no_reference_contacts():
    result = sweep(n=6)
    assert len(result['rows']) == 36
    assert max(r['real_contacts'] + r['swapped_contacts'] for r in result['rows']) == 0
    assert all(r['inflation'] > 0 for r in result['rows'])
