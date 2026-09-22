from agl.eval.same_state_intervention import prepare


def test_same_state_branch_has_matched_pre_state_and_divergent_post_state():
    result = prepare(n_pairs=2, steps=12, device='cpu')
    assert set(result['state_schema']) >= {'p', 'v', 'q'}
    assert max(result['post_pair_max_abs_delta'].values()) > 0
    assert not result['probe_done'].any()
    assert not result['probe_contact'].any()
    assert result['probe_min_clearance'].min() > 0
    for name, tensor in result['pre_state'].items():
        assert tensor.shape[0] == 4
        assert (tensor[0::2] == tensor[1::2]).all(), name
