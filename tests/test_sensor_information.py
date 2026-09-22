from agl.eval.verify_sensor_information import run


def test_hidden_information_reaches_actor_observation():
    out = run(n_pairs=2, steps=12, seed=123)
    assert out["far_obs_vec_pair_max_abs_delta"] == 0.0
    assert out["far_ego_rgb_pair_max_abs_delta"] == 0.0
    assert out["near_obs_vec_pair_max_abs_delta"] > 0.0
    assert out["near_vbody_y_pair_min_abs_delta"] > 0.0
    assert out["min_clearance_m"] > 0.0
