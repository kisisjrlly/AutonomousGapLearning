from agl.eval.verify_info_gate import run_check


def test_information_gate_environment_smoke_cpu():
    out = run_check(n_pairs=2, device="cpu", steps=8, seed=123)
    assert out["far_pair_max_wind_delta"] == 0.0
    assert out["near_pair_min_wind_delta"] > 0.0
    assert out["near_pair_opposite_wind_fraction"] == 1.0
    assert out["pair_min_lateral_velocity_separation"] > 0.0
    assert out["pair_opposite_lateral_motion_fraction"] == 1.0
