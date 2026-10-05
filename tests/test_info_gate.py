from agl.eval.verify_info_gate import run_check


def test_information_gate_environment_smoke_cpu():
    out = run_check(n_pairs=2, device="cpu", steps=8, seed=123)
    assert out["wind_model"] == "disabled"
    assert out["sampled_wind_max"] == 0.0
    assert out["effective_wind_max"] == 0.0
    assert out["paired_motion_max_abs_delta"] == 0.0
    assert out["passed"]
