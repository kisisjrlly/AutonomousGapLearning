import torch

from agl.eval.closed_loop_probe import run
from agl.eval.same_state_intervention import BRANCHES, prepare


def test_same_state_branches_use_complete_recovery_snapshot(tmp_path):
    recovery_dir = tmp_path / "recovery"
    out = run(n_pairs=1, out_dir=recovery_dir)
    assert out["protocol_completed"]

    result = prepare(recovery_dir / "recovery.pt")
    assert result["branches"] == BRANCHES
    assert result["initial_observation_equal"]
    assert result["complete_environment_equal"]
    assert "frame_delay_buf" in result["snapshot_keys"]
    assert "rng_cpu" in result["snapshot_keys"]
    assert result["observation_shapes"]["vec"][-1] == 18

    saved = torch.load(recovery_dir / "recovery.pt", weights_only=False)
    assert result["acceptance"]["protocol_completed"]
    assert saved["acceptance"]["protocol_completed"]
