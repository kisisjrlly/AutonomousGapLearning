import numpy as np
import pytest

from agl.eval.legacy_open_loop_probe_retreat import run
from agl.eval.safe_probe_retreat import run as deprecated_run
from agl.viz.episode import load_eval_episode


def test_legacy_open_loop_remains_explicitly_non_protocol():
    out = run(n_pairs=1, probe_steps=2, stop_steps=2, retreat_steps=3)
    assert out["steps_executed"] > 0
    assert not out["protocol_completed"]
    assert "contact_free_only" in out["acceptance_scope"]


def test_deprecated_entry_point_warns():
    with pytest.deprecated_call():
        out = deprecated_run(n_pairs=1, probe_steps=1, stop_steps=1, retreat_steps=1)
    assert not out["protocol_completed"]


def test_legacy_record_reproducible_and_viewer_compatible(tmp_path):
    paths = [tmp_path / "a.npz", tmp_path / "b.npz"]
    results = [run(n_pairs=1, record=p, seed=7) for p in paths]
    assert results[0] == results[1]
    a, b = [load_eval_episode(p, 0) for p in paths]
    np.testing.assert_array_equal(a.rec["p"], b.rec["p"])
    assert a.phase(0) == "PROBE (COMMAND)"
    assert not results[0]["protocol_completed"]
