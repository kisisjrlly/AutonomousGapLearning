import json
from pathlib import Path

from agl.eval.plot_safe_probe import plot_reference, plot_sweep
from agl.eval.safe_probe_reference import run_bank
from agl.eval.sweep_safe_probe import sweep


def test_reference_and_sweep_plots(tmp_path):
    ref = run_bank(n=6)
    sweep_data = sweep(n=6)
    ref_path = tmp_path / 'reference.json'
    sweep_path = tmp_path / 'sweep.json'
    ref_path.write_text(json.dumps(ref))
    sweep_path.write_text(json.dumps(sweep_data))
    ref_png = tmp_path / 'reference.png'
    sweep_png = tmp_path / 'sweep.png'
    plot_reference(ref, ref_png)
    plot_sweep(sweep_data, sweep_png)
    assert ref_png.stat().st_size > 1000
    assert sweep_png.stat().st_size > 1000
