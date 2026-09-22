import json

import numpy as np

from agl.viz.episode import gap_outline, load_eval_episode, quat_rotate_wxyz


def _fake_eval(path):
    t, n = 5, 2
    p = np.zeros((t, n, 3), np.float32)
    p[:, :, 0] = np.arange(t)[:, None] * .1
    q = np.zeros((t, n, 4), np.float32)
    q[..., 0] = 1.
    rec = {
        "rec_p": p,
        "rec_v": np.zeros((t,n,3), np.float32),
        "rec_q": q,
        "rec_clear": np.ones((t,n), np.float32),
        "rec_attempt_id": np.zeros((t,n), np.float32),
        "rec_collision": np.zeros((t,n), np.float32),
        "rec_frames": np.zeros((t,1,3,4,6), np.uint8),
        "steps": np.array([3, 5]),
        "task_wall_x": np.array([3., 3.]),
        "task_thick": np.array([.2, .2]),
        "task_gap_cy": np.array([0., 0.]),
        "task_gap_cz": np.array([1.5, 1.5]),
        "task_gap_w": np.array([.6, .6]),
        "task_gap_h": np.array([.5, .5]),
        "task_gap_roll": np.array([0., .2]),
        "task_probe_wind": np.array([[0.,1.,0.], [0.,-1.,0.]], np.float32),
        "meta": json.dumps({
            "dt_ctrl": .025, "retry_x": 1.2, "succ_margin": .4,
            "info_gate_enabled": True, "info_probe_distance": 1.,
            "info_probe_ramp": .25,
        }),
    }
    np.savez_compressed(path, **rec)


def test_eval_episode_loader_clips_auto_reset_tail(tmp_path):
    path = tmp_path / "eval.npz"
    _fake_eval(path)
    ep = load_eval_episode(path, 0)
    assert ep.steps == 3
    assert ep.rec["p"].shape == (3, 3)
    assert ep.frames.shape == (3, 4, 6, 3)
    assert ep.task["probe_wind"].tolist() == [0., 1., 0.]


def test_eval_episode_without_saved_camera_for_other_task(tmp_path):
    path = tmp_path / "eval.npz"
    _fake_eval(path)
    ep = load_eval_episode(path, 1)
    assert ep.steps == 5
    assert ep.frames is None


def test_geometry_helpers():
    task = dict(wall_x=3., gap_cy=0., gap_cz=1.5, gap_w=.6, gap_h=.4, gap_roll=0.)
    outline = gap_outline(task)
    assert outline.shape == (5, 3)
    assert np.allclose(outline[0], outline[-1])
    pts = np.array([[1., 0., 0.]])
    assert np.allclose(quat_rotate_wxyz(np.array([1.,0.,0.,0.]), pts), pts)
