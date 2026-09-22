"""Common, dependency-light episode representation for visualization."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


_REQUIRED_REC = ("p", "v", "q", "clear", "attempt_id", "collision")
_TASK_SCALARS = ("wall_x", "thick", "gap_cy", "gap_cz", "gap_w", "gap_h", "gap_roll")


@dataclass
class EvalEpisode:
    source: Path
    task_id: int
    steps: int
    rec: dict[str, np.ndarray]
    task: dict[str, object]
    meta: dict
    frames: np.ndarray | None = None

    def phase(self, i: int) -> str:
        if bool(self.rec.get("collision", np.zeros(self.steps))[i]):
            return "CONTACT"
        if "success" in self.rec and bool(self.rec["success"][i]):
            return "SUCCESS"
        if "gave_up" in self.rec and bool(self.rec["gave_up"][i]):
            return "GIVE_UP"
        if "oob" in self.rec and bool(self.rec["oob"][i]):
            return "OOB"
        if "phase" in self.rec:
            return str(self.rec["phase"][i]).upper() + " (COMMAND)"
        aid = int(self.rec["attempt_id"][i])
        in_attempt = bool(self.rec.get("in_attempt", np.zeros(self.steps))[i])
        return f"ATTEMPT_{aid}" if in_attempt or aid > 0 else "APPROACH"

    def probe_activation(self, i: int) -> float:
        if not self.meta.get("info_gate_enabled", False):
            return 0.0
        distance = float(self.meta.get("info_probe_distance", 0.0))
        ramp = max(float(self.meta.get("info_probe_ramp", 0.0)), 1e-9)
        start = float(self.task["wall_x"]) - distance
        x = float(self.rec["p"][i, 0])
        return float(np.clip((x - start) / ramp, 0.0, 1.0))


def _decode_meta(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    if isinstance(value, dict):
        return value
    return {}


def load_eval_episode(path, task_id: int = 0) -> EvalEpisode:
    """Load exactly one valid episode from an evaluate.py NPZ.

    The per-task steps field is authoritative: data after that boundary may
    belong to an auto-reset environment and is never exposed to the viewer.
    """
    path = Path(path)
    d = np.load(path, allow_pickle=False)
    for key in _REQUIRED_REC:
        npz_key = f"rec_{key}"
        if npz_key not in d:
            raise KeyError(f"missing {npz_key}")
    n = int(d["rec_p"].shape[1])
    if not 0 <= task_id < n:
        raise ValueError(f"task must be in [0,{n})")
    steps = int(d["steps"][task_id]) if "steps" in d else int(d["rec_p"].shape[0])
    if steps <= 0:
        d.close()
        raise ValueError(f"task {task_id} has no valid recorded steps")
    steps = min(steps, int(d["rec_p"].shape[0]))

    rec = {}
    for key in ("p", "v", "q", "act", "clear", "clear_pre", "attempt_id",
                "in_attempt", "end_event", "end_outcome", "success", "collision",
                "collision_high", "done", "oob", "gave_up", "risk", "phase"):
        npz_key = f"rec_{key}"
        if npz_key in d:
            rec[key] = np.asarray(d[npz_key][:steps, task_id])

    task = {}
    for key in _TASK_SCALARS:
        npz_key = f"task_{key}"
        if npz_key not in d:
            raise KeyError(f"missing {npz_key}")
        task[key] = float(d[npz_key][task_id])
    for key in ("mass", "twr", "wind_mag"):
        npz_key = f"task_{key}"
        if npz_key in d:
            task[key] = float(d[npz_key][task_id])
    if "task_probe_wind" in d:
        task["probe_wind"] = np.asarray(d["task_probe_wind"][task_id], dtype=np.float32)
    else:
        task["probe_wind"] = np.zeros(3, dtype=np.float32)

    frames = None
    if "rec_frames" in d and task_id < d["rec_frames"].shape[1]:
        # Stored as T,N,C,H,W; expose T,H,W,C for visualization.
        frames = np.asarray(d["rec_frames"][:steps, task_id]).transpose(0, 2, 3, 1)

    meta = _decode_meta(d["meta"] if "meta" in d else None)
    d.close()
    return EvalEpisode(path, task_id, steps, rec, task, meta, frames)


def quat_rotate_wxyz(q: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Rotate (...,3) body-frame points by a normalized [w,x,y,z] quaternion."""
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    r = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    return p @ r.T


def gap_outline(task: dict, x: float | None = None) -> np.ndarray:
    """Closed 3-D polyline for the rolled rectangular opening."""
    cy, cz = float(task["gap_cy"]), float(task["gap_cz"])
    hw, hh = .5 * float(task["gap_w"]), .5 * float(task["gap_h"])
    roll = float(task["gap_roll"])
    yz = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh], [-hw, -hh]], dtype=float)
    c, s = np.cos(roll), np.sin(roll)
    rot = np.array([[c, -s], [s, c]])
    yz = yz @ rot.T + np.array([cy, cz])
    xx = float(task["wall_x"]) if x is None else float(x)
    return np.c_[np.full(len(yz), xx), yz]


def rect_yz(x: float, y0: float, y1: float, z0: float, z1: float) -> np.ndarray:
    return np.array([[x,y0,z0], [x,y1,z0], [x,y1,z1], [x,y0,z1], [x,y0,z0]], dtype=float)
