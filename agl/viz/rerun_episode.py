"""Rerun backend for inspecting real GapEnv evaluation episodes.

Rerun is an optional dependency. The simulator/training path never imports it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .episode import EvalEpisode, gap_outline, quat_rotate_wxyz, rect_yz


def require_rerun():
    try:
        import rerun as rr
    except ImportError as exc:
        raise RuntimeError(
            "Rerun is optional. Install it with: pip install -r requirements-viz.txt"
        ) from exc
    return rr


def _body_circle(p, q, radius=.16, n=32):
    a = np.linspace(0.0, 2*np.pi, n + 1)
    body = np.c_[radius*np.cos(a), radius*np.sin(a), np.zeros_like(a)]
    return quat_rotate_wxyz(q, body) + np.asarray(p)


def _body_axes(p, q, scale=.25):
    basis = np.array([[scale,0,0], [0,scale,0], [0,0,scale]], dtype=float)
    tips = quat_rotate_wxyz(q, basis) + np.asarray(p)
    origin = np.asarray(p)
    return [np.stack([origin, tip]) for tip in tips]


def _phase_color(phase):
    if phase == "CONTACT" or phase == "OOB":
        return [220, 50, 47]
    if phase == "SUCCESS":
        return [38, 139, 82]
    if phase == "GIVE_UP":
        return [120, 85, 170]
    if phase.startswith("ATTEMPT"):
        return [230, 140, 40]
    return [70, 125, 180]


def _log_static(rr, ep: EvalEpisode):
    task, meta = ep.task, ep.meta
    wx, th = float(task["wall_x"]), float(task["thick"])
    cy, cz = float(task["gap_cy"]), float(task["gap_cz"])
    y0, y1 = min(-2.5, cy - 1.5), max(2.5, cy + 1.5)
    z0, z1 = 0.0, max(3.2, cz + float(task["gap_h"]) + 1.0)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    front = rect_yz(wx, y0, y1, z0, z1)
    back = rect_yz(wx + th, y0, y1, z0, z1)
    corners = [
        np.array([[wx,y,z], [wx+th,y,z]], dtype=float)
        for y, z in ((y0,z0), (y1,z0), (y1,z1), (y0,z1))
    ]
    wall_strips = [front, back, *corners]
    rr.log(
        "world/wall/frame",
        rr.LineStrips3D(
            wall_strips,
            colors=[[105,115,125]] * len(wall_strips),
            radii=[0.012] * len(wall_strips),
        ),
        static=True,
    )
    rr.log(
        "world/wall/gap",
        rr.LineStrips3D(
            [gap_outline(task, wx), gap_outline(task, wx + th)],
            colors=[[210,55,70], [210,55,70]],
            radii=[0.018, 0.018],
        ),
        static=True,
    )

    ground = np.array([
        [min(-1.0, float(ep.rec["p"][:,0].min())-.3), y0, 0],
        [max(wx+th+1.0, float(ep.rec["p"][:,0].max())+.3), y0, 0],
        [max(wx+th+1.0, float(ep.rec["p"][:,0].max())+.3), y1, 0],
        [min(-1.0, float(ep.rec["p"][:,0].min())-.3), y1, 0],
        [min(-1.0, float(ep.rec["p"][:,0].min())-.3), y0, 0],
    ], dtype=float)
    rr.log(
        "world/ground/bounds",
        rr.LineStrips3D([ground], colors=[[130,130,130]], radii=[0.006]),
        static=True,
    )

    retry_x = float(meta.get("retry_x", 1.2))
    retry = rect_yz(retry_x, y0, y1, .2, z1)
    rr.log(
        "world/planes/retry",
        rr.LineStrips3D([retry], colors=[[80,150,220]], radii=[0.008]),
        static=True,
    )

    succ_x = wx + th + float(meta.get("succ_margin", .4))
    success = rect_yz(succ_x, y0, y1, .2, z1)
    rr.log(
        "world/planes/success",
        rr.LineStrips3D([success], colors=[[50,170,90]], radii=[0.008]),
        static=True,
    )

    if meta.get("info_gate_enabled", False):
        start = wx - float(meta.get("info_probe_distance", 1.0))
        active = start + float(meta.get("info_probe_ramp", .25))
        rr.log(
            "world/probe_zone",
            rr.LineStrips3D(
                [rect_yz(start, y0, y1, .2, z1), rect_yz(active, y0, y1, .2, z1)],
                colors=[[235,180,40], [235,180,40]],
                radii=[0.012, 0.012],
            ),
            static=True,
        )


def log_episode(ep: EvalEpisode, out: str | Path | None = None, spawn: bool = False):
    """Write an episode to an RRD file or stream it to a local Viewer."""
    if out is not None and spawn:
        raise ValueError("choose either out=<file.rrd> or spawn=True, not both")
    rr = require_rerun()
    rr.init("AutonomousGapLearning_GapEnv", spawn=spawn, strict=True)
    if out is not None:
        out = Path(out)
        if out.exists():
            raise FileExistsError(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rr.save(out)

    _log_static(rr, ep)
    dt = float(ep.meta.get("dt_ctrl", .025))
    body_r = float(ep.meta.get("body_r", .16))
    last_phase = None

    for i in range(ep.steps):
        rr.set_time("step", sequence=i)
        rr.set_time("sim_time", duration=i * dt)

        p = np.asarray(ep.rec["p"][i], dtype=float)
        q = np.asarray(ep.rec["q"][i], dtype=float)
        v = np.asarray(ep.rec["v"][i], dtype=float)
        phase = ep.phase(i)
        color = _phase_color(phase)

        rr.log(
            "world/drone/body",
            rr.LineStrips3D([_body_circle(p, q, radius=body_r)], colors=[color], radii=[0.016]),
        )
        axes = _body_axes(p, q)
        rr.log(
            "world/drone/axes",
            rr.LineStrips3D(
                axes,
                colors=[[230,60,55], [45,170,85], [55,105,220]],
                radii=[0.012, 0.012, 0.012],
            ),
        )
        rr.log(
            "world/drone/center",
            rr.Points3D([p], colors=[color], radii=[.055], labels=[phase]),
        )
        if i > 0:
            rr.log(
                "world/trajectory",
                rr.LineStrips3D(
                    [ep.rec["p"][:i+1]], colors=[[55,120,195]], radii=[.012]
                ),
            )
        rr.log(
            "world/vectors/velocity",
            rr.LineStrips3D(
                [np.stack([p, p + .25*v])], colors=[[30,190,210]], radii=[.014]
            ),
        )

        activation = ep.probe_activation(i)
        probe_wind = np.asarray(ep.task.get("probe_wind", np.zeros(3)), dtype=float)
        if activation > 0 and np.linalg.norm(probe_wind) > 0:
            rr.log(
                "world/vectors/probe_wind",
                rr.LineStrips3D(
                    [np.stack([p, p + .30*activation*probe_wind])],
                    colors=[[235,180,40]],
                    radii=[.018],
                ),
            )
        else:
            # Rerun keeps the last value of an entity on a timeline. Explicitly
            # clear the privileged vector when the vehicle retreats out of the
            # information gate so a stale arrow cannot look like a current force.
            rr.log("world/vectors/probe_wind", rr.Clear(recursive=False))

        speed = float(np.linalg.norm(v))
        rr.log("telemetry/speed_mps", rr.Scalars(speed))
        rr.log("telemetry/clearance_m", rr.Scalars(float(ep.rec["clear"][i])))
        if "clear_pre" in ep.rec:
            rr.log("telemetry/clearance_pre_m", rr.Scalars(float(ep.rec["clear_pre"][i])))
        rr.log("telemetry/attempt_id", rr.Scalars(float(ep.rec["attempt_id"][i])))
        rr.log("telemetry/probe_activation", rr.Scalars(activation))
        rr.log("telemetry/collision", rr.Scalars(float(bool(ep.rec["collision"][i]))))
        if "risk" in ep.rec:
            rr.log("telemetry/policy_risk", rr.Scalars(float(ep.rec["risk"][i])))
        if "act" in ep.rec:
            for j, name in enumerate(("thrust", "roll_rate", "pitch_rate", "yaw_rate")):
                rr.log(f"action/{name}", rr.Scalars(float(ep.rec["act"][i, j])))

        if ep.frames is not None:
            rr.log("sensors/ego_rgb", rr.Image(ep.frames[i]))

        if phase != last_phase or ("end_event" in ep.rec and bool(ep.rec["end_event"][i])):
            rr.log(
                "events/state",
                rr.TextLog(
                    f"{phase} | step={i} | speed={speed:.3f} m/s | "
                    f"clearance={float(ep.rec['clear'][i]):.3f} m"
                ),
            )
        last_phase = phase

    return out
