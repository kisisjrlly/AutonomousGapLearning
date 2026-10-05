"""Static geometry scan for the current no-wind narrow-gap milestone."""
import argparse
import json
import math
from pathlib import Path
import torch

from ..config import load_config
from ..sim import collision, scene


@torch.no_grad()
def scan(cfg=None):
    cfg = cfg or load_config()
    deg = torch.linspace(0., 90., 181)
    beta = torch.deg2rad(deg)
    width, height = cfg.task.narrow_gap_width, cfg.task.narrow_gap_height
    diameter, body_height = 2 * cfg.sim.body_r, 2 * cfg.sim.body_hh
    req_w = diameter * beta.cos() + body_height * beta.sin()
    req_h = diameter * beta.sin() + body_height * beta.cos()
    slack = torch.minimum((width - req_w) / 2, (height - req_h) / 2)
    task = scene.sample_tasks(len(deg), cfg, 1., "cpu")
    for k, v in dict(gap_w=width, gap_h=height, gap_roll=0., wall_x=3., thick=.15,
                     gap_cy=0., gap_cz=1.5).items(): task[k].fill_(v)
    q = torch.zeros(len(deg), 4); q[:, 0] = torch.cos(beta/2); q[:, 1] = torch.sin(beta/2)
    p = torch.zeros(len(deg), 3); p[:, 2] = 1.5
    bpts = collision.body_points(cfg.sim.body_r, cfg.sim.body_hh, "cpu")
    sampled = torch.full_like(deg, float("inf"))
    for x in torch.linspace(3. - cfg.sim.body_r, 3.15 + cfg.sim.body_r, 65):
        p[:, 0] = x
        sampled = torch.minimum(sampled, collision.clearance(
            p, q, task, bpts, cfg.sim.arena_y, cfg.sim.arena_z))
    allowed = deg <= cfg.task.feas_roll_max_deg
    score = slack.masked_fill(~allowed, -float("inf"))
    best = int(score.argmax())
    fit = deg[(slack > 0) & allowed]
    report = {
        "scope": "static_geometry_NOT_controlled_flight",
        "wind_model": "disabled", "gap_width_m": width, "gap_height_m": height,
        "body_diameter_m": diameter, "body_height_m": body_height,
        "gap_narrower_than_body": bool(width < diameter),
        "level_projection_slack_m": float(slack[0]),
        "first_fitting_roll_deg": float(fit[0]) if len(fit) else None,
        "best_roll_deg_within_limit": float(deg[best]),
        "best_projection_slack_m": float(slack[best]),
        "sampled_sweep_clearance_at_best_m": float(sampled[best]),
        "feasible_with_label_margin": bool(slack[best] >= cfg.task.feas_margin),
    }
    return report, deg.numpy(), slack.numpy(), sampled.numpy()


def plot(path, report, deg, slack, sampled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    import numpy as np
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    w, h = report["gap_width_m"], report["gap_height_m"]
    r, hh = report["body_diameter_m"]/2, report["body_height_m"]/2
    for a, roll in zip(ax[:2], (0., report["best_roll_deg_within_limit"])):
        a.add_patch(Rectangle((-w/2, -h/2), w, h, facecolor="white", edgecolor="black"))
        a.add_patch(Rectangle((-.25, -.3), .5, .6, facecolor="#d9dce2", alpha=.4))
        t = math.radians(roll); R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        corners = np.array([[-r,-hh],[r,-hh],[r,hh],[-r,hh]]) @ R.T
        a.add_patch(Polygon(corners, color="#d65b4b" if roll == 0 else "#368b73", alpha=.8))
        a.set(xlim=(-.25,.25), ylim=(-.3,.3), aspect="equal", title=f"Roll {roll:.0f}°")
    ax[2].plot(deg, slack*1000, label="projected slack")
    ax[2].plot(deg, sampled*1000, "--", label="sampled sweep")
    ax[2].axhline(0, color="black", lw=.8); ax[2].set(xlabel="roll (deg)", ylabel="clearance/slack (mm)")
    ax[2].legend(fontsize=8)
    fig.suptitle("No wind | 0.30 m gap vs 0.32 m body | geometry only")
    fig.savefig(path, dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--plot")
    a = p.parse_args(); result = scan(); print(json.dumps(result[0], indent=2))
    if a.plot: plot(a.plot, *result)
    if not result[0]["gap_narrower_than_body"] or not result[0]["feasible_with_label_margin"]: raise SystemExit(1)


if __name__ == "__main__": main()
