"""Generate all paper figures from results/paper/summary.json + eval npz + logs.

Usage:  python -m agl.analysis.make_figures --summary results/paper/summary.json \
                                             --out paper/figures [--results results]
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from .figures import (fig_adaptation, fig_bars, fig_episode, fig_training)
from ..eval.metrics import segment_attempts


def cond_success_to_plot(s):
    """summary['pooled']['cond_success'] -> dict for fig_adaptation."""
    return {"k": s["k"], "rate": s["rate"], "lo": s["lo"], "hi": s["hi"]}


def pick_episode(summary_path, results_dir, seed_run="full_s1"):
    """Find an episode with >=2 attempts and the deepest abort margin for the anatomy figure."""
    npz = os.path.join(results_dir, seed_run, "eval_id.npz")
    if not os.path.exists(npz):
        return None
    d = np.load(npz, allow_pickle=True)
    rec = {k[4:]: d[k] for k in d.files if k.startswith("rec_")}
    task = {k[5:]: d[k] for k in d.files if k.startswith("task_")}
    atts, eps = segment_attempts(rec, task, d["steps"])
    cands = [e["env"] for e in eps if e["n_attempts"] >= 2]
    if not cands:
        cands = list(range(len(eps)))
    # prefer an episode with a clean abort-then-success pattern
    env_i = None
    for e in eps:
        if e["env"] in cands and e["success"]:
            env_i = e["env"]
            break
    if env_i is None:
        env_i = cands[0]
    return {"rec": rec, "task": task, "attempts": atts,
            "env_i": env_i, "steps_used": d["steps"]}


def fig1_overview(results_dir, out_dir):
    """System schematic + example onboard frames from eval recordings."""
    from matplotlib import pyplot as plt
    from .figures import paper_style
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    # scene: wall with hole + retry plane + success plane
    wx, th = 3.0, 0.15
    ax.add_patch(plt.Rectangle((wx, -3), th, 6, color="#52514e", lw=0))
    ax.add_patch(plt.Rectangle((wx, 0.9), th, 0.6, color="white", lw=0))
    ax.axvline(1.2, color="#9a9891", lw=0.8, ls=":")
    ax.text(1.2, 2.4, "retry plane x=1.2", ha="center", fontsize=6.5, color="#52514e")
    ax.axvline(wx + th + 0.4, color="#9a9891", lw=0.8, ls=":")
    ax.text(wx + th + 0.4, 2.4, "success plane", ha="center", fontsize=6.5, color="#52514e")
    # drone cartoon + attempt arrows
    ax.add_patch(plt.Circle((0.5, 1.4), 0.12, color="#2a78d6", zorder=5))
    ax.annotate("", xy=(1.9, 1.35), xytext=(0.7, 1.4),
                arrowprops=dict(arrowstyle="-|>", color="#e34948", lw=1.2))
    ax.annotate("", xy=(1.0, 1.6), xytext=(1.8, 1.55),
                arrowprops=dict(arrowstyle="-|>", color="#eb6834", lw=1.2))
    ax.annotate("", xy=(3.05, 1.2), xytext=(2.0, 1.35),
                arrowprops=dict(arrowstyle="-|>", color="#1baf7a", lw=1.2))
    ax.text(0.62, 1.75, "drone", fontsize=6.5, color="#2a78d6")
    ax.text(1.35, 1.12, "attempt 1 →", fontsize=6.5, color="#e34948")
    ax.text(0.9, 1.72, "abort/retreat", fontsize=6.5, color="#eb6834")
    ax.text(2.4, 1.05, "attempt 2 →", fontsize=6.5, color="#1baf7a")
    ax.text(0.5, -2.6, "gap: width 0.34–0.90 m, height 0.30–0.80 m, roll ±40°, "
                       "thickness 0.05–0.30 m", fontsize=6.5, color="#52514e")
    ax.text(0.5, -2.9, "dynamics: mass 0.60–0.95 kg, TWR 2.2–3.4, latency 0–50 ms, "
                       "wind 0–2.5 m/s (all hidden from policy)", fontsize=6.5, color="#52514e")
    ax.set_xlim(-0.5, 4.6); ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal"); ax.axis("off")
    # onboard view filmstrip
    ax2 = axes[1]
    npz = os.path.join(results_dir, "full_s1", "eval_id.npz")
    if os.path.exists(npz):
        d = np.load(npz, allow_pickle=True)
        fr = d.get("rec_frames") if "rec_frames" in d.files else None
        if fr is not None:
            rows = []
            for t in (0, len(fr) // 3, 2 * len(fr) // 3):
                rows.append(fr[t, 0].transpose(1, 2, 0))
            grid = np.concatenate(rows, axis=1)
            ax2.imshow(grid, interpolation="nearest")
            ax2.set_title("Onboard view (32×24, flat-shaded)")
        else:
            ax2.text(0.5, 0.5, "no frames recorded", ha="center", transform=ax2.transAxes)
            ax2.set_axis_off()
    else:
        ax2.text(0.5, 0.5, "no eval data yet", ha="center", transform=ax2.transAxes)
        ax2.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_overview.png"))
    plt.close(fig)


def make_all(summary_path, out_dir, results_dir):
    os.makedirs(out_dir, exist_ok=True)
    s = json.load(open(summary_path))
    # --- training curves: read logs directly
    run_logs = {"full": sorted(glob.glob("runs/full_s*/log.csv")),
                "no_memory": ["runs/no_memory/log.csv"],
                "reset_attempts": ["runs/reset_attempts/log.csv"],
                "no_prev_action": ["runs/no_prev_action/log.csv"],
                "no_aux": ["runs/no_aux/log.csv"]}
    existing = {k: [p for p in v if os.path.exists(p)] for k, v in run_logs.items()}
    existing = {k: v for k, v in existing.items() if v}
    if existing:
        fig_training(existing, os.path.join(out_dir, "fig2_training.png"))
    # --- adaptation
    cond = {}
    for name, tag in (("full", "full/id"), ("wipe", "full_wipe/id_wipe"),
                      ("reset_attempts", "reset_attempts/id"),
                      ("no_memory", "no_memory/id")):
        if tag in s:
            cs = s[tag]["pooled"].get("cond_success")
            if cs and len(cs["k"]):
                cond[name] = cond_success_to_plot(cs)
    if cond:
        fig_adaptation(cond, os.path.join(out_dir, "fig3_adaptation.png"))
    # --- grouped bars: OOD generalization + safety
    groups_ood = []
    for split, label in (("id", "ID"), ("ood_geom", "OOD-geom"), ("ood_dyn", "OOD-dyn")):
        tag = f"full/{split}"
        if tag in s:
            p = s[tag]["pooled"]
            mean, lo, hi = p["success_overall_ci"]
            groups_ood.append((label, [("full", mean, lo, hi)]))
    if groups_ood:
        fig_bars(groups_ood, os.path.join(out_dir, "fig6_ood.png"),
                 "Success (feasible)", ylim=(0, 1))
    # --- episode anatomy
    ep = pick_episode(summary_path, results_dir)
    if ep:
        fig_episode(ep["rec"], ep["task"], ep["attempts"], ep["env_i"],
                    ep["steps_used"], os.path.join(out_dir, "fig4_episode.png"))
    fig1_overview(results_dir, out_dir)
    print("figures written to", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/paper/summary.json")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    make_all(args.summary, args.out, args.results)


if __name__ == "__main__":
    main()
