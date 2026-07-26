"""Paper figures. Categorical colors follow the entity (variant), fixed order;
sequential ramps encode attempt index; one axis per panel; thin marks."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# validated categorical palette (light mode), fixed assignment by variant
C = {
    "full": "#2a78d6",
    "no_memory": "#eb6834",
    "reset_attempts": "#1baf7a",
    "no_prev_action": "#eda100",
    "no_aux": "#e87ba4",
    "wipe": "#2a78d6",       # same entity as full; dashed linestyle carries condition
}
LABEL = {
    "full": "Full (memory across attempts)",
    "no_memory": "No memory (frame-only)",
    "reset_attempts": "Memory reset between attempts",
    "no_prev_action": "No previous action",
    "no_aux": "No auxiliary heads",
    "wipe": "Full, context wiped at eval",
}
SEQ = ["#c7dcf5", "#8fbceb", "#5a9ade", "#2a78d6", "#1c5296", "#123663"]  # blue ramp


def paper_style():
    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.4, "legend.frameon": False,
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
    })


def read_log(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
        except ValueError:
            pass
    return out


def smooth(y, w=5):
    if len(y) < w:
        return y
    ker = np.ones(w) / w
    pad = np.concatenate([np.full(w // 2, y[0]), y, np.full(w - 1 - w // 2, y[-1])])
    return np.convolve(pad, ker, mode="valid")


def fig_training(run_logs, out_path):
    """run_logs: {variant: [csv paths (seeds)]}. 4 panels, shared x (steps)."""
    paper_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.7))
    panels = [("ep/success_feasible", "Success rate (feasible tasks)"),
              ("difficulty", "Curriculum difficulty λ"),
              ("ep/coll_high", "High-energy collision rate"),
              ("ep/n_attempts", "Attempts per episode")]
    for ax, (key, title) in zip(axes, panels):
        for variant, paths in run_logs.items():
            curves = []
            for p in paths:
                d = read_log(p)
                if key not in d:
                    continue
                curves.append((d["steps"] / 1e6, smooth(np.nan_to_num(d[key]))))
            if not curves:
                continue
            if len(curves) > 1:
                xs = curves[0][0]
                ys = np.stack([np.interp(xs, c[0], c[1]) for c in curves])
                ax.plot(xs, ys.mean(0), color=C[variant], label=LABEL[variant])
                ax.fill_between(xs, ys.min(0), ys.max(0), color=C[variant], alpha=0.18, lw=0)
            else:
                ax.plot(curves[0][0], curves[0][1], color=C[variant], label=LABEL[variant])
        ax.set_title(title)
        ax.set_xlabel("Env steps (millions)")
        ax.grid(True, lw=0.3, alpha=0.35)
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_adaptation(cond_data, out_path):
    """cond_data: {variant: {"k": [1..], "rate": [...], "lo": [...], "hi": [...]}}
    Conditional success by attempt index with bootstrap CIs."""
    paper_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for variant, d in cond_data.items():
        ls = "--" if variant == "wipe" else "-"
        ks = np.array(d["k"], float)
        ax.errorbar(ks, d["rate"],
                    yerr=[np.array(d["rate"]) - np.array(d["lo"]),
                          np.array(d["hi"]) - np.array(d["rate"])],
                    color=C[variant], ls=ls, marker="o", ms=3.5, capsize=2,
                    lw=1.4, elinewidth=0.8, label=LABEL[variant])
    ax.set_xlabel("Attempt index k")
    ax.set_ylabel("P(success | attempt k reached)")
    ax.set_xticks(sorted({int(k) for d in cond_data.values() for k in d["k"]}))
    ax.set_ylim(0, 1)
    ax.grid(True, lw=0.3, alpha=0.35)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_scene_topdown(ax, task, i):
    wx, th = task["wall_x"][i], task["thick"][i]
    cy, w = task["gap_cy"][i], task["gap_w"][i]
    ax.add_patch(plt.Rectangle((wx, -5), th, 10, color="#52514e", alpha=0.85, lw=0))
    # gap opening projected on y (approx for roll)
    half = 0.5 * w * abs(np.cos(task["gap_roll"][i])) \
        + 0.5 * task["gap_h"][i] * abs(np.sin(task["gap_roll"][i]))
    ax.add_patch(plt.Rectangle((wx, cy - half), th, 2 * half, color="white", lw=0))
    ax.axvline(1.2, color="#9a9891", lw=0.7, ls=":")


def draw_scene_side(ax, task, i):
    wx, th = task["wall_x"][i], task["thick"][i]
    cz, h = task["gap_cz"][i], task["gap_h"][i]
    ax.add_patch(plt.Rectangle((wx, 0), th, 4.5, color="#52514e", alpha=0.85, lw=0))
    half = 0.5 * h * abs(np.cos(task["gap_roll"][i])) \
        + 0.5 * task["gap_w"][i] * abs(np.sin(task["gap_roll"][i]))
    ax.add_patch(plt.Rectangle((wx, cz - half), th, 2 * half, color="white", lw=0))
    ax.axhline(0, color="#9a9891", lw=0.7)


def fig_episode(rec, task, attempts, env_i, steps_used, out_path):
    """One episode anatomy: top-down + side trajectories (attempt-colored) +
    clearance/speed/forward-position time series with abort markers."""
    paper_style()
    L = int(steps_used[env_i])
    p = rec["p"][:L, env_i]
    v = np.linalg.norm(rec["v"][:L, env_i], axis=-1)
    clear = rec["clear"][:L, env_i]
    aid = rec["attempt_id"][:L, env_i].astype(int)
    atts = [a for a in attempts if a["env"] == env_i]
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.7], hspace=0.55, wspace=0.42)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 2])
    draw_scene_topdown(ax1, task, env_i)
    draw_scene_side(ax2, task, env_i)
    kmax = max(aid.max(), 1)
    for k in range(0, kmax + 1):
        m = aid == k
        if not m.any():
            continue
        col = SEQ[min(2 + k, len(SEQ) - 1)] if k > 0 else "#c3c2b7"
        ax1.plot(p[m, 0], p[m, 1], color=col, lw=1.2)
        ax2.plot(p[m, 0], p[m, 2], color=col, lw=1.2)
    for a in atts:
        if a["outcome"] == 0:  # abort: mark deepest point
            t_deep = a["t0"] + int(np.argmax(rec["p"][a["t0"]:a["t1"] + 1, env_i, 0]))
            ax1.plot(p[t_deep, 0], p[t_deep, 1], "x", color="#e34948", ms=5, mew=1.2)
            ax2.plot(p[t_deep, 0], p[t_deep, 2], "x", color="#e34948", ms=5, mew=1.2)
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax1.set_xlim(-1, task["wall_x"][env_i] + 1.2)
    ax1.set_ylim(-2.2, 2.2)
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("z (m)")
    ax2.set_xlim(-1, task["wall_x"][env_i] + 1.2)
    ax2.set_ylim(0, 3.2)
    tt = np.arange(L) * 0.025
    ax3.plot(tt, p[:, 0], color="#2a78d6", lw=1.1)
    ax3.axhline(task["wall_x"][env_i], color="#52514e", lw=0.8)
    ax3.axhline(1.2, color="#9a9891", lw=0.7, ls=":")
    ax3.set_ylabel("x (m)")
    ax3.set_xticklabels([])
    ax4.plot(tt, clear, color="#1baf7a", lw=1.1, label="clearance")
    ax4.plot(tt, v, color="#eb6834", lw=1.1, label="speed")
    ax4.axhline(0, color="#e34948", lw=0.6, ls="--")
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("m / m s$^{-1}$")
    ax4.legend(loc="upper right")
    for ax in (ax3, ax4):
        ax.grid(True, lw=0.3, alpha=0.35)
    fig.savefig(out_path)
    plt.close(fig)


def fig_bars(groups, out_path, ylabel, ylim=(0, 1), figsize=(3.4, 2.2)):
    """groups: [(group_label, [(variant, val, lo, hi)])] grouped bar chart w/ CI."""
    paper_style()
    fig, ax = plt.subplots(figsize=figsize)
    ng = len(groups)
    variants = [v for v, *_ in groups[0][1]]
    nv = len(variants)
    width = 0.8 / nv
    for j, var in enumerate(variants):
        xs, ys, los, his = [], [], [], []
        for gi, (glabel, vals) in enumerate(groups):
            v, val, lo, hi = vals[j]
            xs.append(gi + (j - nv / 2 + 0.5) * width)
            ys.append(val); los.append(val - lo); his.append(hi - val)
        ax.bar(xs, ys, width * 0.92, color=C.get(var, "#9a9891"),
               label=LABEL.get(var, var), lw=0)
        ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor="#0b0b0b",
                    elinewidth=0.7, capsize=1.8)
    ax.set_xticks(range(ng))
    ax.set_xticklabels([g for g, _ in groups])
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", lw=0.3, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
