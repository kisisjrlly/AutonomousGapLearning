"""Offline attempt segmentation and README §12 metrics from eval trajectories."""
import argparse
import json

import numpy as np

ABORT, SUCCESS, COLLISION, CUTOFF = 0, 1, 2, 3


def segment_attempts(rec, task, steps_used, retry_x=1.2, ep_len=960):
    """Returns per-attempt table (list of dicts) and per-episode table."""
    T, N = rec["clear"].shape
    attempts, episodes = [], []
    for i in range(N):
        L = int(steps_used[i]) if steps_used[i] > 0 else T
        aid = rec["attempt_id"][:L, i]
        ina = rec["in_attempt"][:L, i] > 0.5
        ee = rec["end_event"][:L, i] > 0.5
        eo = rec["end_outcome"][:L, i]
        px = rec["p"][:L, i, 0]
        clear = rec["clear"][:L, i]
        done_t = np.nonzero(rec["done"][:L, i] > 0.5)[0]
        done_t = int(done_t[0]) if len(done_t) else L - 1
        n_att = int(aid.max())
        ep_succ = rec["success"][:L, i].any()
        ep_coll = rec["collision"][:L, i].any()
        ep_coll_hi = rec["collision_high"][:L, i].any()
        oob = rec["oob"][:L, i].any()
        gave_up = rec["gave_up"][:L, i].any() and not (ep_succ or ep_coll or oob)
        timeout = (int(steps_used[i]) >= ep_len
                   and not (ep_succ or ep_coll or oob or gave_up))
        for k in range(1, n_att + 1):
            mask = (aid == k) & (ina | ee)
            idxs = np.nonzero(mask)[0]
            if len(idxs) == 0:
                continue
            t0, t1 = int(idxs[0]), int(idxs[-1])
            end_idx = np.nonzero(ee[t0:t1 + 1] & (aid[t0:t1 + 1] == k))[0]
            outcome = int(eo[t0 + end_idx[-1]]) if len(end_idx) else CUTOFF
            seg = slice(t0, t1 + 1)
            deep = t0 + int(np.argmax(px[seg]))
            wall_x = task["wall_x"][i]
            # gap-frame offset at deepest point
            dy = rec["p"][deep, i, 1] - task["gap_cy"][i]
            dz = rec["p"][deep, i, 2] - task["gap_cz"][i]
            cr, sr = np.cos(task["gap_roll"][i]), np.sin(task["gap_roll"][i])
            u_off, v_off = cr * dy + sr * dz, -sr * dy + cr * dz
            # body roll vs gap roll at deepest point (roll of thrust axis in y-z)
            qw, qx, qy_, qz = rec["q"][deep, i]
            bz = np.array([2 * (qx * qz + qw * qy_), 2 * (qy_ * qz - qw * qx),
                           1 - 2 * (qx * qx + qy_ * qy_)])
            body_roll = np.arctan2(-bz[1], bz[2])
            attempts.append({
                "env": i, "k": k, "t0": t0, "t1": t1, "outcome": outcome,
                "min_clear": float(clear[seg].min()),
                "depth": float((px[deep] - retry_x) / max(wall_x - retry_x, 1e-6)),
                "u_off": float(u_off), "v_off": float(v_off),
                "align_err": float(np.hypot(u_off, v_off)),
                "roll_err": float(abs(((body_roll - task["gap_roll"][i]) + np.pi / 2)
                                      % np.pi - np.pi / 2)),
                "speed_deep": float(np.linalg.norm(rec["v"][deep, i])),
                "feasible": bool(task["feasible"][i] > 0.5),
            })
        sat = (np.abs(rec["act"][:L, i]) >= 0.99).any(axis=-1).mean() if L else 0.0
        # shadow safety kernel (measurement only): conservative stopping envelope
        twr = float(task.get("twr", np.full(N, 2.8))[i])
        a_brake = 9.81 * max(twr * twr - 1.0, 0.25) ** 0.5
        spd = np.linalg.norm(rec["v"][:L, i], axis=-1)
        stop_d = spd * spd / (2 * a_brake) + spd * 0.075
        dclear = np.diff(clear, prepend=clear[:1])
        viol = (clear < stop_d) & (dclear < 0) & ina
        stop_viol = float(viol.mean()) if L else 0.0
        episodes.append({
            "env": i, "n_attempts": n_att, "success": bool(ep_succ),
            "collision": bool(ep_coll), "collision_high": bool(ep_coll_hi),
            "timeout": bool(timeout), "oob": bool(oob), "gave_up": bool(gave_up),
            "ep_len": int(done_t + 1), "min_clear": float(clear.min()) if L else 10.0,
            "sat_frac": float(sat),
            "feasible": bool(task["feasible"][i] > 0.5),
            "gap_w": float(task["gap_w"][i]), "geo_margin": float(task["geo_margin"][i]),
            "stop_viol": stop_viol,
        })
    return attempts, episodes


def _rate(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def summarize(attempts, episodes, max_k=5):
    att = attempts
    eps = episodes
    feas = [e for e in eps if e["feasible"]]
    infeas = [e for e in eps if not e["feasible"]]
    by_k = {k: [a for a in att if a["k"] == k and a["feasible"]] for k in range(1, max_k + 1)}
    aborts = [a for a in att if a["outcome"] == ABORT]
    # abort recovery: an abort recovers if a later attempt starts or episode
    # ends without collision/oob
    ep_by_env = {e["env"]: e for e in eps}
    recov = []
    for a in aborts:
        e = ep_by_env[a["env"]]
        later = any(b["env"] == a["env"] and b["k"] == a["k"] + 1 for b in att)
        recov.append(later or not (e["collision"] or e["oob"]))
    succ_after_abort = []
    for a in [x for x in att if x["outcome"] == ABORT and x["feasible"]]:
        nxt = [b for b in att if b["env"] == a["env"] and b["k"] == a["k"] + 1]
        if nxt:
            succ_after_abort.append(nxt[0]["outcome"] == SUCCESS)
    n_att_succ = [e["n_attempts"] for e in feas if e["success"]]
    out = {
        "n_episodes": len(eps), "n_feasible": len(feas), "n_infeasible": len(infeas),
        "success_overall_feasible": _rate([e["success"] for e in feas]),
        "success_within": {k: _rate([e["success"] and e["n_attempts"] <= k for e in feas])
                           for k in range(1, max_k + 1)},
        "first_attempt_success": _rate([a["outcome"] == SUCCESS for a in by_k.get(1, [])]),
        "cond_success_by_attempt": {
            k: _rate([a["outcome"] == SUCCESS for a in by_k[k]]) for k in by_k if by_k[k]},
        "n_reached_attempt": {k: len(by_k[k]) for k in by_k},
        "success_given_prior_abort": _rate(succ_after_abort),
        "n_prior_abort_pairs": len(succ_after_abort),
        "attempts_to_success_mean": _rate(n_att_succ),
        "abort_rate_per_attempt": _rate([a["outcome"] == ABORT for a in att]),
        "abort_min_clear_mean": _rate([a["min_clear"] for a in aborts]),
        "abort_depth_mean": _rate([a["depth"] for a in aborts]),
        "abort_recovery_rate": _rate(recov),
        "n_attempts_mean": _rate([e["n_attempts"] for e in eps]),
        "ep_with_abort_frac": _rate([any(a["outcome"] == ABORT for a in att
                                         if a["env"] == e["env"]) for e in eps]),
        "collision_rate_ep": _rate([e["collision"] for e in eps]),
        "collision_high_rate_ep": _rate([e["collision_high"] for e in eps]),
        "collision_rate_per_attempt": _rate([a["outcome"] == COLLISION for a in att]),
        "succ_traversal_min_clear_mean": _rate(
            [a["min_clear"] for a in att if a["outcome"] == SUCCESS]),
        "giveup_rate_infeasible": _rate([e["gave_up"] for e in infeas]),
        "giveup_rate_feasible": _rate([e["gave_up"] for e in feas]),
        "attempts_before_giveup_infeasible": _rate(
            [e["n_attempts"] for e in infeas if e["gave_up"]]),
        "success_rate_infeasible_label": _rate([e["success"] for e in infeas]),
        "timeout_rate": _rate([e["timeout"] for e in eps]),
        "oob_rate": _rate([e["oob"] for e in eps]),
        "sat_frac_mean": _rate([e["sat_frac"] for e in eps]),
        "stop_envelope_violation_frac": _rate([e["stop_viol"] for e in eps]),
        "ep_len_mean": _rate([e["ep_len"] for e in eps]),
    }
    # in-context adaptation evidence: alignment-error change after an abort
    pairs = []
    for a in [x for x in att if x["outcome"] == ABORT]:
        nxt = [b for b in att if b["env"] == a["env"] and b["k"] == a["k"] + 1]
        if nxt:
            pairs.append((a["align_err"], nxt[0]["align_err"]))
    if pairs:
        p1 = np.array([p[0] for p in pairs])
        p2 = np.array([p[1] for p in pairs])
        out["align_err_before_after_abort"] = [float(p1.mean()), float(p2.mean())]
        out["align_improve_frac"] = float((p2 < p1).mean())
        out["n_align_pairs"] = len(pairs)
    return out


def load_and_summarize(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    rec = {k[4:]: d[k] for k in d.files if k.startswith("rec_")}
    task = {k[5:]: d[k] for k in d.files if k.startswith("task_")}
    att, eps = segment_attempts(rec, task, d["steps"])
    s = summarize(att, eps)
    s["meta"] = json.loads(str(d["meta"]))
    return s, att, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    all_s = {}
    for p in args.npz:
        s, _, _ = load_and_summarize(p)
        all_s[p] = s
        print(f"== {p}")
        print(json.dumps(s, indent=1, default=float))
    if args.out:
        json.dump(all_s, open(args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
