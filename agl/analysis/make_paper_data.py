"""Aggregate all eval outputs into the single JSON the paper and figures cite.

For the main model, episodes from the three seeds are pooled (episode = the
independent unit; seed-level spread is reported alongside). All CIs are
95% bootstrap; attempt-level rates use episode-cluster bootstrap.
"""
import argparse
import glob
import json
import os

import numpy as np

from ..eval.metrics import ABORT, SUCCESS, load_and_summarize
from .stats import bootstrap_ci, cluster_bootstrap_rate, paired_bootstrap_diff, two_proportion_z

SEEDS = ("full_s1", "full_s2", "full_s3")
ABLATIONS = ("no_memory", "reset_attempts", "no_prev_action", "no_aux")


def pool(results_dir, runs, tag):
    """Pool attempt/episode tables across runs for one eval tag."""
    atts, eps = [], []
    off = 0
    per_seed = []
    for r in runs:
        p = os.path.join(results_dir, r, f"eval_{tag}.npz")
        if not os.path.exists(p):
            continue
        s, att, ep = load_and_summarize(p)
        per_seed.append({"run": r, "summary": s})
        for a in att:
            a = dict(a)
            a["env"] += off
            atts.append(a)
        for e in ep:
            e = dict(e)
            e["env"] += off
            eps.append(e)
        off += s["n_episodes"]
    return atts, eps, per_seed


def cond_success_ci(atts, max_k=4):
    out = {"k": [], "rate": [], "lo": [], "hi": [], "n": []}
    for k in range(1, max_k + 1):
        rows = [{"env": a["env"], "s": float(a["outcome"] == SUCCESS)}
                for a in atts if a["k"] == k and a["feasible"]]
        if len(rows) < 10:
            break
        pt, lo, hi = cluster_bootstrap_rate(rows, "s")
        out["k"].append(k)
        out["rate"].append(pt)
        out["lo"].append(lo)
        out["hi"].append(hi)
        out["n"].append(len(rows))
    return out


def adaptation_tests(atts):
    """Paired alignment-error change after aborts + attempt-k success tests."""
    res = {}
    pairs = []
    for a in [x for x in atts if x["outcome"] == ABORT]:
        nxt = [b for b in atts if b["env"] == a["env"] and b["k"] == a["k"] + 1]
        if nxt:
            pairs.append((a["align_err"], nxt[0]["align_err"]))
    if pairs:
        res["align_paired"] = paired_bootstrap_diff([p[0] for p in pairs],
                                                    [p[1] for p in pairs])
    k1 = [a for a in atts if a["k"] == 1 and a["feasible"]]
    k2 = [a for a in atts if a["k"] == 2 and a["feasible"]]
    if k1 and k2:
        x1 = sum(a["outcome"] == SUCCESS for a in k1)
        x2 = sum(a["outcome"] == SUCCESS for a in k2)
        z, p = two_proportion_z(x2, len(k2), x1, len(k1))
        res["k2_vs_k1"] = {"z": z, "p": p, "rate1": x1 / len(k1),
                           "rate2": x2 / len(k2), "n1": len(k1), "n2": len(k2)}
    return res


def full_summary(atts, eps):
    from ..eval.metrics import summarize
    s = summarize(atts, eps)
    feas = [e for e in eps if e["feasible"]]
    s["success_overall_ci"] = bootstrap_ci([float(e["success"]) for e in feas])
    s["first_attempt_success_ci"] = cluster_bootstrap_rate(
        [{"env": a["env"], "s": float(a["outcome"] == SUCCESS)}
         for a in atts if a["k"] == 1 and a["feasible"]], "s")
    s["collision_ep_ci"] = bootstrap_ci([float(e["collision"]) for e in eps])
    s["cond_success"] = cond_success_ci(atts)
    s["adaptation"] = adaptation_tests(atts)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/paper/summary.json")
    args = ap.parse_args()
    out = {}
    conditions = [("full", SEEDS, "id"), ("full", SEEDS, "ood_geom"),
                  ("full", SEEDS, "ood_dyn"), ("full_wipe", SEEDS, "id_wipe")]
    for abl in ABLATIONS:
        conditions.append((abl, (abl,), "id"))
        for sp in ("ood_geom", "ood_dyn"):
            if glob.glob(os.path.join(args.results, abl, f"eval_{sp}.npz")):
                conditions.append((abl, (abl,), sp))
    for name, runs, tag in conditions:
        atts, eps, per_seed = pool(args.results, runs, tag)
        if not eps:
            print(f"[skip] {name}/{tag}: no data")
            continue
        out[f"{name}/{tag}"] = {
            "pooled": full_summary(atts, eps),
            "per_seed": [{"run": d["run"],
                          "success": d["summary"]["success_overall_feasible"],
                          "first": d["summary"]["first_attempt_success"],
                          "coll": d["summary"]["collision_rate_ep"]}
                         for d in per_seed],
        }
        print(f"[ok] {name}/{tag}: {len(eps)} eps, {len(atts)} attempts")
    # cross-condition significance: does wiping context kill the k2 gain?
    def get(cond, *keys):
        d = out.get(cond, {}).get("pooled", {})
        for k in keys:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d
    a_full = get("full/id", "adaptation", "k2_vs_k1")
    a_wipe = get("full_wipe/id_wipe", "adaptation", "k2_vs_k1")
    if a_full and a_wipe:
        z, p = two_proportion_z(
            round(a_full["rate2"] * a_full["n2"]), a_full["n2"],
            round(a_wipe["rate2"] * a_wipe["n2"]), a_wipe["n2"])
        out["tests/full_vs_wipe_k2"] = {"z": z, "p": p}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
