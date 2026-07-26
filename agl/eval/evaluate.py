"""Evaluation harness: fixed seeded task banks, full trajectory recording.

Splits:
  id        held-out instances from the training distribution (difficulty=1)
  ood_geom  gap width/roll/thickness extrapolated ~15% beyond training range
  ood_dyn   wind/mass/thrust/latency-constant extrapolated ~15%

Each env runs exactly one episode from the bank; policy is deterministic
(mean action). --wipe-context zeroes the GRU hidden state whenever an attempt
ends (abort), isolating cross-attempt memory at eval time only.
"""
import argparse
import copy
import json
import os

import numpy as np
import torch

from ..config import load_config, Config
from ..models.policy import Policy
from ..sim import scene
from ..sim.env import GapEnv, OUTCOME


def split_cfg(cfg: Config, split: str) -> Config:
    c = copy.deepcopy(cfg)
    t = c.task
    if split == "ood_geom":
        t.width_lo = round(t.width_lo * 0.85, 4)          # 0.34 -> 0.289
        t.roll_max_deg = t.roll_max_deg * 1.15            # 40 -> 46 deg
        t.thick_hi = round(t.thick_hi * 1.15, 4)          # 0.30 -> 0.345
    elif split == "ood_dyn":
        t.wind_max = t.wind_max * 1.15
        t.mass_lo, t.mass_hi = round(t.mass_lo * 0.85, 4), round(t.mass_hi * 1.15, 4)
        t.twr_lo, t.twr_hi = round(t.twr_lo * 0.9, 4), round(t.twr_hi * 1.1, 4)
        t.tau_rate_hi = round(t.tau_rate_hi * 1.15, 4)
    elif split != "id":
        raise ValueError(split)
    return c


SPLIT_SEED = {"id": 7001, "ood_geom": 7002, "ood_dyn": 7003}


def make_bank(cfg, split, n, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(SPLIT_SEED[split])
    return scene.sample_tasks(n, split_cfg(cfg, split), 1.0, device, gen)


@torch.no_grad()
def run_eval(model, cfg, split, n_tasks, device, wipe_context=False,
             reset_between_attempts=False, noise_seed=1234):
    torch.manual_seed(noise_seed)  # observation noise reproducibility
    ecfg = copy.deepcopy(cfg)
    ecfg.sim.n_envs = n_tasks
    ecfg.curriculum.enabled = False
    env = GapEnv(ecfg, device, difficulty=1.0)
    bank = make_bank(ecfg, split, n_tasks, device)
    env._reset_envs(torch.arange(n_tasks, device=device), tasks=bank)
    T = ecfg.sim.ep_len + 2
    N = n_tasks
    rec = {k: np.zeros((T, N), dtype=np.float32) for k in
           ("clear", "attempt_id", "in_attempt", "end_event", "end_outcome",
            "success", "collision", "collision_high", "done", "oob", "gave_up")}
    for k, d in (("p", 3), ("v", 3), ("q", 4), ("act", 4)):
        rec[k] = np.zeros((T, N, d), dtype=np.float32)
    task_np = {k: bank[k].cpu().numpy() for k in
               ("gap_w", "gap_h", "gap_roll", "wall_x", "thick", "gap_cy",
                "gap_cz", "feasible", "geo_margin")}
    task_np["wind_mag"] = bank["dyn"]["wind_steady"].norm(dim=-1).cpu().numpy()
    task_np["mass"] = bank["dyn"]["mass"].cpu().numpy()

    h = model.init_hidden(N, device)
    obs = env.observe()
    priv = torch.zeros(N, 21, device=device)
    finished = torch.zeros(N, dtype=torch.bool, device=device)
    steps_used = np.zeros(N, dtype=np.int64)
    for t in range(T):
        mean, _, _, h_new = model.step(obs["img"], obs["vec"], priv, h)
        act = mean.clamp(-1, 1)
        rec["p"][t] = env.state["p"].cpu().numpy()
        rec["v"][t] = env.state["v"].cpu().numpy()
        rec["q"][t] = env.state["q"].cpu().numpy()
        rec["act"][t] = act.cpu().numpy()
        obs, rew, done, info = env.step(act)
        priv = info["priv"]
        h = h_new
        for k_src, k_dst in (("clearance", "clear"), ("attempt_id", "attempt_id"),
                             ("in_attempt", "in_attempt"), ("end_event", "end_event"),
                             ("end_outcome", "end_outcome"), ("success", "success"),
                             ("collision", "collision"), ("collision_high", "collision_high"),
                             ("oob", "oob"), ("gave_up", "gave_up")):
            rec[k_dst][t] = info[k_src].float().cpu().numpy()
        newly = done & ~finished
        rec["done"][t] = newly.float().cpu().numpy()
        steps_used[newly.cpu().numpy().astype(bool)] = t + 1
        finished |= done
        wipe = done.clone()
        if wipe_context or reset_between_attempts:
            wipe |= info["end_event"]
        h = h * (~wipe).unsqueeze(-1).float()
        if finished.all():
            break
    tmax = int(steps_used.max())
    rec = {k: v[:tmax] for k, v in rec.items()}
    return rec, task_np, steps_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="id,ood_geom,ood_dyn")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--wipe-context", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--noise-seed", type=int, default=1234)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    cfg = load_config(overrides=ck["cfg"])
    model = Policy(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    os.makedirs(args.out, exist_ok=True)
    for split in args.splits.split(","):
        rec, task, steps = run_eval(
            model, cfg, split, args.n, args.device,
            wipe_context=args.wipe_context,
            reset_between_attempts=cfg.model.reset_between_attempts,
            noise_seed=args.noise_seed)
        tag = f"{split}_wipe" if args.wipe_context else split
        np.savez_compressed(
            os.path.join(args.out, f"eval_{tag}.npz"),
            steps=steps, **{f"rec_{k}": v for k, v in rec.items()},
            **{f"task_{k}": v for k, v in task.items()},
            meta=json.dumps({"ckpt": args.ckpt, "split": split, "n": args.n,
                             "wipe": args.wipe_context,
                             "train_steps": ck.get("steps", -1),
                             "noise_seed": args.noise_seed}))
        print(f"saved eval_{tag}.npz  (T={rec['clear'].shape[0]})", flush=True)


if __name__ == "__main__":
    main()
