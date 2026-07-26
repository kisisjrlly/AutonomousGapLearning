"""Training entry: rollout collection + recurrent PPO + auto-curriculum."""
import argparse
import csv
import os
import subprocess
import time

import torch
import yaml

from ..config import load_config
from ..models.policy import Policy
from ..sim import collision
from ..sim.env import GapEnv
from .ppo import PPO, Rollout, compute_gae

LOG2PI = torch.log(torch.tensor(2 * torch.pi))


class EpisodeStats:
    FIELDS = ("success", "coll_high", "coll_soft", "oob", "gave_up", "timeout",
              "n_attempts", "ep_len", "min_clear", "feasible")

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = {k: 0.0 for k in self.FIELDS}
        self.n = 0
        self.succ_feas = 0.0
        self.n_feas = 0
        self.giveup_infeas = 0.0
        self.n_infeas = 0

    def add(self, rec):
        n = rec["success"].numel()
        self.n += n
        for k in self.FIELDS:
            self.sums[k] += rec[k].sum().item()
        feas = rec["feasible"] > 0.5
        self.n_feas += int(feas.sum())
        self.succ_feas += rec["success"][feas].sum().item()
        self.n_infeas += int((~feas).sum())
        self.giveup_infeas += rec["gave_up"][~feas].sum().item()

    def summary(self):
        if self.n == 0:
            return {}
        out = {f"ep/{k}": v / self.n for k, v in self.sums.items()}
        out["ep/n"] = self.n
        if self.n_feas:
            out["ep/success_feasible"] = self.succ_feas / self.n_feas
        if self.n_infeas:
            out["ep/giveup_infeasible"] = self.giveup_infeas / self.n_infeas
        return out


class Trainer:
    def __init__(self, cfg, run_dir, device=None):
        self.cfg = cfg
        self.dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        torch.manual_seed(cfg.ppo.seed)
        self.dev = torch.device(device or cfg.sim.device)
        self.env = GapEnv(cfg, self.dev)
        self.model = Policy(cfg).to(self.dev)
        self.ppo = PPO(cfg, self.model, self.dev)
        self.ro = Rollout(cfg.ppo.rollout_len, cfg.sim.n_envs, cfg, self.dev)
        self.h = self.model.init_hidden(cfg.sim.n_envs, self.dev)
        self.pending = torch.zeros(cfg.sim.n_envs, dtype=torch.bool, device=self.dev)
        self.obs = self.env.observe()
        self.priv = self._priv_now()
        self.stats = EpisodeStats()
        self.succ_ema = 0.0
        self.iter = 0
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg.to_dict(), f)
        try:
            h = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            open(os.path.join(run_dir, "GIT_COMMIT"), "w").write(h + "\n")
        except Exception:
            pass
        self.csv_path = os.path.join(run_dir, "log.csv")
        self.csv_keys = (
            ["iter", "steps", "time", "sps", "difficulty", "succ_ema", "rew_mean",
             "log_std", "pi_loss", "v_loss", "ent", "aux_loss", "clipfrac", "kl", "lr"]
            + [f"ep/{k}" for k in EpisodeStats.FIELDS]
            + ["ep/n", "ep/success_feasible", "ep/giveup_infeasible"])
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(self.csv_keys)
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(os.path.join(run_dir, "tb"))
        except Exception:
            pass

    def _priv_now(self):
        clear = collision.clearance(self.env.state["p"], self.env.state["q"],
                                    self.env.task, self.env.bpts,
                                    self.cfg.sim.arena_y, self.cfg.sim.arena_z)
        return self.env.privileged(clear)

    @torch.no_grad()
    def collect(self):
        cfg, ro, env = self.cfg, self.ro, self.env
        L = cfg.ppo.bptt_chunk
        for t in range(ro.T):
            ro.hreset[t] = self.pending
            self.h = self.h * (~self.pending).unsqueeze(-1).float()
            if t % L == 0:
                ro.h0[t // L] = self.h
            img = self.obs["img"]
            ro.img[t] = (img * 255.0).to(torch.uint8)
            ro.vec[t] = self.obs["vec"]
            ro.priv[t] = self.priv
            mean, std, val, h_new = self.model.step(img, self.obs["vec"], self.priv, self.h)
            act = mean + std * torch.randn_like(mean)
            logp = (-0.5 * ((act - mean) / std).square()
                    - std.log() - 0.5 * LOG2PI.to(act.device)).sum(-1)
            ro.act[t], ro.logp[t], ro.val[t] = act, logp, val
            self.obs, rew, done, info = env.step(act)
            self.priv = info["priv"]
            ro.rew[t] = rew
            ro.done[t] = done
            ro.trunc[t] = info["truncated"]
            ro.clear[t] = info["clearance"]
            ro.collision[t] = info["collision"]
            ro.in_attempt[t] = info["in_attempt"]
            ro.attempt_id[t] = info["attempt_id"]
            ro.end_event[t] = info["end_event"]
            ro.end_outcome[t] = info["end_outcome"]
            wipe = done
            if cfg.model.reset_between_attempts:
                wipe = wipe | info["end_event"]
            self.pending = wipe
            self.h = h_new
            if "records" in info:
                self.stats.add(info["records"])
        self.h = self.h * (~self.pending).unsqueeze(-1).float()
        _, _, last_val, _ = self.model.step(self.obs["img"], self.obs["vec"], self.priv, self.h)
        compute_gae(ro, last_val, cfg.ppo.gamma, cfg.ppo.gae_lambda)

    def curriculum(self):
        cc = self.cfg.curriculum
        if not cc.enabled or self.stats.n_feas == 0:
            return
        rate = self.stats.succ_feas / self.stats.n_feas
        self.succ_ema = cc.ema * self.succ_ema + (1 - cc.ema) * rate \
            if self.iter > 0 else rate
        lam = self.env.difficulty
        if self.succ_ema > cc.up_thresh:
            lam = min(1.0, lam + cc.step_up)
        elif self.succ_ema < cc.dn_thresh:
            lam = max(0.0, lam - cc.step_dn)
        self.env.difficulty = lam

    def log(self, row):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.csv_keys])
        if self.tb:
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, row["steps"])

    def save(self, tag):
        torch.save({
            "model": self.model.state_dict(),
            "opt": self.ppo.opt.state_dict(),
            "steps": self.ppo.steps_done,
            "iter": self.iter,
            "difficulty": self.env.difficulty,
            "succ_ema": self.succ_ema,
            "cfg": self.cfg.to_dict(),
        }, os.path.join(self.dir, f"ckpt_{tag}.pt"))

    def load(self, path):
        ck = torch.load(path, map_location=self.dev)
        self.model.load_state_dict(ck["model"])
        self.ppo.opt.load_state_dict(ck["opt"])
        self.ppo.steps_done = ck["steps"]
        self.iter = ck["iter"]
        self.env.difficulty = ck["difficulty"]
        self.succ_ema = ck.get("succ_ema", 0.0)

    def run(self):
        cfg = self.cfg
        batch = cfg.ppo.rollout_len * cfg.sim.n_envs
        t0 = time.time()
        while self.ppo.steps_done < cfg.ppo.total_steps:
            it_t = time.time()
            self.collect()
            self.ppo.steps_done += batch
            up = self.ppo.update(self.ro)
            self.curriculum()
            self.iter += 1
            if self.iter % cfg.ppo.log_every == 0:
                row = {"iter": self.iter, "steps": self.ppo.steps_done,
                       "time": round(time.time() - t0, 1),
                       "sps": int(batch / max(time.time() - it_t, 1e-9)),
                       "difficulty": round(self.env.difficulty, 4),
                       "succ_ema": round(self.succ_ema, 4),
                       "rew_mean": self.ro.rew.mean().item(),
                       "log_std": self.model.log_std.mean().item()}
                row.update({k: round(v, 5) for k, v in up.items()})
                row.update({k: round(v, 5) if isinstance(v, float) else v
                            for k, v in self.stats.summary().items()})
                self.log(row)
                print({k: row[k] for k in ("iter", "steps", "sps", "difficulty",
                                           "succ_ema", "rew_mean") if k in row},
                      {k: round(row[k], 3) for k in ("ep/success", "ep/coll_high",
                                                     "ep/n_attempts", "ep/gave_up")
                       if k in row}, flush=True)
                self.stats.reset()
            if self.iter % cfg.ppo.ckpt_every == 0:
                self.save("latest")
        self.save("final")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.total_steps:
        cfg.ppo.total_steps = args.total_steps
    tr = Trainer(cfg, os.path.join("runs", args.run), args.device)
    if args.resume:
        tr.load(args.resume)
    tr.run()


if __name__ == "__main__":
    main()
