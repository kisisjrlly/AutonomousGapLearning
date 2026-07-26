"""Recurrent PPO with truncated BPTT over multi-attempt episodes."""
import math
import torch

from ..models.policy import dist_logp_ent
from ..sim.env import OUTCOME


class Rollout:
    """GPU-resident rollout storage. Images kept as uint8."""

    def __init__(self, T, N, cfg, device):
        s = cfg.sensor
        self.T, self.N = T, N
        self.img = torch.zeros(T, N, 3, s.img_h, s.img_w, dtype=torch.uint8, device=device)
        self.vec = torch.zeros(T, N, 18, device=device)
        self.priv = torch.zeros(T, N, 21, device=device)
        self.act = torch.zeros(T, N, 4, device=device)
        self.logp = torch.zeros(T, N, device=device)
        self.val = torch.zeros(T, N, device=device)
        self.rew = torch.zeros(T, N, device=device)
        self.done = torch.zeros(T, N, dtype=torch.bool, device=device)
        self.hreset = torch.zeros(T, N, dtype=torch.bool, device=device)
        self.clear = torch.zeros(T, N, device=device)
        self.collision = torch.zeros(T, N, dtype=torch.bool, device=device)
        self.in_attempt = torch.zeros(T, N, dtype=torch.bool, device=device)
        self.attempt_id = torch.zeros(T, N, dtype=torch.long, device=device)
        self.end_event = torch.zeros(T, N, dtype=torch.bool, device=device)
        self.end_outcome = torch.zeros(T, N, dtype=torch.long, device=device)
        nchunks = T // cfg.ppo.bptt_chunk
        self.h0 = torch.zeros(nchunks, N, cfg.model.gru_hidden, device=device)
        self.adv = torch.zeros(T, N, device=device)
        self.ret = torch.zeros(T, N, device=device)


def compute_gae(ro: Rollout, last_val, gamma, lam):
    adv = torch.zeros_like(ro.rew)
    gae = torch.zeros(ro.N, device=ro.rew.device)
    for t in reversed(range(ro.T)):
        nonterm = (~ro.done[t]).float()
        v_next = last_val if t == ro.T - 1 else ro.val[t + 1]
        delta = ro.rew[t] + gamma * v_next * nonterm - ro.val[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    ro.adv = adv
    ro.ret = adv + ro.val


def aux_labels(ro: Rollout, horizon: int):
    """Backward scans: collision-within-horizon (+validity) and attempt outcome."""
    T, N, dev = ro.T, ro.N, ro.rew.device
    coll_l = torch.zeros(T, N, device=dev)
    coll_m = torch.zeros(T, N, dtype=torch.bool, device=dev)
    succ_l = torch.zeros(T, N, device=dev)
    succ_m = torch.zeros(T, N, dtype=torch.bool, device=dev)
    c = torch.zeros(N, dtype=torch.long, device=dev)          # collision-ahead counter
    vis = torch.full((N,), -1, dtype=torch.long, device=dev)  # known-future marker
    cur_out = torch.full((N,), -1, dtype=torch.long, device=dev)
    cur_aid = torch.zeros(N, dtype=torch.long, device=dev)
    H = horizon
    for t in reversed(range(T)):
        coll, done = ro.collision[t], ro.done[t]
        c = torch.where(coll, torch.full_like(c, H),
                        torch.where(done, torch.zeros_like(c), (c - 1).clamp_min(0)))
        vis = torch.where(done, torch.full_like(vis, H),
                          torch.where(vis < 0, vis, (vis + 1).clamp_max(H)))
        coll_l[t] = (c > 0).float()
        coll_m[t] = (c > 0) | (vis == H)
        ee = ro.end_event[t]
        cur_out = torch.where(ee, ro.end_outcome[t], cur_out)
        cur_aid = torch.where(ee, ro.attempt_id[t], cur_aid)
        # cutoff outcomes carry no label; episode boundary clears stale state
        cur_out = torch.where(ee & (ro.end_outcome[t] == OUTCOME["cutoff"]),
                              torch.full_like(cur_out, -1), cur_out)
        valid = ro.in_attempt[t] & (ro.attempt_id[t] == cur_aid) & (cur_out >= 0)
        succ_l[t] = (cur_out == OUTCOME["success"]).float() * valid.float()
        succ_m[t] = valid
        cur_out = torch.where(done & ~ee, torch.full_like(cur_out, -1), cur_out)
    return coll_l, coll_m, succ_l, succ_m


class PPO:
    def __init__(self, cfg, model, device):
        self.cfg = cfg
        self.model = model
        self.dev = device
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.ppo.lr, eps=1e-5)
        self.steps_done = 0

    def anneal(self):
        p = self.cfg.ppo
        f = min(1.0, self.steps_done / max(p.total_steps, 1))
        lr = p.lr + (p.lr_final - p.lr) * f
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.ent_coef = p.ent_coef + (p.ent_coef_final - p.ent_coef) * f
        return lr

    def update(self, ro: Rollout):
        p, m = self.cfg.ppo, self.cfg.model
        lr = self.anneal()
        coll_l, coll_m, succ_l, succ_m = aux_labels(ro, p.coll_horizon)
        clear_l = ro.clear.clamp(-0.2, 2.0)
        adv = ro.adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        T, N = ro.T, ro.N
        L = p.bptt_chunk
        nchunks = T // L
        group = N // p.n_env_groups
        stats = {k: 0.0 for k in ("pi_loss", "v_loss", "ent", "aux_loss", "clipfrac", "kl")}
        nmb = 0
        for _ in range(p.epochs):
            perm = torch.randperm(N, device=self.dev)
            for gidx in range(p.n_env_groups):
                envs = perm[gidx * group:(gidx + 1) * group]
                # stack chunks into batch: (L, nchunks*group, ...)
                def gather(x):
                    xs = [x[c * L:(c + 1) * L, envs] for c in range(nchunks)]
                    return torch.cat(xs, dim=1)
                imgs = gather(ro.img).float() / 255.0
                vecs, privs = gather(ro.vec), gather(ro.priv)
                hres = gather(ro.hreset)
                acts, logp_old = gather(ro.act), gather(ro.logp)
                val_old, advs, rets = gather(ro.val), gather(adv), gather(ro.ret)
                h0 = torch.cat([ro.h0[c, envs] for c in range(nchunks)], dim=0)
                mean, value, aux = self.model.seq_forward(imgs, vecs, privs, hres, h0)
                logp, ent = dist_logp_ent(mean, self.model.log_std, acts)
                ratio = (logp - logp_old).exp()
                pl = -torch.min(ratio * advs,
                                ratio.clamp(1 - p.clip, 1 + p.clip) * advs).mean()
                v_clip = val_old + (value - val_old).clamp(-p.clip, p.clip)
                vl = 0.5 * torch.max((value - rets).square(),
                                     (v_clip - rets).square()).mean()
                aux_l = torch.tensor(0.0, device=self.dev)
                if m.use_aux:
                    cl = gather(clear_l)
                    cm, cl2 = gather(coll_m).float(), gather(coll_l)
                    sm, sl = gather(succ_m).float(), gather(succ_l)
                    aux_clear = (aux[..., 0] - cl).square().mean()
                    bce = torch.nn.functional.binary_cross_entropy_with_logits
                    aux_coll = (bce(aux[..., 1], cl2, reduction="none") * cm).sum() / cm.sum().clamp_min(1)
                    aux_succ = (bce(aux[..., 2], sl, reduction="none") * sm).sum() / sm.sum().clamp_min(1)
                    aux_l = aux_clear + aux_coll + aux_succ
                loss = pl + p.vf_coef * vl - self.ent_coef * ent + p.aux_coef * aux_l
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), p.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats["pi_loss"] += pl.item()
                    stats["v_loss"] += vl.item()
                    stats["ent"] += ent.item()
                    stats["aux_loss"] += float(aux_l)
                    stats["clipfrac"] += ((ratio - 1).abs() > p.clip).float().mean().item()
                    stats["kl"] += (logp_old - logp).mean().item()
                nmb += 1
        for k in stats:
            stats[k] /= max(nmb, 1)
        stats["lr"] = lr
        return stats
