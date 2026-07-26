"""Recurrent sensorimotor policy: CNN + GRU trunk, Gaussian actor, asymmetric
critic (privileged input, training only), auxiliary prediction heads (training
only). Deployment path is strictly obs -> GRU -> action."""
import torch
import torch.nn as nn

from ..sim.env import STATE_DIM, PRIV_DIM


def _ortho(m, gain=2 ** 0.5):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.model
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ELU(), nn.Flatten())
        with torch.no_grad():
            nf = self.conv(torch.zeros(1, 3, cfg.sensor.img_h, cfg.sensor.img_w)).shape[1]
        self.img_fc = nn.Sequential(nn.Linear(nf, c.img_feat), nn.ELU())
        self.vec_fc = nn.Sequential(nn.Linear(STATE_DIM, c.state_feat), nn.ELU(),
                                    nn.Linear(c.state_feat, c.state_feat), nn.ELU())
        self.merge = nn.Sequential(nn.Linear(c.img_feat + c.state_feat, c.merge_feat), nn.ELU())
        self.apply(_ortho)

    def forward(self, img, vec):
        z = torch.cat([self.img_fc(self.conv(img)), self.vec_fc(vec)], dim=-1)
        return self.merge(z)


class Policy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.model
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.hidden = c.gru_hidden
        self.rnn = nn.GRUCell(c.merge_feat, c.gru_hidden) if c.use_memory else None
        trunk_in = c.gru_hidden
        if not c.use_memory:
            self.ff = nn.Sequential(nn.Linear(c.merge_feat, c.gru_hidden), nn.ELU())
        self.actor = nn.Sequential(nn.Linear(trunk_in, 256), nn.ELU(), nn.Linear(256, 4))
        self.log_std = nn.Parameter(torch.full((4,), c.init_log_std))
        self.priv_fc = nn.Sequential(nn.Linear(PRIV_DIM, c.priv_feat), nn.ELU())
        self.critic = nn.Sequential(nn.Linear(trunk_in + c.priv_feat, 256), nn.ELU(),
                                    nn.Linear(256, 1))
        self.aux = nn.Sequential(nn.Linear(trunk_in, 64), nn.ELU(), nn.Linear(64, 3))
        self.actor.apply(_ortho)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        self.priv_fc.apply(_ortho)
        self.critic.apply(_ortho)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)
        self.aux.apply(_ortho)

    def init_hidden(self, n, device):
        return torch.zeros(n, self.hidden, device=device)

    def core(self, img, vec, h):
        feat = self.encoder(img, vec)
        if self.rnn is not None:
            h_new = self.rnn(feat, h)
            return h_new, h_new
        out = self.ff(feat)
        return out, h

    @torch.no_grad()
    def step(self, img, vec, priv, h):
        """Rollout step: returns action-mean, std, value, new hidden."""
        out, h_new = self.core(img, vec, h)
        mean = self.actor(out)
        value = self.critic(torch.cat([out, self.priv_fc(priv)], dim=-1)).squeeze(-1)
        return mean, self.log_std.exp(), value, h_new

    def seq_forward(self, imgs, vecs, privs, hresets, h0):
        """BPTT over a chunk. imgs: (L,B,3,H,W), hresets: (L,B) bool, h0: (B,Hid).
        Returns mean (L,B,4), value (L,B), aux (L,B,3)."""
        h = h0
        means, values, auxs = [], [], []
        for t in range(imgs.shape[0]):
            h = h * (~hresets[t]).unsqueeze(-1).float()
            out, h = self.core(imgs[t], vecs[t], h)
            means.append(self.actor(out))
            values.append(self.critic(torch.cat([out, self.priv_fc(privs[t])], dim=-1)).squeeze(-1))
            auxs.append(self.aux(out))
        return torch.stack(means), torch.stack(values), torch.stack(auxs)


def dist_logp_ent(mean, log_std, action):
    std = log_std.exp()
    var = std.square()
    logp = (-0.5 * ((action - mean).square() / var)
            - log_std - 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(-1)
    ent = (log_std + 0.5 * (1.0 + torch.log(torch.tensor(2 * torch.pi)))).sum()
    return logp, ent
