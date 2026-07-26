"""Deployment-cost probe: single-stream policy inference latency and memory.

Measures the deployed decision path only (image+state -> GRU -> action):
batch=1 sequential steps, fp32, on GPU and CPU. Reported in the paper as the
onboard-compute feasibility bound (README §12 last item).
"""
import argparse
import time

import torch

from ..config import load_config
from ..models.policy import Policy


def bench(model, device, steps=500):
    cfg = model.cfg
    img = torch.rand(1, 3, cfg.sensor.img_h, cfg.sensor.img_w, device=device)
    vec = torch.rand(1, 18, device=device)
    priv = torch.rand(1, 21, device=device)
    h = model.init_hidden(1, device)
    for _ in range(50):
        _, _, _, h = model.step(img, vec, priv, h)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        _, _, _, h = model.step(img, vec, priv, h)
        if device == "cuda":
            torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    return dt * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        cfg = load_config(overrides=ck["cfg"])
        model = Policy(cfg)
        model.load_state_dict(ck["model"])
    else:
        cfg = load_config()
        model = Policy(cfg)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    # deployment path excludes critic/aux/priv branches
    n_deploy = sum(p.numel() for n, p in model.named_parameters()
                   if not any(k in n for k in ("critic", "priv_fc", "aux")))
    print(f"params total={n_params/1e6:.2f}M deploy-path={n_deploy/1e6:.2f}M")
    for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        m = model.to(dev)
        ms = bench(m, dev)
        print(f"{dev}: {ms:.3f} ms/step  ({1000/ms:.0f} Hz max, control needs 40 Hz)")


if __name__ == "__main__":
    main()
