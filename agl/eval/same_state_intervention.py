"""Strict same-environment branch preparation for future history interventions.

This module does NOT inject policy history yet. Its only job is to prove that
future correct/removed/swapped policy-history branches can start from exactly
the same complete GapEnv recovery snapshot and receive the same first
observation/noise realization.

Input must be an accepted recovery.pt produced by closed_loop_probe.py. Raw
physical-state dictionaries are intentionally rejected because they omit sensor
biases, action/image latency queues, episode bookkeeping, frames and RNG state.
"""
import argparse
import copy
import json
from pathlib import Path

import torch

from ..sim.env import GapEnv


BRANCHES = ("correct", "removed", "swapped")


def _tree_equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_tree_equal(a[k], b[k]) for k in a)
    return a == b


@torch.no_grad()
def prepare(recovery_path, device="cpu"):
    """Validate strict branch identity from a complete accepted recovery file."""
    recovery_path = Path(recovery_path)
    saved = torch.load(recovery_path, map_location=device, weights_only=False)
    if "env" not in saved or "config" not in saved:
        raise ValueError("recovery file must contain complete env snapshot and config")
    acceptance = saved.get("acceptance", {})
    if not acceptance.get("protocol_completed", False):
        raise ValueError("recovery snapshot was not accepted by the braking protocol")

    cfg = copy.deepcopy(saved["config"])
    if str(device) != "cpu":
        # Current accepted recovery baseline is CPU; moving every nested snapshot
        # tensor across devices silently would weaken reproducibility semantics.
        raise ValueError("same-state branch preparation currently requires device='cpu'")
    cfg.sim.device = "cpu"
    snapshot = saved["env"]
    required = {"frame_delay_buf", "frame_buf_ptr", "delay_buf", "rng_cpu", "v_bias", "z_bias"}
    missing = sorted(required.difference(snapshot))
    if missing:
        raise ValueError(
            "recovery snapshot predates strict same-state requirements; regenerate it. "
            f"missing={missing}"
        )
    env = GapEnv(cfg, "cpu", difficulty=snapshot.get("difficulty", 1.0))

    branch_obs = {}
    branch_snaps = {}
    for name in BRANCHES:
        env.restore(snapshot)
        # observe() consumes sensor noise RNG. Restoring before every branch is
        # therefore essential: each branch gets the same noise realization.
        obs = env.observe()
        branch_obs[name] = {
            "img": obs["img"].clone(),
            "vec": obs["vec"].clone(),
        }
        branch_snaps[name] = env.snapshot()

    reference = branch_obs["correct"]
    obs_equal = all(
        torch.equal(reference["img"], branch_obs[name]["img"])
        and torch.equal(reference["vec"], branch_obs[name]["vec"])
        for name in BRANCHES[1:]
    )
    env_equal = all(
        _tree_equal(branch_snaps["correct"]["task"], branch_snaps[name]["task"])
        and _tree_equal(branch_snaps["correct"]["state"], branch_snaps[name]["state"])
        and torch.equal(
            branch_snaps["correct"]["frame"], branch_snaps[name]["frame"]
        )
        and torch.equal(
            branch_snaps["correct"]["frame_delay_buf"],
            branch_snaps[name]["frame_delay_buf"],
        )
        and torch.equal(
            branch_snaps["correct"]["delay_buf"], branch_snaps[name]["delay_buf"]
        )
        for name in BRANCHES[1:]
    )
    if not obs_equal or not env_equal:
        raise RuntimeError("same-state branch identity check failed")

    return {
        "scope": "same_complete_snapshot_branch_preparation_NOT_history_intervention",
        "source": str(recovery_path),
        "branches": BRANCHES,
        "initial_observation_equal": obs_equal,
        "complete_environment_equal": env_equal,
        "snapshot_keys": sorted(snapshot.keys()),
        "observation_shapes": {
            "img": tuple(reference["img"].shape),
            "vec": tuple(reference["vec"].shape),
        },
        # Store one canonical pre-observation snapshot plus identical observed
        # values. History tensors will be added by a later policy-specific tool.
        "snapshot": copy.deepcopy(snapshot),
        "initial_observation": {
            "img": reference["img"].cpu(),
            "vec": reference["vec"].cpu(),
        },
        "acceptance": acceptance,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recovery", required=True)
    p.add_argument("--out", required=True,
                   help="output prefix; writes <out>.pt and <out>.json")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    pt = Path(a.out + ".pt")
    js = Path(a.out + ".json")
    if pt.exists() or js.exists():
        raise FileExistsError("same-state outputs already exist")

    result = prepare(a.recovery, a.device)
    with pt.open("xb") as h:
        torch.save(
            {
                "snapshot": result["snapshot"],
                "initial_observation": result["initial_observation"],
                "branches": result["branches"],
                "acceptance": result["acceptance"],
            },
            h,
        )
    serial = {
        k: v for k, v in result.items()
        if k not in ("snapshot", "initial_observation", "acceptance")
    }
    serial["acceptance_scope"] = result["acceptance"].get("scope")
    with js.open("x") as h:
        json.dump(serial, h, indent=2)
    print(json.dumps(serial, indent=2))


if __name__ == "__main__":
    main()
