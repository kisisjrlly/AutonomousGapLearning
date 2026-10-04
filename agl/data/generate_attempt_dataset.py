"""Generate grouped GapEnv attempts for contextual world-model training.

Task-hidden simulator parameters are stored under the audit_ prefix.
AttemptDataset.__getitem__ never exposes those arrays to the model.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from ..attempt.executor import execute_attempt_batch
from ..attempt.spec import AttemptRanges, SPEC_FIELDS, sample_specs
from ..attempt.outcome import TARGET_FIELDS
from ..config import load_config
from ..sim import scene
from ..sim.env import GapEnv
from .attempt_dataset import save_attempt_dataset


def _repeat_tree(x, repeats):
    if isinstance(x, dict):
        return {k: _repeat_tree(v, repeats) for k, v in x.items()}
    if torch.is_tensor(x):
        return x.repeat_interleave(repeats, dim=0)
    return copy.deepcopy(x)


def _match_runtime_biases_by_task(env, attempts_per_task):
    m = attempts_per_task
    for start in range(0, env.n, m):
        stop = start + m
        env.v_bias[start:stop].copy_(env.v_bias[start].unsqueeze(0).expand(m, -1))
        env.z_bias[start:stop].copy_(env.z_bias[start].expand(m))


def _audit_task_bank(base):
    dyn = base["dyn"]
    return {
        "audit_gap_w": base["gap_w"].detach().cpu().numpy(),
        "audit_gap_h": base["gap_h"].detach().cpu().numpy(),
        "audit_gap_roll": base["gap_roll"].detach().cpu().numpy(),
        "audit_wall_x": base["wall_x"].detach().cpu().numpy(),
        "audit_mass": dyn["mass"].detach().cpu().numpy(),
        "audit_twr": (dyn["tmax"] / (dyn["mass"] * 9.81)).detach().cpu().numpy(),
        "audit_delay": dyn["delay"].detach().cpu().numpy(),
        "audit_wind_steady": dyn["wind_steady"].detach().cpu().numpy(),
        "audit_probe_wind": dyn["probe_wind"].detach().cpu().numpy(),
    }


@torch.no_grad()
def generate(
    out,
    *,
    num_tasks=64,
    attempts_per_task=8,
    batch_tasks=8,
    seed=0,
    device="cpu",
    difficulty=1.0,
    disable_gust=True,
    ranges=None,
    executor_kwargs=None,
):
    if min(num_tasks, attempts_per_task, batch_tasks) <= 0:
        raise ValueError("task/attempt counts must be positive")
    if batch_tasks > num_tasks:
        batch_tasks = num_tasks

    out = Path(out)
    if out.exists():
        raise FileExistsError(out)

    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    task_gen = torch.Generator(device=device)
    task_gen.manual_seed(seed + 1009)

    all_task_id = []
    all_attempt_index = []
    all_spec = []
    all_target = []
    audit_accum = {}
    next_task_id = 0

    for task_start in range(0, num_tasks, batch_tasks):
        bt = min(batch_tasks, num_tasks - task_start)
        cfg = copy.deepcopy(load_config())
        cfg.sim.device = device
        cfg.sim.n_envs = bt * attempts_per_task
        cfg.curriculum.enabled = False
        cfg.task.info_gate_enabled = True
        cfg.sim.ep_len = max(cfg.sim.ep_len, 800)

        base = scene.sample_tasks(bt, cfg, difficulty, device, task_gen)
        repeated = _repeat_tree(base, attempts_per_task)
        if disable_gust:
            repeated["dyn"]["gust_sigma"].zero_()

        env = GapEnv(cfg, device, difficulty=difficulty)
        env._reset_envs(
            torch.arange(env.n, device=env.dev),
            tasks=repeated,
        )
        _match_runtime_biases_by_task(env, attempts_per_task)

        specs = sample_specs(env.n, np_rng, ranges or AttemptRanges())
        outcomes = execute_attempt_batch(
            env, specs, **(executor_kwargs or {})
        )

        task_ids = np.repeat(
            np.arange(next_task_id, next_task_id + bt, dtype=np.int64),
            attempts_per_task,
        )
        attempt_indices = np.tile(
            np.arange(attempts_per_task, dtype=np.int64), bt
        )
        all_task_id.append(task_ids)
        all_attempt_index.append(attempt_indices)
        all_spec.append(np.stack([s.as_vector() for s in specs]))
        all_target.append(np.stack([o.as_target_vector() for o in outcomes]))

        audit = _audit_task_bank(base)
        for key, value in audit.items():
            audit_accum.setdefault(key, []).append(value)
        next_task_id += bt

    arrays = {
        "task_id": np.concatenate(all_task_id),
        "attempt_index": np.concatenate(all_attempt_index),
        "spec": np.concatenate(all_spec),
        "target": np.concatenate(all_target),
    }
    audit = {
        key: np.concatenate(chunks, axis=0)
        for key, chunks in audit_accum.items()
    }
    meta = {
        "schema": "attempt_dataset_v0",
        "seed": int(seed),
        "num_tasks": int(num_tasks),
        "attempts_per_task": int(attempts_per_task),
        "difficulty": float(difficulty),
        "device_used_for_generation": str(device),
        "gust_disabled": bool(disable_gust),
        "spec_fields": list(SPEC_FIELDS),
        "target_fields": list(TARGET_FIELDS),
        "model_input_rule": (
            "Only task_id/attempt_index/spec/target are model-eligible. "
            "audit_* arrays are simulator ground truth for analysis only."
        ),
        "split_rule": "Split by task_id; never split attempts from one task across train/test.",
    }
    save_attempt_dataset(out, meta=meta, audit=audit, **arrays)
    return {
        "out": str(out),
        "num_records": int(len(arrays["task_id"])),
        "num_tasks": int(num_tasks),
        "attempts_per_task": int(attempts_per_task),
        "success_rate": float(arrays["target"][:, TARGET_FIELDS.index("success")].mean()),
        "recovered_rate": float(arrays["target"][:, TARGET_FIELDS.index("recovered")].mean()),
        "contact_rate": float(arrays["target"][:, TARGET_FIELDS.index("contact")].mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", type=int, default=64)
    p.add_argument("--attempts-per-task", type=int, default=8)
    p.add_argument("--batch-tasks", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--keep-gust", action="store_true")
    a = p.parse_args()
    result = generate(
        a.out,
        num_tasks=a.tasks,
        attempts_per_task=a.attempts_per_task,
        batch_tasks=a.batch_tasks,
        seed=a.seed,
        device=a.device,
        disable_gust=not a.keep_gust,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
