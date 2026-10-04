"""Leakage-resistant storage for attempt-level world-model data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..attempt.spec import SPEC_FIELDS
from ..attempt.outcome import TARGET_FIELDS


REQUIRED_KEYS = ("task_id", "attempt_index", "spec", "target")


def split_task_ids(task_ids, *, train_frac=.8, val_frac=.1, seed=0):
    """Split by physical task, never by individual attempts."""
    task_ids = np.asarray(task_ids, dtype=np.int64)
    unique = np.unique(task_ids)
    if unique.size < 3:
        raise ValueError("at least 3 tasks are required for train/val/test splitting")
    if not (0 < train_frac < 1 and 0 <= val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("invalid split fractions")
    rng = np.random.default_rng(seed)
    ids = unique.copy()
    rng.shuffle(ids)
    n_train = max(1, int(round(len(ids) * train_frac)))
    n_val = max(1, int(round(len(ids) * val_frac)))
    if n_train + n_val >= len(ids):
        n_train = max(1, len(ids) - 2)
        n_val = 1
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


class AttemptDataset:
    """Thin NPZ reader exposing only model-eligible arrays by default."""

    def __init__(self, path):
        self.path = Path(path)
        d = np.load(self.path, allow_pickle=False)
        missing = [k for k in REQUIRED_KEYS if k not in d]
        if missing:
            d.close()
            raise KeyError(f"missing dataset arrays: {missing}")
        self.task_id = np.asarray(d["task_id"], dtype=np.int64)
        self.attempt_index = np.asarray(d["attempt_index"], dtype=np.int64)
        self.spec = np.asarray(d["spec"], dtype=np.float32)
        self.target = np.asarray(d["target"], dtype=np.float32)
        self.meta = json.loads(str(d["meta"].item())) if "meta" in d else {}
        self._audit = {
            k: np.asarray(d[k])
            for k in d.files if k.startswith("audit_")
        }
        d.close()

        n = len(self.task_id)
        if not (len(self.attempt_index) == len(self.spec) == len(self.target) == n):
            raise ValueError("dataset arrays have inconsistent lengths")
        if self.spec.shape[1:] != (len(SPEC_FIELDS),):
            raise ValueError("spec dimension does not match SPEC_FIELDS")
        if self.target.shape[1:] != (len(TARGET_FIELDS),):
            raise ValueError("target dimension does not match TARGET_FIELDS")

    def __len__(self):
        return len(self.task_id)

    def __getitem__(self, index):
        """Return only values eligible for model training."""
        return {
            "spec": self.spec[index],
            "target": self.target[index],
        }

    def record_metadata(self, index):
        """Grouping metadata; never feed these IDs into the world model."""
        return {
            "task_id": int(self.task_id[index]),
            "attempt_index": int(self.attempt_index[index]),
        }

    @property
    def audit(self):
        """Hidden simulator metadata for analysis only; never returned by __getitem__."""
        return self._audit

    def task_indices(self, task_id):
        return np.flatnonzero(self.task_id == int(task_id))

    def split(self, *, train_frac=.8, val_frac=.1, seed=0):
        ids = split_task_ids(
            self.task_id, train_frac=train_frac, val_frac=val_frac, seed=seed
        )
        return {
            name: np.flatnonzero(np.isin(self.task_id, selected))
            for name, selected in ids.items()
        }


def save_attempt_dataset(path, *, task_id, attempt_index, spec, target,
                         meta, audit=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "task_id": np.asarray(task_id, dtype=np.int64),
        "attempt_index": np.asarray(attempt_index, dtype=np.int64),
        "spec": np.asarray(spec, dtype=np.float32),
        "target": np.asarray(target, dtype=np.float32),
        "meta": json.dumps(meta),
    }
    for key, value in (audit or {}).items():
        if not key.startswith("audit_"):
            raise ValueError("audit arrays must use the audit_ prefix")
        arrays[key] = np.asarray(value)
    np.savez_compressed(path, **arrays)
    return path
