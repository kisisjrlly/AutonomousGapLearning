import json

import numpy as np

from agl.data.attempt_dataset import AttemptDataset, save_attempt_dataset
from agl.data.generate_attempt_dataset import generate
from agl.attempt.spec import SPEC_FIELDS
from agl.attempt.outcome import TARGET_FIELDS


def test_attempt_dataset_hides_audit_and_splits_by_task(tmp_path):
    path = tmp_path / "synthetic.npz"
    task_id = np.repeat(np.arange(6), 2)
    attempt_index = np.tile(np.arange(2), 6)
    spec = np.zeros((12, len(SPEC_FIELDS)), np.float32)
    target = np.zeros((12, len(TARGET_FIELDS)), np.float32)
    save_attempt_dataset(
        path,
        task_id=task_id,
        attempt_index=attempt_index,
        spec=spec,
        target=target,
        meta={"schema": "test"},
        audit={"audit_mass": np.ones(6, np.float32)},
    )
    ds = AttemptDataset(path)
    assert len(ds) == 12
    assert set(ds[0]) == {"spec", "target"}
    assert "task_id" not in ds[0] and "attempt_index" not in ds[0]
    assert ds.record_metadata(0) == {"task_id": 0, "attempt_index": 0}
    assert "audit_mass" in ds.audit

    splits = ds.split(seed=7)
    task_sets = {
        name: set(ds.task_id[idx].tolist()) for name, idx in splits.items()
    }
    assert task_sets["train"].isdisjoint(task_sets["val"])
    assert task_sets["train"].isdisjoint(task_sets["test"])
    assert task_sets["val"].isdisjoint(task_sets["test"])


def test_small_generated_dataset_is_grouped_and_readable(tmp_path):
    path = tmp_path / "attempts.npz"
    result = generate(
        path,
        num_tasks=3,
        attempts_per_task=2,
        batch_tasks=3,
        seed=4,
        device="cpu",
        executor_kwargs={
            "settle_steps": 2,
            "max_approach_steps": 25,
            "max_retreat_steps": 35,
        },
    )
    assert result["num_records"] == 6
    ds = AttemptDataset(path)
    assert ds.meta["schema"] == "attempt_dataset_v0"
    assert ds.meta["split_rule"].startswith("Split by task_id")
    assert np.array_equal(np.bincount(ds.task_id), np.array([2, 2, 2]))
    assert set(ds.audit) >= {"audit_mass", "audit_wind_steady", "audit_probe_wind"}
