# Next Task Guide

## Current execution — 2026-10-04

**Default route:** Attempt-level Contextual World Model.

The old GRU verification / reward-shaping / recipe_v3 decision tree is retired.
Do not wait for `mean_diff`, do not restart the 300M-step PPO campaign, and do
not choose a new policy architecture from the old GRU diagnostics.

Read first:

1. `HANDOFF.md`
2. `docs/CONTEXTUAL_WORLD_MODEL_PLAN.md`
3. `docs/PIPELINE.md`

## What is implemented now

- `agl/attempt/spec.py`: 6-D AttemptSpec
- `agl/attempt/outcome.py`: standard AttemptOutcome
- `agl/attempt/executor.py`: privileged GapEnv candidate executor
- `agl/data/attempt_dataset.py`: grouped NPZ dataset + task-level split
- `agl/data/generate_attempt_dataset.py`: attempt dataset generator

These are data-generation foundations only. There is **no contextual world
model, no CEM planner and no learned retry yet**.

## Immediate local verification

```bash
cd /home/zhaoguodong/work/code/AutonomousGapLearning
git pull origin main

PY=/home/zhaoguodong/miniconda3/bin/python3

$PY -m pytest tests/ -q
$PY -m pytest tests/test_attempt_schema.py tests/test_attempt_dataset.py -q

mkdir -p datasets
$PY -m agl.data.generate_attempt_dataset \
  --out datasets/attempt_v0_smoke.npz \
  --tasks 16 \
  --attempts-per-task 8 \
  --batch-tasks 8 \
  --device cpu
```

Report:

- total pytest pass/fail/warnings;
- generated record count;
- success / recovered / contact rates;
- whether every task has exactly 8 attempts;
- whether all spec/target values are finite;
- whether train/val/test task IDs are disjoint;
- any executor timeout/reset/contact anomaly.

## Go / No-Go

**GO:** dataset is valid, task grouping is correct, outcomes are diverse enough
to train a predictor.

**NO-GO:** task leakage, non-finite labels, reset contamination, or a degenerate
outcome distribution. Fix the dataset/executor first.

## Next implementation after GO

Only then implement:

1. `NoContextWorldModel`
2. `ContextWorldModel`
3. identical task-level train/val/test split
4. prediction benchmark on unexecuted candidates
5. No / Correct / Swapped / Shuffled Context evaluation

Do **not** implement a planner before Correct Context gives a meaningful
held-out prediction improvement over No Context.

## Legacy status

Keep for final-paper baselines, but do not extend by default:

- `agl/models/policy.py`
- `agl/train/ppo.py`
- `agl/train/train.py`
- `scripts/run_campaign.sh`
- recipe_v2 / recipe_v3 experiments
