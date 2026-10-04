# Contextual World Model Plan

> Status: **new default research route (2026-10-04)**.
> Legacy recurrent PPO remains a baseline only. This document defines the next
> implementation path under the actual resource constraint: one 16 GB GPU,
> several months, real drone and test site available.

## 1. Final capability

The target is not "retry the same maneuver until it works". The target is:

1. observe an unseen gap and current vehicle condition;
2. use a learned predictive model to score several candidate traversal attempts;
3. execute one candidate only if an independent recovery/safety layer permits it;
4. if the attempt becomes unpromising, abort before contact and recover behind
   the retry plane;
5. add the *actual observed outcome* to context;
6. predict unexecuted candidates again;
7. autonomously change start point, entry direction, speed and acceleration
   profile for the next attempt;
8. pass safely, or give up if no admissible candidate remains.

The scientific causal chain is:

```text
real experience
  -> world-model prediction changes
  -> prediction of unexecuted candidates becomes more accurate
  -> next attempt changes
  -> real task performance improves
```

A change of action alone is not sufficient evidence.

## 2. First attempt space

V0 deliberately uses only six dimensions:

```text
AttemptSpec =
  start_y_offset
  start_z_offset
  entry_yaw
  entry_speed
  accel_early
  accel_late
```

The start offsets are relative to the current gap center on the safe side of the
retry plane. The acceleration values shape the forward-speed command before and
after the information-zone split.

Do **not** add splines, dense waypoints or raw motor trajectories until the
six-dimensional version is validated.

## 3. Attempt outcome

The simulation data generator records:

```text
success
recovered
contact
aborted
lateral_drift
vertical_drift
min_clearance
stopping_distance
max_tilt
terminal_speed
```

plus audit-only values such as the abort position and number of control steps.

V0 context will use previous AttemptSpec/outcome pairs. V1 will replace or
augment that summary with real sensor sequences (grayscale camera, IMU, VIO and
executed action).

## 4. World-model query

The learned predictor will eventually implement:

```math
\hat{Y}=W_\theta(C,\xi)
```

where:

- `C` is the history of **previous attempts from the same physical task**;
- `\xi` is an unexecuted candidate AttemptSpec;
- `\hat{Y}` predicts candidate outcome and uncertainty.

The first model should be small (single-GPU friendly): a compact attempt encoder,
2-4 layer context transformer/GRU and MLP outcome head. Do not train a video
generator.

## 5. Dataset rules

The dataset is organized by physical task:

```text
Task 0
  Attempt 0
  Attempt 1
  ...
Task 1
  Attempt 0
  Attempt 1
  ...
```

A task fixes geometry, hidden dynamics and persistent sensor properties.
Different attempts explore different AttemptSpec values.

Hard rules:

- train/validation/test split **by task_id**, never by attempt;
- hidden simulator parameters use the `audit_` prefix and are forbidden as
  model inputs;
- query-attempt outcome must never appear in its context;
- auto-reset data from a new task must never be appended to the old task;
- real-flight data must record requested action and actually executed action
  separately when a safety layer modifies/rejects the request.

## 6. Current implementation milestone

Implemented foundation:

- `agl/attempt/spec.py`: AttemptSpec + sampling ranges;
- `agl/attempt/outcome.py`: standard AttemptOutcome target schema;
- `agl/attempt/executor.py`: privileged vectorized GapEnv executor for honest
  candidate rollouts;
- `agl/data/attempt_dataset.py`: leakage-resistant NPZ storage and task-level
  splits;
- `agl/data/generate_attempt_dataset.py`: grouped candidate dataset generator.

The executor is **simulation data-generation infrastructure**, not the final
planner, not the learned policy and not a real-flight safety guarantee.

## 7. Immediate validation

Run:

```bash
PY=/home/zhaoguodong/miniconda3/bin/python3

$PY -m pytest tests/test_attempt_schema.py tests/test_attempt_dataset.py -q

mkdir -p datasets
$PY -m agl.data.generate_attempt_dataset \
  --out datasets/attempt_v0_smoke.npz \
  --tasks 16 \
  --attempts-per-task 8 \
  --batch-tasks 8 \
  --device cpu
```

Inspect that:

- every task has exactly the requested number of attempts;
- spec and target fields are finite;
- train/val/test task IDs do not overlap;
- `audit_*` values are not returned by AttemptDataset samples;
- outcome distribution contains useful diversity (not 100% success or 100%
  contact/abort).

If outcome diversity is poor, adjust AttemptRanges/executor before training a
world model.

## 8. Next commit: prediction benchmark

Only after the dataset foundation is verified:

1. add `NoContextWorldModel`;
2. add `ContextWorldModel`;
3. train both on identical task-level splits;
4. evaluate on *unexecuted candidate attempts*;
5. compare No Context / Correct Context / Swapped Context / Shuffled Context.

Go criterion for the project (internal engineering threshold, not a publication
claim): Correct Context should produce a clear held-out prediction improvement
over No Context without relying on hidden audit metadata.

If it does not, stop before adding a planner.

## 9. Planner comes later

Only after prediction gain is established, add a CEM attempt planner:

```text
context
  -> world model
  -> batch candidate predictions
  -> score success/clearance/recovery/cost
  -> independent safety shield
  -> execute one candidate
  -> append real outcome to context
```

The planner is not a learned network in V0.

## 10. Legacy code status

Keep, but do not extend as the default route:

- `agl/models/policy.py`
- `agl/train/ppo.py`
- `agl/train/train.py`
- old 300M-step campaigns

They provide recurrent-policy baselines for the final paper.

Keep and reuse as infrastructure:

- GapEnv and task randomization;
- substep collision checking;
- snapshot/restore;
- dynamic braking/recovery baselines;
- Rerun visualization.

## 11. Real-flight sequence

Do not start with narrow-gap learning.

1. open-space payload/wind/delay experiments;
2. verify that context improves prediction of the *next unexecuted motion*;
3. wide soft gate;
4. progressively narrower gaps;
5. full autonomous try -> abort -> recover -> replan -> retry.

The final claim must distinguish empirical zero-contact results from formal
safety guarantees.
