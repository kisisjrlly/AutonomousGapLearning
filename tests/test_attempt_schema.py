import math

import numpy as np
import torch

from agl.attempt import AttemptOutcome, AttemptSpec
from agl.attempt.executor import execute_attempt_batch
from agl.config import load_config
from agl.sim import scene
from agl.sim.env import GapEnv


def test_attempt_spec_and_outcome_roundtrip():
    spec = AttemptSpec(.1, -.05, .08, .45, .6, -.2)
    vec = spec.as_vector()
    restored = AttemptSpec.from_vector(vec)
    assert np.allclose(restored.as_vector(), vec)

    out = AttemptOutcome(
        success=False,
        recovered=True,
        contact=False,
        aborted=True,
        lateral_drift=.12,
        vertical_drift=-.03,
        min_clearance=.08,
        stopping_distance=.11,
        max_tilt=.2,
        terminal_speed=.04,
        abort_x=2.1,
        steps=42,
    )
    target = out.as_target_vector()
    restored_out = AttemptOutcome.from_target_vector(
        target, abort_x=out.abort_x, steps=out.steps
    )
    assert np.allclose(restored_out.as_target_vector(), target)
    assert restored_out.safe_terminal


def test_attempt_executor_returns_finite_labels():
    cfg = load_config()
    cfg.sim.device = "cpu"
    cfg.sim.n_envs = 2
    cfg.curriculum.enabled = False
    cfg.task.info_gate_enabled = True
    env = GapEnv(cfg, "cpu", difficulty=1.0)
    bank = scene.sample_tasks(2, cfg, 1.0, "cpu")
    env._reset_envs(torch.arange(2), tasks=bank)

    specs = [
        AttemptSpec(0.0, 0.0, 0.0, .35, .5, 0.0),
        AttemptSpec(.08, -.04, .06, .40, .7, -.15),
    ]
    outcomes = execute_attempt_batch(
        env,
        specs,
        settle_steps=3,
        max_approach_steps=40,
        max_retreat_steps=60,
    )
    assert len(outcomes) == 2
    for out in outcomes:
        assert np.isfinite(out.as_target_vector()).all()
        assert out.steps > 0
        assert math.isfinite(out.min_clearance)
