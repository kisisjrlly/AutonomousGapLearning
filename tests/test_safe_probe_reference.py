import json

import numpy as np
import pytest

from agl.eval.safe_probe_reference import Safety, Task, clearance, execute, observe, run_bank
from agl.eval.analyze_attempts import analyze_attempts
from agl.eval.verify_gru_context import analyze_results


def test_paired_reference():
    result = run_bank()
    assert result == run_bank()
    assert result['n_eligible'] == 24
    assert result['summary']['real']['successes'] == 16
    assert result['summary']['real']['abstentions'] == 8
    assert result['summary']['removed']['successes'] == 0
    assert result['summary']['swapped']['interventions'] == 16
    for row in result['tasks']:
        assert row['retreat_trace'][-1] == (0., 0.)
        assert row['retreat_ok']
        assert not row['probe']['contact']
        for branch in row['branches'].values():
            assert branch['start'] == [0., 0.]
            assert not branch['contact']
    json.dumps(result, allow_nan=False)


def test_information_is_view_gated():
    import random
    task = Task(.22, .52)
    assert observe(task, (0., 0.), random.Random(0)) is None
    measurement = observe(task, (1.4, .4), random.Random(0))
    assert abs(measurement['center'] - task.center) <= measurement['center_error']


def test_guard_rejects_contact_and_delay_risk():
    task = Task(.22, .52)
    assert clearance((0., 0.), (3., 0.), task) < Safety().radius
    rejected = execute(task, [(3., 0.)], Safety())
    assert rejected['intervention'] and rejected['trace'] == [(0., 0.)]
    path = [(0., .22), (3., .22)]
    assert execute(task, path, Safety())['success']
    assert not execute(task, path, Safety(latency=1.))['success']
    assert not execute(task, path, Safety(position_error=.1))['success']
    with pytest.raises(ValueError):
        Safety(brake_accel=0).inflation()
    with pytest.raises(ValueError):
        clearance((0., 0.), (1., 1.), task)


def test_missing_history_samples_are_unknown():
    s = analyze_results({'action_diffs': np.empty((0, 4)), 'n_tasks': 4})
    assert s['mean_diff'] is None
    assert s['conclusion'] == 'insufficient_safe_abort_samples'
    a = analyze_attempts({'episodes': [{'n_attempts': 0, 'final_success': False,
                                        'attempts': []}]})
    assert a['improvement_rate'] is None


def test_cumulative_success_uses_tasks_not_attempts():
    def attempt(k, success):
        return {'attempt_num': k, 'success': success, 'min_clearance': .2,
                'outcome': 'success' if success else 'abort'}
    stats = analyze_attempts({'episodes': [
        {'n_attempts': 1, 'final_success': True, 'attempts': [attempt(1, True)]},
        {'n_attempts': 2, 'final_success': True,
         'attempts': [attempt(1, False), attempt(2, True)]}]})
    assert stats['cumulative_success_by_attempt'] == {'1': 50., '2': 100.}


def test_gru_diagnostic_waits_for_each_safe_abort(monkeypatch):
    import torch
    import agl.eval.verify_gru_context as module
    from agl.config import load_config
    from agl.sim.env import OUTCOME

    class Env:
        def __init__(self, *args, **kwargs):
            self.t = 0

        def _reset_envs(self, *args, **kwargs):
            pass

        def observe(self):
            return {'img': torch.zeros(2, 1), 'vec': torch.zeros(2, 1)}

        def step(self, action):
            self.t += 1
            # Row 0 terminates/reset early; row 1 aborts after the old 50-step cap.
            done = torch.tensor([self.t == 1, self.t == 62])
            event = torch.tensor([self.t in (1, 3), self.t == 60])
            return self.observe(), None, done, {
                'priv': torch.zeros(2, 21), 'end_event': event,
                'end_outcome': torch.tensor([OUTCOME['abort'], OUTCOME['abort']]),
                'collision': torch.zeros(2, dtype=torch.bool)}

    class Model:
        def init_hidden(self, n, device):
            return torch.zeros(n, 1)

        def step(self, img, vec, priv, h):
            return torch.zeros(2, 4), None, None, h + 1

        def core(self, img, vec, h):
            return h + 1, h + 1

        def act_in(self, h):
            return h

        def actor(self, h):
            return h.repeat(1, 4) / 100

    monkeypatch.setattr(module, 'GapEnv', Env)
    monkeypatch.setattr(module, 'make_bank', lambda *args: {})
    cfg = load_config()
    cfg.sim.ep_len = 65
    results = module.run_verification(Model(), cfg, 2, 'cpu')
    assert results['task_ids'] == [1]
    assert results['action_diffs'].shape == (1, 4)
    assert cfg.sim.n_envs != 2  # Diagnostic must not mutate caller configuration.
