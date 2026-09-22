import numpy as np
import pytest
import torch

from agl.eval.closed_loop_probe import run, ready, feedback
from agl.sim.env import GapEnv
from agl.viz.episode import load_eval_episode


def test_feedback_recovery_and_snapshot_replay(tmp_path):
    dest = tmp_path/'accepted'
    result = run(n_pairs=2, out_dir=dest)
    assert result['protocol_completed']
    assert result['min_clearance'] > .15
    assert all(s['speed_max'] < .08 for s in result['stages'])
    saved = torch.load(dest/'recovery.pt', weights_only=False)
    env = GapEnv(saved['config'], 'cpu', difficulty=1.)
    env.restore(saved['env'])
    target = env.state['p'].clone()
    assert ready(env.state, target).all()
    snap = env.snapshot(); action = feedback(env, target)
    first = env.step(action)
    p = env.state['p'].clone()
    env.restore(snap)
    second = env.step(action)
    assert torch.equal(p, env.state['p'])
    assert torch.equal(first[0]['img'], second[0]['img'])
    ep = load_eval_episode(dest/'trajectory.npz',0)
    assert set(ep.rec['phase']) == {'approach','stop','retreat'}
    assert ep.probe_activation(0) == 0
    assert max(ep.probe_activation(i) for i in range(ep.steps)) > 0
    assert np.all(ep.rec['clear_pre'] > .15)


def test_timeout_cannot_save_recovery(tmp_path):
    dest = tmp_path/'rejected'
    out = run(n_pairs=1,phase_limit=1,out_dir=dest)
    assert not out['protocol_completed']
    assert out['failure_reason'] == 'phase_timeout'
    assert not (dest/'recovery.pt').exists()
    with pytest.raises(FileExistsError):
        run(out_dir=dest)


def test_contact_rejection_does_not_save_reset_state(tmp_path, monkeypatch):
    original = GapEnv.step
    def fail(self, action):
        obs, rew, done, info = original(self, action)
        done[:] = True; info['collision'][:] = True
        self._reset_envs(torch.arange(self.n))
        return obs, rew, done, info
    monkeypatch.setattr(GapEnv, 'step', fail)
    dest = tmp_path/'contact'
    result = run(n_pairs=1, out_dir=dest)
    assert not result['protocol_completed']
    assert result['failure_reason'] == 'terminal_or_contact'
    assert not (dest/'recovery.pt').exists()
    assert load_eval_episode(dest/'trajectory.npz',0).steps == 1


def test_nonfinite_state_rejected(tmp_path, monkeypatch):
    original = GapEnv.step
    def corrupt(self, action):
        result = original(self, action)
        self.state['v'][0, 0] = float('nan')
        return result
    monkeypatch.setattr(GapEnv, 'step', corrupt)
    dest = tmp_path/'nonfinite'
    assert run(n_pairs=1, out_dir=dest)['failure_reason'] == 'nonfinite_state'
    assert not (dest/'recovery.pt').exists()
