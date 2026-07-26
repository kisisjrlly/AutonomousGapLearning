import torch

from agl.config import load_config
from agl.models.policy import Policy
from agl.sim.env import OUTCOME, PRIV_DIM, STATE_DIM, GapEnv
from agl.train.ppo import PPO, Rollout, aux_labels, compute_gae
from agl.train.train import Trainer

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def small_cfg(n=16):
    cfg = load_config()
    cfg.sim.n_envs = n
    cfg.sim.device = DEV
    cfg.ppo.rollout_len = 32
    cfg.ppo.bptt_chunk = 16
    cfg.ppo.n_env_groups = 2
    cfg.ppo.epochs = 1
    return cfg


def hover_action(env):
    u = env.task["dyn"]["mass"] * 9.81 / env.task["dyn"]["tmax"]
    a = torch.zeros(env.n, 4, device=env.dev)
    a[:, 0] = 2 * u - 1
    return a


def test_env_shapes_and_reset():
    cfg = small_cfg()
    env = GapEnv(cfg, DEV, difficulty=0.0)
    obs = env.observe()
    assert obs["img"].shape == (16, 3, cfg.sensor.img_h, cfg.sensor.img_w)
    assert obs["vec"].shape == (16, STATE_DIM)
    obs2, rew, done, info = env.step(hover_action(env))
    assert info["priv"].shape == (16, PRIV_DIM)
    assert torch.isfinite(obs2["img"]).all() and torch.isfinite(rew).all()


def test_attempt_state_machine():
    cfg = small_cfg(4)
    env = GapEnv(cfg, DEV, difficulty=0.0)
    a = hover_action(env)
    # force wall/gap layout so hover at (x, 0, 1.5) is contact-free
    env.task["wall_x"][:] = 3.5
    env.task["gap_cy"][:] = 0.0
    env.task["gap_cz"][:] = 1.5
    env.task["gap_w"][:] = 0.6
    env.task["gap_h"][:] = 0.6
    env.task["gap_roll"][:] = 0.0
    env.state["p"][:] = torch.tensor([0.5, 0.0, 1.5], device=DEV)
    env.state["v"][:] = 0.0
    _, _, _, info = env.step(a)
    assert info["attempt_id"].max().item() == 0
    # teleport into attempt zone
    env.state["p"][:, 0] = 2.0
    env.state["v"][:] = 0.0
    _, rew, done, info = env.step(a)
    assert (info["attempt_id"] == 1).all() and info["in_attempt"].all()
    assert not done.any()
    # retreat -> abort event
    env.state["p"][:, 0] = 0.8
    env.state["v"][:] = 0.0
    _, rew, done, info = env.step(a)
    assert info["end_event"].all()
    assert (info["end_outcome"] == OUTCOME["abort"]).all()
    assert not info["in_attempt"].any()
    # success: teleport beyond success plane
    env.state["p"][:, 0] = 2.0
    env.state["v"][:] = 0.0
    env.step(a)
    env.state["p"][:, 0] = env.task["wall_x"][0] + env.task["thick"][0] + 0.6
    env.state["p"][:, 1] = 0.0
    env.state["p"][:, 2] = 1.5
    env.state["v"][:] = 0.0
    _, rew, done, info = env.step(a)
    assert done.all() and info["success"].all()
    assert rew.min().item() > 5.0
    assert "records" in info and info["records"]["success"].mean().item() == 1.0


def test_collision_terminates_negative():
    cfg = small_cfg(4)
    env = GapEnv(cfg, DEV, difficulty=0.0)
    a = hover_action(env)
    env.state["p"][:, 0] = env.task["wall_x"] + 0.02
    env.state["p"][:, 1] = 3.0   # far from any gap
    env.state["p"][:, 2] = 1.5
    env.state["v"][:] = torch.tensor([2.0, 0.0, 0.0], device=DEV)
    _, rew, done, info = env.step(a)
    assert done.all() and info["collision"].all()
    assert rew.max().item() < -5.0


def test_giveup_detection():
    cfg = small_cfg(2)
    env = GapEnv(cfg, DEV, difficulty=0.0)
    a = hover_action(env)
    env.task["wall_x"][:] = 4.0
    env.task["gap_cz"][:] = 1.5
    env.task["gap_cy"][:] = 0.0
    env.task["gap_w"][:] = 0.6
    env.task["gap_h"][:] = 0.6
    env.state["p"][:] = torch.tensor([1.5, 0.0, 1.5], device=DEV)
    env.state["v"][:] = 0.0
    env.step(a)  # registers attempt 1
    env.state["p"][:, 0] = 0.0
    env.state["v"][:] = 0.0
    done_seen = False
    for _ in range(env.giveup_steps + 20):
        _, _, done, info = env.step(a)
        env.state["v"][:] = 0.0  # pin: pure dwell test
        if done.any():
            assert info["records"]["gave_up"].mean().item() == 1.0
            done_seen = True
            break
    assert done_seen


def test_aux_label_scans():
    cfg = small_cfg(2)
    ro = Rollout(8, 2, cfg, DEV)
    # env0: collision at t=5 inside attempt 1; env1: abort at t=6
    ro.collision[5, 0] = True
    ro.done[5, 0] = True
    ro.end_event[5, 0] = True
    ro.end_outcome[5, 0] = OUTCOME["collision"]
    ro.in_attempt[2:6, 0] = True
    ro.attempt_id[2:, 0] = 1
    ro.end_event[6, 1] = True
    ro.end_outcome[6, 1] = OUTCOME["abort"]
    ro.in_attempt[1:6, 1] = True
    ro.attempt_id[1:, 1] = 1
    coll_l, coll_m, succ_l, succ_m = aux_labels(ro, horizon=4)
    assert coll_l[5, 0] == 1 and coll_l[2, 0] == 1 and coll_l[1, 0] == 0
    assert coll_m[1, 0]  # known: collision at 5 is beyond horizon 4 from t=1
    assert succ_m[3, 0] and succ_l[3, 0] == 0        # attempt ends in collision
    assert succ_m[3, 1] and succ_l[3, 1] == 0        # abort attempt: label 0
    assert not succ_m[7, 1]                           # after abort, not in attempt


def test_ppo_end_to_end_smoke():
    cfg = small_cfg(16)
    cfg.ppo.total_steps = 32 * 16 * 2
    tr = Trainer(cfg, "/tmp/claude-1000/-home-zhaoguodong-work-code-AutonomousGapLearning/6824d512-47fc-4c76-be8f-cc4b18bbc0ef/scratchpad/ppo_smoke", DEV)
    tr.run()
    assert tr.iter == 2
    assert torch.isfinite(tr.ro.rew).all()
    for pr in tr.model.parameters():
        assert torch.isfinite(pr).all()


def test_no_memory_variant_runs():
    cfg = small_cfg(8)
    cfg.model.use_memory = False
    cfg.ppo.total_steps = 32 * 8
    tr = Trainer(cfg, "/tmp/claude-1000/-home-zhaoguodong-work-code-AutonomousGapLearning/6824d512-47fc-4c76-be8f-cc4b18bbc0ef/scratchpad/ppo_smoke_nomem", DEV)
    tr.run()
    assert tr.iter == 1
