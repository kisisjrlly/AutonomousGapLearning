"""Central configuration. YAML files override dataclass defaults recursively."""
from dataclasses import dataclass, field, asdict
import yaml


@dataclass
class SimCfg:
    n_envs: int = 3072
    dt_ctrl: float = 0.025          # 40 Hz control
    substeps: int = 5               # 200 Hz physics
    ep_len: int = 960               # 24 s per task instance (multi-attempt episode)
    device: str = "cuda"
    # arena (world frame: x forward toward wall, z up)
    arena_y: float = 5.0
    arena_z: float = 4.5
    arena_x_min: float = -2.0
    arena_x_extra: float = 3.0      # arena extends this far behind the wall
    retry_x: float = 1.2            # attempt zone boundary plane
    succ_margin: float = 0.4        # success plane = wall_x + thickness + margin
    # drone collision geometry: flat cylinder (prop-guard quad)
    body_r: float = 0.16
    body_hh: float = 0.05


@dataclass
class TaskCfg:
    # gap geometry ranges (difficulty lambda in [0,1] interpolates *_easy -> value)
    width_lo: float = 0.34
    width_lo_easy: float = 0.80
    width_hi: float = 0.90
    width_hi_easy: float = 1.40
    height_lo: float = 0.30
    height_lo_easy: float = 0.60
    height_hi: float = 0.80
    height_hi_easy: float = 1.00
    gap_cz_spread_easy: float = 0.25  # easy: gap centers near start altitude
    thick_lo: float = 0.05
    thick_hi: float = 0.30
    roll_max_deg: float = 40.0      # in-plane gap roll; scaled by difficulty
    roll_max_deg_easy: float = 10.0
    wall_x_lo: float = 2.5
    wall_x_hi: float = 4.5
    wall_x_hi_easy: float = 3.0
    gap_cy: float = 0.8             # |gap center y| <= this
    gap_cz_lo: float = 1.0
    gap_cz_hi: float = 2.2
    # infeasible instances (README: policy must learn to give up)
    infeasible_frac: float = 0.08   # scaled by difficulty
    infeasible_w_lo: float = 0.20
    infeasible_w_hi: float = 0.30
    feas_margin: float = 0.03       # per-side clearance margin for feasibility label
    feas_roll_max_deg: float = 65.0 # max sustainable roll for feasibility label only
    # dynamics randomization
    mass_lo: float = 0.60
    mass_hi: float = 0.95
    twr_lo: float = 2.2             # max thrust / (m*g); hidden from policy (in-context sysid)
    twr_hi: float = 3.4
    tau_thrust_lo: float = 0.03
    tau_thrust_hi: float = 0.06
    tau_rate_lo: float = 0.03
    tau_rate_hi: float = 0.07
    alpha_max_lo: float = 25.0      # rad/s^2 achievable body angular accel
    alpha_max_hi: float = 55.0
    delay_max_steps: int = 2        # action delay 0..2 control steps (0-50 ms)
    kd_lin_lo: float = 0.05
    kd_lin_hi: float = 0.25
    kd_quad_lo: float = 0.005
    kd_quad_hi: float = 0.02
    wind_max: float = 2.5           # m/s steady, scaled by difficulty
    wind_gust_sigma: float = 0.8
    wind_tau: float = 2.0


@dataclass
class SensorCfg:
    img_w: int = 32
    img_h: int = 24
    fov_deg: float = 90.0           # horizontal
    cam_pitch_deg: float = 0.0
    img_delay_steps: int = 1        # vision latency = 1 control step
    px_noise_lo: float = 0.01
    px_noise_hi: float = 0.05
    gyro_noise: float = 0.01
    gyro_bias: float = 0.02
    acc_noise: float = 0.10
    acc_bias: float = 0.10
    gdir_noise: float = 0.02
    vel_noise: float = 0.05
    vel_bias_sigma: float = 0.15    # OU bias on VIO velocity
    vel_bias_tau: float = 10.0
    z_noise: float = 0.02
    z_bias_sigma: float = 0.10
    z_bias_tau: float = 10.0


@dataclass
class RewardCfg:
    # rewards are pre-scaled ~[-10, 10]
    success: float = 10.0
    success_time_bonus: float = 4.0     # * fraction of episode time remaining
    collision_high: float = -10.0
    collision_soft: float = -4.0
    collision_high_easy: float = -3.0   # annealed to full penalty by anneal_end
    collision_soft_easy: float = -1.5
    coll_anneal_end: float = 0.5        # difficulty at which penalty reaches full
    oob: float = -4.0
    contact_soft_ke: float = 0.25       # J threshold high/soft contact energy
    progress_k: float = 0.6             # potential: +k * forward progress toward target
    progress_asymmetric: bool = True    # no penalty for retreating (enables safe abort exploration)
    step_cost: float = 0.002
    attempt_cost: float = 0.3
    first_attempt_bonus: float = 0.5    # once per episode, counters passivity
    abort_bonus: float = 0.3            # base reward for a safe return to retry zone
    abort_depth_bonus: float = 1.2      # * depth reached in the aborted attempt (anti-farming)
    wall_prox_k: float = 0.5            # approach-region wall proximity penalty
    wall_prox_margin: float = 0.15
    wall_prox_xgate: float = 0.20       # only active at x < wall_x - gate
    boundary_k: float = 0.05            # soft arena-boundary inward shaping
    bound_y: float = 3.5
    bound_z: float = 3.5
    bound_x_back: float = -1.0
    smooth_k: float = 0.01
    giveup_hover_s: float = 4.0         # measurement: dwell in retry zone => give up
    giveup_speed: float = 0.5


@dataclass
class ModelCfg:
    img_feat: int = 128
    state_feat: int = 64
    merge_feat: int = 256
    gru_hidden: int = 512
    priv_feat: int = 64
    init_log_std: float = -0.7
    # ablation switches
    use_memory: bool = True             # False => feedforward (frame-only)
    reset_between_attempts: bool = False  # wipe hidden when re-entering retry zone
    use_prev_action: bool = True
    use_aux: bool = True


@dataclass
class PPOCfg:
    total_steps: int = 300_000_000
    rollout_len: int = 96
    bptt_chunk: int = 32
    n_env_groups: int = 8               # minibatches per epoch = groups
    epochs: int = 2
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    lr_final: float = 1e-4
    ent_coef: float = 3e-3
    ent_coef_final: float = 1e-3
    vf_coef: float = 0.5
    aux_coef: float = 0.5
    max_grad_norm: float = 1.0
    coll_horizon: int = 20              # aux label: collision within 0.5 s
    seed: int = 1
    log_every: int = 10
    ckpt_every: int = 50


@dataclass
class CurriculumCfg:
    enabled: bool = True
    start: float = 0.0
    step_up: float = 0.01
    step_dn: float = 0.01
    up_thresh: float = 0.70             # feasible-task success EMA
    dn_thresh: float = 0.40
    ema: float = 0.98


@dataclass
class Config:
    sim: SimCfg = field(default_factory=SimCfg)
    task: TaskCfg = field(default_factory=TaskCfg)
    sensor: SensorCfg = field(default_factory=SensorCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    ppo: PPOCfg = field(default_factory=PPOCfg)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
    run_name: str = "dev"

    def to_dict(self):
        return asdict(self)


def load_config(path: str | None = None, overrides: dict | None = None) -> Config:
    cfg = Config()
    data = {}
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    if overrides:
        _merge(data, overrides)
    for section, values in data.items():
        if not hasattr(cfg, section):
            raise KeyError(f"unknown config section: {section}")
        target = getattr(cfg, section)
        if isinstance(values, dict):
            for k, v in values.items():
                if not hasattr(target, k):
                    raise KeyError(f"unknown config key: {section}.{k}")
                setattr(target, k, v)
        else:
            setattr(cfg, section, values)
    return cfg


def _merge(base: dict, extra: dict):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
