"""Vectorized multi-attempt gap-traversal environment.

One episode = one task instance = up to ep_len steps containing an open number
of attempts. Attempt segmentation (crossing the retry plane) is measurement
side only: it feeds rewards-as-events and metrics, and is never observed by
the policy. The policy sees only egocentric sensors (no global position).
"""
import torch
import copy

from . import dynamics, scene, render, collision
from .maths import quat_rotate_inv, quat_from_yaw

STATE_DIM = 18
PRIV_DIM = 21
OUTCOME = {"abort": 0, "success": 1, "collision": 2, "cutoff": 3}


class GapEnv:
    def __init__(self, cfg, device=None, difficulty=None):
        self.cfg = cfg
        self.dev = torch.device(device or cfg.sim.device)
        self.n = cfg.sim.n_envs
        self.difficulty = cfg.curriculum.start if difficulty is None else difficulty
        self.rays = render.camera_rays(cfg.sensor, self.dev)
        self.bpts = collision.body_points(cfg.sim.body_r, cfg.sim.body_hh, self.dev)
        self.giveup_steps = int(cfg.reward.giveup_hover_s / cfg.sim.dt_ctrl)
        self.task = scene.sample_tasks(self.n, cfg, self.difficulty, self.dev)
        self.state = dynamics.make_state(self.n, self.dev)
        n = self.n
        z3 = lambda: torch.zeros(n, 3, device=self.dev)
        z1 = lambda: torch.zeros(n, device=self.dev)
        self.prev_action = torch.zeros(n, 4, device=self.dev)
        self.delay_buf = torch.zeros(n, cfg.task.delay_max_steps + 1, 4, device=self.dev)
        self.buf_ptr = 0
        self.v_bias, self.z_bias = z3(), z1()
        self.t_step = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.attempts = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.in_attempt = torch.zeros(n, dtype=torch.bool, device=self.dev)
        self.attempt_depth = torch.zeros(n, device=self.dev)
        self.retry_dwell = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.ep_min_clear = torch.full((n,), 10.0, device=self.dev)
        self.prev_dist = z1()
        self.prev_x = z1()
        self.frame = torch.zeros(n, 3, cfg.sensor.img_h, cfg.sensor.img_w, device=self.dev)
        self._reset_envs(torch.arange(n, device=self.dev))

    # ------------------------------------------------------------------ reset
    def _reset_envs(self, idx, tasks=None):
        if idx.numel() == 0:
            return
        cfg, m = self.cfg, idx.numel()
        new = tasks if tasks is not None else scene.sample_tasks(m, cfg, self.difficulty, self.dev)
        scene.scatter_tasks(self.task, new, idx)
        self.state["p"][idx] = torch.stack([
            -0.3 + 0.9 * torch.rand(m, device=self.dev),
            -0.6 + 1.2 * torch.rand(m, device=self.dev),
            1.1 + 0.9 * torch.rand(m, device=self.dev)], dim=-1)
        yaw = (torch.rand(m, device=self.dev) - 0.5) * 0.52
        self.state["q"][idx] = quat_from_yaw(yaw)
        self.state["v"][idx] = 0.0
        self.state["w"][idx] = 0.0
        self.state["wind"][idx] = 0.0
        self.state["spec_force"][idx] = 0.0
        dyn_i = {k: v[idx] for k, v in self.task["dyn"].items() if torch.is_tensor(v)}
        self.state["thrust"][idx] = dyn_i["mass"] * 9.81
        hov = dynamics.hover_thrust_action(dyn_i)
        act0 = torch.zeros(m, 4, device=self.dev)
        act0[:, 0] = hov
        self.prev_action[idx] = act0
        self.delay_buf[idx] = act0.unsqueeze(1)
        self.v_bias[idx] = 0.05 * torch.randn(m, 3, device=self.dev)
        self.z_bias[idx] = 0.05 * torch.randn(m, device=self.dev)
        self.t_step[idx] = 0
        self.attempts[idx] = 0
        self.in_attempt[idx] = False
        self.attempt_depth[idx] = 0.0
        self.retry_dwell[idx] = 0
        self.ep_min_clear[idx] = 10.0
        self.prev_dist[idx] = (self.state["p"][idx] - self._target()[idx]).norm(dim=-1)
        self.prev_x[idx] = self.state["p"][idx][:, 0]
        self.frame[idx] = render.render(self.state["p"][idx], self.state["q"][idx],
                                        self._task_slice(idx), self.rays, cfg.sensor)

    def snapshot(self):
        """Capture all mutable environment state needed for deterministic replay.

        This is for offline intervention experiments. It includes observation
        biases, delay queues, episode bookkeeping, rendered frame, task tensors,
        and global torch RNG state; omitting any of these would make a later
        history comparison confounded by different observations or noise.
        """
        def clone_tree(x):
            if torch.is_tensor(x):
                return x.clone()
            if isinstance(x, dict):
                return {k: clone_tree(v) for k, v in x.items()}
            return copy.deepcopy(x)
        snap = {
            "task": clone_tree(self.task), "state": clone_tree(self.state),
            "prev_action": self.prev_action.clone(), "delay_buf": self.delay_buf.clone(),
            "buf_ptr": int(self.buf_ptr), "v_bias": self.v_bias.clone(),
            "z_bias": self.z_bias.clone(), "t_step": self.t_step.clone(),
            "attempts": self.attempts.clone(), "in_attempt": self.in_attempt.clone(),
            "attempt_depth": self.attempt_depth.clone(), "retry_dwell": self.retry_dwell.clone(),
            "ep_min_clear": self.ep_min_clear.clone(), "prev_dist": self.prev_dist.clone(),
            "prev_x": self.prev_x.clone(), "frame": self.frame.clone(),
            "rng_cpu": torch.get_rng_state(),
        }
        if self.dev.type == "cuda":
            snap["rng_cuda"] = torch.cuda.get_rng_state(self.dev)
        return snap

    def restore(self, snapshot):
        """Restore a snapshot produced by :meth:`snapshot` in-place."""
        def restore_tree(dst, src):
            if isinstance(dst, dict):
                if dst.keys() != src.keys():
                    raise ValueError("snapshot schema does not match environment")
                for k in dst:
                    restore_tree(dst[k], src[k])
            elif torch.is_tensor(dst):
                if dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
                    raise ValueError("snapshot tensor does not match environment")
                dst.copy_(src)
            else:
                if dst != src:
                    raise ValueError("snapshot scalar does not match environment")
        restore_tree(self.task, snapshot["task"])
        restore_tree(self.state, snapshot["state"])
        for name in ("prev_action", "delay_buf", "v_bias", "z_bias", "t_step",
                     "attempts", "in_attempt", "attempt_depth", "retry_dwell",
                     "ep_min_clear", "prev_dist", "prev_x", "frame"):
            getattr(self, name).copy_(snapshot[name])
        self.buf_ptr = int(snapshot["buf_ptr"])
        torch.set_rng_state(snapshot["rng_cpu"])
        if self.dev.type == "cuda":
            torch.cuda.set_rng_state(snapshot["rng_cuda"], self.dev)

    def _task_slice(self, idx):
        out = {}
        for k, v in self.task.items():
            out[k] = {k2: (v2[idx] if torch.is_tensor(v2) else v2) for k2, v2 in v.items()} \
                if isinstance(v, dict) else v[idx]
        return out

    def _target(self):
        return torch.stack([self.task["wall_x"] + self.task["thick"] + 0.35,
                            self.task["gap_cy"], self.task["gap_cz"]], dim=-1)

    # ------------------------------------------------------------------- obs
    def observe(self):
        cfg, st, sens = self.cfg, self.state, self.cfg.sensor
        n = self.n
        rn = lambda *s: torch.randn(*s, device=self.dev)
        gyro = st["w"] + self.task["sens"]["gyro_bias"] + sens.gyro_noise * rn(n, 3)
        acc = st["spec_force"] + self.task["sens"]["acc_bias"] + sens.acc_noise * rn(n, 3)
        gdir = quat_rotate_inv(st["q"], torch.tensor([0.0, 0.0, -1.0], device=self.dev).expand(n, 3))
        gdir = gdir + sens.gdir_noise * rn(n, 3)
        v_body = quat_rotate_inv(st["q"], st["v"] + self.v_bias) + sens.vel_noise * rn(n, 3)
        z_est = st["p"][:, 2] + self.z_bias + sens.z_noise * rn(n)
        u_act = st["thrust"] / self.task["dyn"]["tmax"]
        prev_a = self.prev_action if self.cfg.model.use_prev_action \
            else torch.zeros_like(self.prev_action)
        vec = torch.cat([gyro / 6.0, acc / 15.0, gdir, v_body / 5.0,
                         (z_est / 3.0).unsqueeze(-1), prev_a, u_act.unsqueeze(-1)], dim=-1)
        return {"img": self.frame.clone(), "vec": vec}

    def privileged(self, clear):
        st, task = self.state, self.task
        rel = self._gap_center() - st["p"]
        rel_b = quat_rotate_inv(st["q"], rel)
        dyn = task["dyn"]
        return torch.cat([
            rel_b / 5.0,
            task["gap_w"].unsqueeze(-1), task["gap_h"].unsqueeze(-1),
            task["gap_roll"].sin().unsqueeze(-1), task["gap_roll"].cos().unsqueeze(-1),
            st["v"] / 5.0,
            clear.clamp(-0.2, 2.0).unsqueeze(-1) / 2.0,
            (self._effective_wind_steady() + st["wind"]) / 3.0,
            (dyn["mass"] / 0.775 - 1.0).unsqueeze(-1),
            (dyn["tmax"] / (dyn["mass"] * 9.81) / 3.0).unsqueeze(-1),
            (dyn["delay"].float() / 2.0).unsqueeze(-1),
            (dyn["tau_rate"] * 20.0).unsqueeze(-1),
            task["feasible"].float().unsqueeze(-1),
            (task["geo_margin"] / 0.3).clamp(-1, 1).unsqueeze(-1),
            ((st["p"][:, 0] - task["wall_x"]) / 5.0).unsqueeze(-1),
        ], dim=-1)

    def _gap_center(self):
        return torch.stack([self.task["wall_x"] + 0.5 * self.task["thick"],
                            self.task["gap_cy"], self.task["gap_cz"]], dim=-1)

    def _effective_wind_steady(self):
        """Wind acting now; optional latent crosswind appears only near the gap."""
        dyn = self.task["dyn"]
        base = dyn["wind_steady"]
        local = dyn.get("probe_wind")
        if local is None or not getattr(self.cfg.task, "info_gate_enabled", False):
            return base
        start = self.task["wall_x"] - self.cfg.task.info_probe_distance
        ramp = max(float(self.cfg.task.info_probe_ramp), 1e-6)
        alpha = ((self.state["p"][:, 0] - start) / ramp).clamp(0.0, 1.0)
        return base + alpha.unsqueeze(-1) * local

    # ------------------------------------------------------------------- step
    def step(self, action):
        cfg, st = self.cfg, self.state
        n, dev = self.n, self.dev
        action = action.clamp(-1.0, 1.0)
        # action delay ring buffer (per-env latency)
        D = self.delay_buf.shape[1]
        self.delay_buf[:, self.buf_ptr] = action
        sel = (self.buf_ptr - self.task["dyn"]["delay"]) % D
        applied = self.delay_buf[torch.arange(n, device=dev), sel]
        self.buf_ptr = (self.buf_ptr + 1) % D
        t_cmd, w_cmd = dynamics.action_to_cmd(applied, self.task["dyn"])
        # substep-rate collision accumulation (prevents tunneling/grazing misses)
        min_clear = torch.full((n,), 1e9, device=dev)
        contact_speed = torch.full((n,), -1.0, device=dev)

        def _cb(s):
            nonlocal min_clear, contact_speed
            c = collision.clearance(s["p"], s["q"], self.task, self.bpts,
                                    cfg.sim.arena_y, cfg.sim.arena_z)
            newly = (c < 0.0) & (contact_speed < 0.0)
            contact_speed = torch.where(newly, s["v"].norm(dim=-1), contact_speed)
            min_clear = torch.minimum(min_clear, c)

        dyn_step = dict(self.task["dyn"])
        dyn_step["wind_steady"] = self._effective_wind_steady()
        dynamics.step(st, t_cmd, w_cmd, dyn_step, cfg.sim.dt_ctrl,
                      cfg.sim.substeps, substep_cb=_cb)
        # sensor bias OU
        s = cfg.sensor
        dt = cfg.sim.dt_ctrl
        self.v_bias += dt * (-self.v_bias / s.vel_bias_tau) \
            + (dt ** 0.5) * s.vel_bias_sigma * torch.randn(n, 3, device=dev) * 0.1
        self.z_bias += dt * (-self.z_bias / s.z_bias_tau) \
            + (dt ** 0.5) * s.z_bias_sigma * torch.randn(n, device=dev) * 0.1
        self.t_step += 1

        clear = min_clear
        self.ep_min_clear = torch.minimum(self.ep_min_clear, clear)
        x = st["p"][:, 0]
        speed = st["v"].norm(dim=-1)

        # ---- events ----
        contact = clear < 0.0
        cspeed = torch.where(contact_speed >= 0.0, contact_speed, speed)
        ke = 0.5 * self.task["dyn"]["mass"] * cspeed.square()
        coll_high = contact & (ke > cfg.reward.contact_soft_ke)
        coll_soft = contact & ~coll_high
        succ_x = self.task["wall_x"] + self.task["thick"] + cfg.sim.succ_margin
        success = (x > succ_x) & ~contact
        oob = (x < cfg.sim.arena_x_min) | (st["p"][:, 1].abs() > cfg.sim.arena_y) \
            | (st["p"][:, 2] > cfg.sim.arena_z)
        oob = oob & ~success & ~contact

        crossed_in = ~self.in_attempt & (x >= cfg.sim.retry_x) & ~contact
        self.attempts = self.attempts + crossed_in.long()
        self.in_attempt = self.in_attempt | crossed_in
        # deepest point reached in the current attempt (drives the depth-scaled abort reward)
        self.attempt_depth = torch.where(self.in_attempt, torch.maximum(self.attempt_depth, x),
                                         torch.zeros_like(x))
        crossed_out = self.in_attempt & (x < cfg.sim.retry_x)
        aborted = crossed_out & ~contact & ~oob
        self.in_attempt = self.in_attempt & ~crossed_out

        idle = (self.attempts >= 1) & ~self.in_attempt & (speed < cfg.reward.giveup_speed)
        self.retry_dwell = torch.where(idle, self.retry_dwell + 1,
                                       torch.zeros_like(self.retry_dwell))
        gave_up = self.retry_dwell >= self.giveup_steps

        terminated = contact | success | oob | gave_up
        truncated = (self.t_step >= cfg.sim.ep_len) & ~terminated
        done = terminated | truncated

        # attempt end bookkeeping (for aux labels / metrics)
        end_event = aborted | (self.in_attempt & (success | contact | oob | truncated))
        end_outcome = torch.full((n,), OUTCOME["cutoff"], dtype=torch.long, device=dev)
        end_outcome = torch.where(aborted, torch.tensor(OUTCOME["abort"], device=dev), end_outcome)
        end_outcome = torch.where(success, torch.tensor(OUTCOME["success"], device=dev), end_outcome)
        end_outcome = torch.where(contact, torch.tensor(OUTCOME["collision"], device=dev), end_outcome)

        # ---- reward ----
        r = cfg.reward
        dist = (st["p"] - self._target()).norm(dim=-1)
        dp = self.prev_dist - dist
        if r.progress_asymmetric:
            dp = dp.clamp_min(0.0)      # retreat earns no progress penalty (enables safe aborts)
        rew = r.progress_k * dp
        self.prev_dist = dist
        if r.retreat_reward_k > 0.0:
            # direct incentive for the retreat motion: backing away inside an attempt
            retreat = (self.prev_x - st["p"][:, 0]).clamp_min(0.0) * self.in_attempt.float()
            rew = rew + r.retreat_reward_k * retreat
        self.prev_x = st["p"][:, 0].clone()
        rew = rew - r.step_cost
        near_gate = x < (self.task["wall_x"] - r.wall_prox_xgate)
        rew = rew - r.wall_prox_k * torch.where(
            near_gate, (r.wall_prox_margin - clear).clamp_min(0.0), torch.zeros_like(clear))
        rew = rew - r.smooth_k * (action - self.prev_action).square().sum(-1)
        # progressive-commitment: approach slow enough to retain the abort option.
        # In the approach zone, if braking distance exceeds clearance -> penalty.
        # Training-only shaping proxy; do not invent a positive braking lower bound.
        a_brake = (self.task["dyn"]["tmax"] / self.task["dyn"]["mass"] - 9.81).clamp_min(1e-6)
        brake_dist = speed.square() / (2.0 * a_brake) + speed * r.brake_reaction
        in_approach = x < (self.task["wall_x"] - r.brake_gate)
        rew = rew - r.brake_k * (in_approach.float()
                                 * (brake_dist - clear).clamp_min(0.0))
        # soft arena boundary: gentle inward field, task-independent
        over = (st["p"][:, 1].abs() - r.bound_y).clamp_min(0.0) \
            + (st["p"][:, 2] - r.bound_z).clamp_min(0.0) \
            + (r.bound_x_back - st["p"][:, 0]).clamp_min(0.0)
        rew = rew - r.boundary_k * over
        rew = rew - r.attempt_cost * crossed_in.float()
        rew = rew + r.first_attempt_bonus * (crossed_in & (self.attempts == 1)).float()
        depth_frac = ((self.attempt_depth - cfg.sim.retry_x) /
                      (self.task["wall_x"] - cfg.sim.retry_x)).clamp(0.0, 1.0)
        rew = rew + r.abort_bonus * aborted.float() \
            + r.abort_depth_bonus * aborted.float() * depth_frac
        frac_left = 1.0 - self.t_step.float() / cfg.sim.ep_len
        rew = rew + success.float() * (r.success + r.success_time_bonus * frac_left)
        anneal = min(self.difficulty / max(r.coll_anneal_end, 1e-6), 1.0)
        pen_hi = r.collision_high_easy + (r.collision_high - r.collision_high_easy) * anneal
        pen_so = r.collision_soft_easy + (r.collision_soft - r.collision_soft_easy) * anneal
        rew = rew + coll_high.float() * pen_hi
        rew = rew + coll_soft.float() * pen_so
        rew = rew + oob.float() * r.oob
        self.prev_action = action.clone()  # clone: _reset_envs mutates rows in place

        info = {
            "clearance": clear, "success": success, "collision": contact,
            "collision_high": coll_high, "attempt_id": self.attempts.clone(),
            "in_attempt": self.in_attempt.clone(), "end_event": end_event,
            "end_outcome": end_outcome, "terminated": terminated, "truncated": truncated,
            "oob": oob, "gave_up": gave_up & ~contact,
        }

        # ---- episode records + auto-reset ----
        done_idx = done.nonzero(as_tuple=False).squeeze(-1)
        if done_idx.numel() > 0:
            di = done_idx
            info["records"] = {
                "success": success[di].float(), "coll_high": coll_high[di].float(),
                "coll_soft": coll_soft[di].float(), "oob": oob[di].float(),
                "gave_up": (gave_up[di] & ~contact[di]).float(),
                "timeout": truncated[di].float(),
                "n_attempts": self.attempts[di].float(),
                "ep_len": self.t_step[di].float(),
                "min_clear": self.ep_min_clear[di],
                "feasible": self.task["feasible"][di].float(),
                "gap_w": self.task["gap_w"][di],
                "geo_margin": self.task["geo_margin"][di],
            }
            self._reset_envs(di)
        # privileged vector must describe the post-reset state for done envs
        # (it is stored aligned with the NEXT observation by the trainer)
        if done_idx.numel() > 0:
            clear2 = clear.clone()
            clear2[done_idx] = collision.clearance(
                st["p"][done_idx], st["q"][done_idx], self._task_slice(done_idx),
                self.bpts, cfg.sim.arena_y, cfg.sim.arena_z)
        else:
            clear2 = clear
        info["priv"] = self.privileged(clear2)
        obs = self.observe()
        # refresh delayed frame for non-reset envs
        live = ~done
        if live.any():
            li = live.nonzero(as_tuple=False).squeeze(-1)
            self.frame[li] = render.render(st["p"][li], st["q"][li],
                                           self._task_slice(li), self.rays, cfg.sensor)
        return obs, rew, done, info
