# GapEnv v2：历史信息门控实验（已停用）

日期：2026-09-22。

## 目标

本文件记录的隐藏局部横风方案已从当前主线移除。所有风配置字段仍可读取旧文件，
但当前动力学、任务采样、特权观测和可视化都强制使用零风。

## 配置

```yaml
task:
  info_gate_enabled: true
  info_probe_distance: 1.0
  info_probe_ramp: 0.25
  info_probe_wind: 1.2
```

旧版本曾在

```text
x <= wall_x - info_probe_distance
```

时将横风激活；这段语义不再适用于当前实验。

默认 `info_gate_enabled=False`，因此旧 checkpoint、旧基线和已有训练语义保持兼容。

## 兼容接口

`scene.paired_information_tasks()` 生成

```text
pair0a, pair0b, pair1a, pair1b, ...
```

每一对任务逐元素相同，`wind_steady`、`gust_sigma`、`probe_wind` 永远为零。

## 无训练 smoke check

```bash
python3 -m agl.eval.verify_info_gate --pairs 8 --steps 20 --device cpu
```

该命令现在是无风回归检查：即使手动注入旧风字段，也不能改变动力学或观测。

## 当前提交不声称什么

当前主线只验证无风条件下窄缝几何、控制和安全退出基础设施，不声称策略已经学会主动试探、退出或重试。
后续必须完成：

1. 低速刚体 probe → stop → retreat 的零接触脚本/优化器基线；
   `agl.eval.same_state_intervention` 已提供相同初态/匹配任务的分支快照准备，尚未注入策略历史。
   `GapEnv.snapshot()/restore()` 现在覆盖任务、物理状态、延迟队列、观测偏置、frame、尝试 bookkeeping 和 Torch RNG；确定性回放测试通过后，才允许进入三分支 history intervention。
   probe 快照准备现在逐步检查 `done/collision` 和最小净空；一旦发生终止、自动 reset 或接触，工具直接失败，不保存 post-probe 结果。
   旧开环实现已移至 `agl.eval.legacy_open_loop_probe_retreat`，`safe_probe_retreat` 仅保留弃用兼容入口；当前受控恢复基线是 `agl.eval.closed_loop_probe` 的非零速度 dynamic braking 版本。
2. 保存 post-retreat 物理 snapshot；
3. 从同一 snapshot 分支 correct / removed / swapped history；
4. 将短期状态估计 memory 与跨尝试 task memory 分离，避免简单 GRU wipe 的混杂；
5. 比较一次谨慎通过、无记忆重试、规划器、模仿/离线 RL 与 learned policy；
6. 通过后再增加真正的视觉信息门（遮挡、双层 aperture），并用 Isaac Sim 做交叉验证。

## Safety replay 对齐

旧评估把动作前 `v[t]` 与动作执行后的 substep 最小净空 `clear[t]` 混在同一时刻解释。
新评估额外保存 `clear_pre[t]`，安全回放使用：

```text
(clear_pre[t], v_pre[t]) -> gate(action_t) -> collision_after[t]
```

因此旧 NPZ 缺少 `rec_clear_pre` 时必须重新评估，禁止插值或错位补算。


## 当前执行基线（dynamic braking v2）

当前推荐入口是：

```bash
python3 -m agl.eval.closed_loop_probe --out artifacts/progress/dynamic-brake-check
```

该基线使用仿真真值反馈，不是学习策略。与旧版本不同，它以非零 x 速度进入 information gate，
到近墙触发面后实际制动，再撤退并停稳；验收记录 brake-entry speed、stopping distance、最小净空、
阶段峰值速度以及最终速度/角速度。只有全部环境完成恢复门槛后才保存 `recovery.pt`。

随后严格 same-state 基础检查使用完整 recovery snapshot：

```bash
python3 -m agl.eval.same_state_intervention \
  --recovery artifacts/progress/dynamic-brake-check/recovery.pt \
  --out /tmp/same-state
```

这一步只证明未来三个 history 分支能从完整相同环境和相同第一帧观测出发，尚未注入 GRU/task memory。
