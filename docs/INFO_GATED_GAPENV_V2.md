# GapEnv v2：信息门控最小物理任务

日期：2026-09-22。

## 目标

旧 GapEnv 的多数随机因素可以在首次视觉或短时本体状态中直接处理，因此策略很容易退化成“一次冲过或撞墙”。
要研究“安全试探后利用经历改变下一次尝试”，任务必须包含一个**初始不可知、但能通过安全交互获得**的变量。

第一版采用隐藏局部横风，而不是再增加 oracle 传感器。它直接作用于现有 6-DoF 动力学，却不进入 actor 的显式观测。

## 配置

```yaml
task:
  info_gate_enabled: true
  info_probe_distance: 1.0
  info_probe_ramp: 0.25
  info_probe_wind: 1.2
```

每个任务采样一个横向 `probe_wind`。在

```text
x <= wall_x - info_probe_distance
```

时它严格为零；进入 probe zone 后在 `info_probe_ramp` 内平滑激活。策略只能从 IMU、VIO 速度与真实控制误差中逐步辨识它。

默认 `info_gate_enabled=False`，因此旧 checkpoint、旧基线和已有训练语义保持兼容。

## 配对任务

`scene.paired_information_tasks()` 生成

```text
pair0(+wind), pair0(-wind), pair1(+wind), pair1(-wind), ...
```

每一对任务的几何、纹理、基础风、质量、执行器参数与传感器偏置逐元素相同，仅隐藏局部风符号不同。
这为后续 same-state history intervention 提供最小可控实验单元。

## 无训练 smoke check

```bash
python3 -m agl.eval.verify_info_gate --pairs 8 --steps 20 --device cpu
```

预期结构：远处 pair wind delta 为 0；进入 probe zone 后隐藏横风方向相反；从匹配刚体初态施加相同
hover 控制后，pair 的横向速度应产生可测分叉。这个检查只验证环境信息结构，不是策略适应证据。

## 当前提交不声称什么

本提交只建立“必须交互才能出现新信息”的物理基础设施，不声称策略已经学会主动试探、退出或重试。
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
