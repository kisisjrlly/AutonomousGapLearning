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

## 当前提交不声称什么

本提交只建立“必须交互才能出现新信息”的物理基础设施，不声称策略已经学会主动试探、退出或重试。
后续必须完成：

1. 低速刚体 probe → stop → retreat 的零接触脚本/优化器基线；
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
