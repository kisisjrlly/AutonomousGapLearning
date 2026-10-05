# 开发过程可视化

## 2026-10-04 本地已生成结果

- 动态制动 GIF：artifacts/progress/dynamic-20261004-accepted-s0/flight.gif。
- 交互录制：artifacts/viz/dynamic-20261004-verified.rrd。
- 使用独立可视化环境打开，避免把训练环境 NumPy 1.x 升级为 Rerun 所需 2.x：

```bash
/home/zhaoguodong/.local/share/agl-viz-venv/bin/rerun artifacts/viz/dynamic-20261004-verified.rrd
```

已验证 RRD 生成与 headless Viewer 读取。seed 0 成功不代表整个任务分布通过，seed 1/2 存在撤退终止。

当前主调试界面是 **GapEnv + Rerun**。GIF 只作为无额外依赖的 fallback；抽象 safe-probe reference
动画只用于协议示意，不能作为学习/飞行能力证据。

完整说明见 `docs/VISUALIZATION_STACK.md`。

## 当前 dynamic-braking v2

先生成真实 GapEnv 轨迹：

```bash
python3 -m agl.eval.closed_loop_probe \
  --pairs 2 \
  --seed 0 \
  --out /tmp/agl-dynamic-brake
```

只有 `summary.json` 中 `protocol_completed=true` 时，`recovery.pt` 才存在。当前协议是：

```text
non-zero-speed APPROACH_PROBE
        ↓
near-wall BRAKE trigger
        ↓
feedback braking + dwell
        ↓
RETREAT
        ↓
settled recovery snapshot
```

它使用仿真真值反馈，不是 learned policy，也不是 predictive safety shield。

### Rerun

安装：

```bash
pip install -r requirements-viz.txt
```

打开：

```bash
python3 -m agl.eval.view_eval_rerun \
  --input /tmp/agl-dynamic-brake/trajectory.npz \
  --task 0
```

当前 Viewer 可同时显示：

- 墙体/0.30 m 窄缝区域；
- dynamic-brake trigger plane；
- 无人机真实 3D 姿态、轨迹和速度向量；
- `APPROACH_PROBE / BRAKE / RETREAT` phase；
- ego RGB（策略相机实际延迟后的图像）；
- actor 18 维 observation vector；
- clearance / clear_pre / attempt / collision；
- CTBR 四通道动作；
- 风向量不再显示；旧 `probe_wind` 字段仅作为全零兼容字段。

当前实验不存在可显示的 `probe_wind`；旧记录中的该字段必须解释为历史数据。

### GIF fallback

```bash
python3 -m agl.eval.animate_eval_3d \
  --input /tmp/agl-dynamic-brake/trajectory.npz \
  --task 0 \
  --out /tmp/agl-dynamic-brake.gif
```

GIF 仍是实际 GapEnv 积分轨迹，但信息密度低于 Rerun。

## 结构性信息检查

```bash
python3 -m agl.eval.verify_sensor_information --pairs 8 --steps 20
```

该工具关闭观测/像素随机噪声，只验证信息通路结构：

- information gate 外 matched pair 的 actor observation 必须一致；
- 成功/失败轨迹只用于检查姿态、净空、接触和退出时序。

它不是 noisy deployment classifier，也不是 learned adaptation 证据。

## same-state 分支基础

dynamic braking 通过后：

```bash
python3 -m agl.eval.same_state_intervention \
  --recovery /tmp/agl-dynamic-brake/recovery.pt \
  --out /tmp/agl-same-state
```

这里的 correct / removed / swapped 目前只是**预留分支名**。工具只证明完整环境 snapshot、
image/action latency buffer、sensor bias、RNG 和第一帧 observation 一致；尚未注入 GRU/task-memory history。

## 旧 open-loop 反例

旧固定 hover/pitch rollout 已移至：

```text
agl/eval/legacy_open_loop_probe_retreat.py
```

原 `safe_probe_retreat.py` 只保留兼容警告。旧结果中“无接触”但撤退末仍高速运动，不能用于恢复能力验收。

## 训练期间看 checkpoint

训练热路径仍不直接写 Rerun。需要观察策略时使用固定验证任务：

```bash
python3 -m agl.eval.view_checkpoint_rerun \
  --ckpt runs/<run>/ckpt_latest.pt \
  --task 0 \
  --device cuda
```

后续自动化应采用低频 checkpoint monitor，而不是给 3072 个训练环境逐帧发送 UI 数据。

## 仍待加入

- candidate action 执行前的 predictive recovery/shield 决定；
- stopping/recovery margin；
- correct / removed / swapped history 的真实注入；
- 三条 branch 的同步 ghost trajectory；
- learned retry 与穿缝结果；
- Isaac Sim 小规模高保真交叉验证。
