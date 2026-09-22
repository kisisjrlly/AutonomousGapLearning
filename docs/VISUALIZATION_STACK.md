# GapEnv 可视化与回放体系

日期：2026-09-22。

## 目标

本项目不再把可视化理解为“论文图”或“训练结束后生成一个 GIF”。可视化是机器人算法开发基础设施，
作用类似 RViz / Isaac Lab Viewer：让开发者直接观察无人机在每个控制周期做了什么，并把数值指标和行为对应起来。

当前第一版采用：

- **GapEnv**：继续作为高速 GPU 训练/评估后端；
- **Rerun**：交互式 3D 检查、时间轴、图像与标量回放；
- **Matplotlib GIF**：无额外依赖的轻量 fallback；
- **Isaac Sim**：后续物理/视觉交叉验证，不作为当前训练后端替代品。

Rerun 是可选依赖，核心训练代码不会 import 它。

## 安装

\`\`\`bash
pip install -r requirements-viz.txt
\`\`\`

## 推荐工作流：重新评估并保存 ego camera

\`\`\`bash
python3 -m agl.eval.evaluate \
  --ckpt runs/<run>/ckpt_final.pt \
  --out results/<run> \
  --splits id \
  --n 64 \
  --save-frames 8 \
  --device cuda
\`\`\`

\`--save-frames 8\` 只保存 task id 0..7 的 32x24 ego RGB，避免给全部并行环境保存图像造成不必要的
磁盘开销。其他任务仍完整记录位置、姿态、速度、动作、净空、risk、attempt 与事件。

## 直接打开交互式 Viewer

\`\`\`bash
python3 -m agl.eval.view_eval_rerun \
  --input results/<run>/eval_id.npz \
  --task 0
\`\`\`

Viewer 中的主要实体：

\`\`\`text
world/
  wall/frame            墙体线框
  wall/gap              真实滚转窄缝前/后轮廓
  planes/retry          attempt/retry 测量平面
  planes/success        success 测量平面
  probe_zone            information gate 起点与完全激活平面
  trajectory            当前 episode 的轨迹前缀
  drone/body            无人机桨叶保护圆盘近似
  drone/axes            机体系 x/y/z，直接显示姿态
  vectors/velocity      当前速度向量
  vectors/probe_wind    DEBUG 特权显示：当前 information gate 隐藏局部风

sensors/
  ego_rgb               策略真正收到的 RGB（仅 --save-frames 保存的 task）

telemetry/
  speed_mps
  clearance_m
  clearance_pre_m
  attempt_id
  probe_activation
  collision
  policy_risk

action/
  thrust
  roll_rate
  pitch_rate
  yaw_rate

events/
  state                 phase 切换与 attempt/event 日志
\`\`\`

注意：\`probe_wind\` 是 **debug-only privileged visualization**。它不是 actor 观测，不能在算法结果展示中让人误以为
策略直接获得了风向真值。

## 保存可重复回放的 .rrd

\`\`\`bash
python3 -m agl.eval.view_eval_rerun \
  --input results/<run>/eval_id.npz \
  --task 0 \
  --out artifacts/viz/<run>-task0.rrd
\`\`\`

之后直接：

\`\`\`bash
rerun artifacts/viz/<run>-task0.rrd
\`\`\`

可以任意拖动时间轴、一帧一帧检查碰撞前发生了什么。episode loader 强制使用 NPZ 中每个 task 的 \`steps\`
截断轨迹，所以环境 done 后 auto-reset 产生的新任务不会混进同一回放。

## 旧 NPZ 的兼容性

旧评估文件没有 \`rec_frames\` 时，3D、轨迹和 telemetry 仍可以打开，只是没有 ego camera。
旧文件没有 information-gate 元数据时，也不会伪造 probe zone。建议对重要 checkpoint 用新 \`evaluate.py\`
重新评估。

## 训练阶段如何看

v1 首先解决“真实测试 episode 可审计”的问题，不把 Rerun 注入 3072-env 热路径，避免未验证的日志开销改变训练吞吐。
后续训练可视化应复用本目录 backend，以低频方式运行：

\`\`\`text
高吞吐 3072-env train
        |
        +-- 每 N 个 checkpoint / iteration
                |
                +-- 固定 task bank 上跑 4~8 个 deterministic monitor episodes
                        |
                        +-- 写 .rrd / Viewer
\`\`\`

这样训练本身保持纯 GPU 高吞吐，而你可以观察“同一组固定场景上的行为随 checkpoint 怎样变化”，比随机挑一个训练
环境更适合判断策略是否真的从 rush-through 变成 probe/stop/retreat/retry。

## 下一版必须叠加的研究事件

当真正的 6-DoF safe probe 和 history intervention 落地后，Viewer 必须继续加入：

- safety gate allow/reject；
- stopping/recovery margin；
- probe / stop / retreat / retry phase；
- post-retreat snapshot；
- correct / removed / swapped 三个同步 ghost trajectories；
- task-memory / GRU intervention 标记；
- 相同物理状态下第一步动作差异。

届时可视化本身仍然只是诊断与因果实验审计工具，不替代安全统计或正式验收。


## 2026-09-22 dynamic-braking v2 additions

`closed_loop_probe.py` recordings now include `rec_frames` and `rec_obs_vec`. Rerun renders the dynamic brake
trigger plane and separates `APPROACH_PROBE`, `BRAKE` and `RETREAT` phases. This allows direct inspection of
“what the actor could see” alongside privileged state used by the scripted baseline.

The current baseline is still privileged feedback. Viewer overlays do not turn privileged data into policy input and
do not constitute safety certification.
