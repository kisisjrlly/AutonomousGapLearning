# 开发过程可视化

当前可视化分成三层，证据等级必须区分：

1. **Rerun GapEnv Viewer（主调试界面）**：真实评估/真实 checkpoint 的 3D 姿态、轨迹、ego RGB、
   clearance、risk、attempt、动作和 information-gate 事件。
2. **Matplotlib/GIF fallback**：用于无 Rerun 环境的轻量离线回放。
3. **safe-probe reference 图/动画**：只检查抽象协议，不能当成策略、视觉或刚体飞行结果。

完整设计与命令见 \`docs/VISUALIZATION_STACK.md\`。

## 1. 推荐：真实 GapEnv 交互回放

安装可选依赖：

\`\`\`bash
pip install -r requirements-viz.txt
\`\`\`

建议对重要 checkpoint 重新评估，并保存少量 task 的 ego camera：

\`\`\`bash
python3 -m agl.eval.evaluate \
  --ckpt runs/<run>/ckpt_final.pt \
  --out results/<run> \
  --splits id \
  --n 64 \
  --save-frames 8 \
  --device cuda
\`\`\`

打开 task 0：

\`\`\`bash
python3 -m agl.eval.view_eval_rerun \
  --input results/<run>/eval_id.npz \
  --task 0
\`\`\`

或保存成可重复打开的 Rerun recording：

\`\`\`bash
python3 -m agl.eval.view_eval_rerun \
  --input results/<run>/eval_id.npz \
  --task 0 \
  --out artifacts/viz/<run>-task0.rrd

rerun artifacts/viz/<run>-task0.rrd
\`\`\`

Viewer loader 会按每个 task 的 \`steps\` 截断，禁止把 done 后 auto-reset 的下一任务误当作同一个 episode。

## 2. 训练期间检查 checkpoint 行为

不要把 Rerun 注入 3072-env 训练热路径。另开终端，对固定验证 task 检查当前 checkpoint：

\`\`\`bash
python3 -m agl.eval.view_checkpoint_rerun \
  --ckpt runs/<run>/ckpt_latest.pt \
  --split id \
  --task 0 \
  --device cuda
\`\`\`

这会跑一个小规模 deterministic monitor episode 并打开 Viewer。需要完全避免占用训练 GPU 时，可在 CPU 上运行，
只是速度更慢：

\`\`\`bash
python3 -m agl.eval.view_checkpoint_rerun \
  --ckpt runs/<run>/ckpt_latest.pt \
  --task 0 \
  --device cpu
\`\`\`

后续若要做自动训练监视，应采用“固定 monitor task bank + 低频 checkpoint replay”，而不是随机抓一个训练 env。
这样才能直观看到同一场景上的策略行为随训练如何变化。

## 3. Viewer 当前能看什么

\`\`\`text
3D:
  wall/gap + wall thickness
  retry/success planes
  information-gate probe-zone boundaries
  drone body-disc + body x/y/z axes
  trajectory prefix
  velocity vector
  probe-wind vector (debug-only privileged display)

2D:
  ego RGB actually fed to policy

time series:
  speed
  clearance / clear_pre
  attempt id
  probe activation
  collision
  policy risk
  four action channels

events:
  approach / attempt / contact / success / give-up transitions
\`\`\`

\`probe_wind\` 的显示是调试特权信息，绝不能把它描述成 actor 输入。

## 4. 旧 GIF fallback

已有评估 NPZ 仍可生成 GIF：

\`\`\`bash
python3 -m agl.eval.animate_eval_3d \
  --input results/full_s1/eval_id.npz \
  --task 0 \
  --out artifacts/progress/full-s1-task0.gif
\`\`\`

它使用真实记录的 \`p/v/clearance/attempt_id/collision\` 与任务几何，但交互能力和信息密度明显低于 Rerun。
保留它只是为了不安装额外依赖时快速看轨迹。

## 5. 抽象 safe-probe 参考可视化

\`\`\`bash
python3 -m agl.eval.safe_probe_reference --n 24 --out /tmp/agl-reference.json
python3 -m agl.eval.sweep_safe_probe --n 24 --out /tmp/agl-sweep.json
python3 -m agl.eval.plot_safe_probe --reference /tmp/agl-reference.json --out /tmp/agl-reference.png
python3 -m agl.eval.plot_safe_probe --sweep /tmp/agl-sweep.json --out /tmp/agl-sweep.png
python3 -m agl.eval.animate_safe_probe_3d \
  --input artifacts/progress/reference.json \
  --out artifacts/progress/reference-3d.gif
\`\`\`

这些图中的抽象障碍和手写规划不是学习结果、不是 RGB 感知结果、不是 Isaac Sim 或真机证据。

## 6. 下一步叠加项

真正的刚体 safe probe/history intervention 完成后，同一 Viewer 必须加入：

- safety gate allow/reject；
- recovery/stopping margin；
- probe → stop → retreat → retry phase；
- post-retreat snapshot；
- correct / removed / swapped 三个同步 ghost trajectories；
- history intervention 类型与第一步动作差异。

## 7. Isaac Sim 的位置

当前不直接迁移全部训练到 Isaac Sim。GapEnv 保持高速训练后端；Rerun 负责统一行为审计；后续只把少量
关键场景迁到 Isaac Sim 做更高保真物理/视觉交叉验证。这样不会因为“想看见无人机”而推翻当前训练基础设施。
