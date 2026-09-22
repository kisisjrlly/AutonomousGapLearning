# 开发过程可视化

当前提供两个轻量可视化：

1. **单任务路径图**：显示抽象试探、原路退出、墙体和窄缝；旁边显示保留信息、移除信息、错误替换信息的结果分解。
2. **安全参数扫描热图**：显示在某个最大延迟下，不同速度和定位误差假设的参考任务完成数。
3. **三维动态演示**：播放同一参考任务的试探、退出和重试路径。

它们服务于开发检查：可以看到是否发生试探、是否返回、哪些分支被安全拒绝，以及保守参数变化的影响。图中的抽象障碍和手写规划不是学习结果、不是 RGB 感知结果、不是 Isaac Sim 或真机证据。

## 运行

```bash
python3 -m agl.eval.safe_probe_reference --n 24 --out /tmp/agl-reference.json
python3 -m agl.eval.sweep_safe_probe --n 24 --out /tmp/agl-sweep.json
python3 -m agl.eval.plot_safe_probe --reference /tmp/agl-reference.json --out /tmp/agl-reference.png
python3 -m agl.eval.plot_safe_probe --sweep /tmp/agl-sweep.json --out /tmp/agl-sweep.png
python3 -m agl.eval.animate_safe_probe_3d --input artifacts/progress/reference.json --out artifacts/progress/reference-3d.gif
```

抽象参考图仍只用于协议检查；真实 `GapEnv` 轨迹已经可由下节工具回放。下一步可视化重点不再是美化参考动画，
而是给真实刚体回放叠加 `probe_wind` 激活、`clear_pre`、安全门决定和介入事件。

## 真实 GapEnv 轨迹回放

已有评估 NPZ 可直接播放：

```bash
python3 -m agl.eval.animate_eval_3d \
  --input results/full_s1/eval_id.npz \
  --task 0 \
  --out artifacts/progress/full-s1-task0.gif
```

这会使用真实记录的 `p/v/clearance/attempt_id/collision` 和任务几何，并显示记录的窄缝滚转轮廓。它是离线回放，不能改变策略，也不能把已有碰撞记录解释为安全成果。

## Isaac Sim 的位置

当前不直接接入 Isaac Sim。现有任务的关键验证是“信息是否通过安全试探获得”和“带延迟/误差能否退出”；Isaac Sim 可提高动力学和传感器真实性，但不会自动提供这两个实验的对照设计。应先让现有 GPU 环境输出完整可视化轨迹，再用一个小规模 Isaac Sim 场景做物理交叉验证，而不是迁移全部训练管线。
