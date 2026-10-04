# AutonomousGapLearning

**目标：无人机利用已发生的无接触尝试，改善对未执行方案的预测，自主改变下一次的起点、进近方向或速度曲线，完成窄缝穿越或安全放弃。**

> 2026-10-04：对 `e5753c7e78826471f5715bf9c6f89051030674f2` 的复审和数据层修正。
> Science Robotics 是投稿目标，不是当前工作已经达到的等级。尚无训练后的上下文预测增益、完整自主重试或真机结果。

先读 [HANDOFF](HANDOFF.md)、[研究方案](docs/CONTEXTUAL_WORLD_MODEL_PLAN.md) 和 [本轮审计](docs/REVIEW_E5753C7.md)。旧入口已存入 `docs/archive/e5753c7/`，仅供追溯。

## 研究路线，不再变成另一场大模型预训练

机载可获得的历史观测与动作 → 上下文条件预测 → 比较候选尝试 → 独立运行时安全检查 → 实际执行 → 新证据。

当前先做**尝试结果预测器**，即某个固定执行器、固定中止规则及时间预算下的结果代理模型。它还不是完整的动力学世界模型；若后续要作此主张，必须增加可查询的动作条件短时响应预测及其验证。

采用显式上下文条件训练是允许的，不称“伪 ICL”，也不冒称 GPT-3 式自然涌现。部署冻结权重不等于从未训练过适应机制。

## 本次已实现

- 六参数版本化 `AttemptSpec`：起点横向/高度偏移、入缝路径切线方向、速度命令上限、前后段加速度。
- 独立仿真试验执行器：机体从指定夹具状态初始化，**不代表无人机已经自行飞到新起点**。
- 固定名义推力标定，不读取每个任务的真实质量/推力上限来做控制补偿；V0 仍使用真实位置、姿态和已知几何，明确标为仿真特权执行器。
- `GapEnv.step(capture_transition=True)` 提供自动重置前的状态及延迟后指令，仅作标签和审计。
- V1 数据分离：传感器经历、当前查询输入、带适用掩码的标签、任务分组和隐藏真值。
- 逐任务/物理组划分；禁止把查询结果、其他任务或“未来”的记录作为正确上下文。
- 可保存现有 Rerun/GIF 格式轨迹，显式记录指令阶段，不能解释为学习策略行为。

## 当前未实现

训练后的预测模型、CEM/其他尝试规划器、连续自主重试、真实转移到新起点、部署安全保护层、真实视觉几何估计、真机实验。独立采集多条轨迹不是这几个能力的替代品。

## 最小运行

在本机已有 PyTorch 环境中执行。不要为此改动原先稳定的训练环境。

```bash
PY=/home/zhaoguodong/miniconda3/bin/python3
$PY -m pytest tests/test_attempt_schema.py tests/test_attempt_dataset.py -q

# 使用新的输出名，已有文件会被拒绝覆盖。
$PY -m agl.data.generate_attempt_dataset \
  --out datasets/attempt_v1_smoke.npz --tasks 3 --attempts-per-task 4 \
  --probes-per-task 2 --batch-tasks 3 --seed 0 --device cpu \
  --trace-dir artifacts/progress/attempt_v1_smoke

$PY -m agl.data.audit_attempt_dataset --input datasets/attempt_v1_smoke.npz
$PY -m agl.eval.view_eval_rerun \
  --input artifacts/progress/attempt_v1_smoke/batch_0000.npz --task 0
```

Rerun 沿用仓库可选依赖及本地已建立的隔离环境。未安装时可以使用原来的 `agl.eval.animate_eval_3d`。

## 数据解释红线

V0 NPZ 不可用于新的训练入口：缺失的传感器证据不能由真值标签补造。V1 读取器直接拒绝，需重新采集。

`inputs` 中没有查询的 `target`、任务 ID 或 `audit_*`。上下文只用已执行试验的模拟机载观测与请求指令，排除了 `thrust/tmax` 这个未确认可机载获取的通道。标签可以来自仿真真值，但不是神经网络的部署输入。

`recovered` 是这次控制器实际退出并停稳，不是“任何情况下都可恢复”；未执行刹停的 `stopping_distance` 用缺失掩码，不填零当作观测。提前中止不能标成“若继续必然撞”。

默认固定几何只是机制诊断，不能作为未知窄缝泛化证据。`--vary-geometry` 在尚无观测几何输入时主动拒绝。

## 安全与历史

仿真数据可包含接触以检查失败及标签，必须如实保留。真机禁止通过故意撞击收集经验。

当前有限点/有限时间的净空检查只是数值诊断，不是连续扫掠体或安全证明。整段轨迹经过一次预测也不能替代飞行过程中持续检查。

保留原 PPO、GRU、动态制动脚本作为历史基线。不得恢复旧训练来代替本路线，也不得以模型大小、试验次数或某个任意误差下降百分比宣布达到期刊标准。
