# PIPELINE — 复审后的执行顺序

当前先验证 V1 数据管线，不运行旧 PPO campaign，不启动长训练。原流程见 `archive/e5753c7/PIPELINE.md`，不再作为当前指令。

```bash
PY=/home/zhaoguodong/miniconda3/bin/python3
$PY -m pytest tests/test_attempt_schema.py tests/test_attempt_dataset.py -q
$PY -m pytest tests/ -q

# 输出必须是新文件名；不建议用 rm -rf 清空历史实验。
$PY -m agl.data.generate_attempt_dataset \
  --out datasets/attempt_v1_smoke.npz --tasks 3 --attempts-per-task 4 \
  --probes-per-task 2 --batch-tasks 3 --seed 0 --device cpu \
  --trace-dir artifacts/progress/attempt_v1_smoke
$PY -m agl.data.audit_attempt_dataset --input datasets/attempt_v1_smoke.npz

$PY -m agl.eval.view_eval_rerun \
  --input artifacts/progress/attempt_v1_smoke/batch_0000.npz --task 0
```

每条小规模试验都要看：阶段、真实轨迹、请求动作、重置前的终止速度、接触事件、缺失标签。可视化工具中已有的净空是该控制区间的结果，不要误解为动作前状态。

本地随后用新输出名测试 `--device cuda`，记录显存/耗时/依赖版本；不能把 CPU 成功直接当成 CUDA 或机载验收。

数据读法：

```python
from agl.data import AttemptDataset
D = AttemptDataset('datasets/attempt_v1_smoke.npz')
parts = D.split(seed=0)  # 按物理 group_id 分组；不是按帧/attempt 分
j = int(D.candidate_indices[0])
history = [i for i in D.task_indices(D.task_id[j])
           if D.attempt_index[i] < D.attempt_index[j] and D.context_eligible[i]]
sample = D.make_query(j, history)
# model(**sample['inputs'])；sample['targets'] 和 mask 只能用于监督/评分。
```

`context_eligible` 是本采集器判定的无接触恢复记录，不是通用安全认证。无上下文候选不能静默删除；应作为单独条件保留。

机械验收通过后，先做小模型和简单回归/GP基线，不做 CEM 和真机穿缝。待预测优势和样本支持范围清楚后，再比较规划选择。所有训练/评估命令、commit、source hash、数据分组清单、失败数都写入新的实验记录。
