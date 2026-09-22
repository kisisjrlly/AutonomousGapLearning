# M0.1 代码库架构分析报告

**分析日期**: 2026-09-17  
**分析者**: Claude (Project Manager)  
**状态**: ✅ 完成

---

## 📊 代码库概览

### 基本统计
- **Python文件总数**: 23个
- **主要模块**: 6个 (models, train, sim, eval, analysis, config)
- **配置文件**: 15个YAML配置
- **文档文件**: 5个关键Markdown文档

### 目录结构
```
AutonomousGapLearning/
├── agl/                    # 核心代码包
│   ├── models/            # 策略网络定义
│   ├── train/             # 训练循环和PPO
│   ├── sim/               # 仿真环境
│   ├── eval/              # 评估工具
│   ├── analysis/          # 分析工具
│   └── config.py          # 配置管理
├── configs/               # 配置文件目录
├── docs/                  # 项目文档
├── paper/                 # 论文草稿
└── .waylog/              # 历史对话记录
```

---

## 🔍 核心组件深度分析

### 1. 策略网络架构 (`agl/models/policy.py`)

**当前架构**：
```python
Policy
├── Encoder (CNN + State MLP)
│   ├── Conv: 3→16→32→64 (visual processing)
│   ├── Vec FC: STATE_DIM(18) → state_feat(64)
│   └── Merge: (img_feat + state_feat) → merge_feat(256)
├── GRU Cell (可选)
│   └── merge_feat(256) → gru_hidden(512)
├── Actor
│   └── trunk + risk_feedback(可选) → 256 → 4D action
├── Critic (privileged)
│   └── trunk + priv_feat(64) → 256 → value
└── Aux Heads (辅助任务)
    └── trunk → 64 → 3 (collision pred等)
```

**关键发现**：

✅ **已有的优势**：
- GRU记忆机制已经存在 (`use_memory: bool`)
- 支持前一帧动作作为输入 (`use_prev_action: bool`)
- 支持风险反馈机制 (`use_risk_feedback: bool`)
- 辅助任务头部已实现 (`aux`)

⚠️ **关键问题**：
1. **GRU可能未被充分利用**：
   - `actor`接收`trunk + risk`，但不清楚GRU hidden state是否真正影响决策
   - 没有显式的"evidence extraction"机制
   - 无法验证GRU是否在利用历史信息

2. **缺少Context-Aware设计**：
   - 没有显式的"trial history"输入
   - 没有"base policy + context adapter"的分离设计
   - 无法区分"基础决策"和"基于经验的调整"

3. **训练目标单一**：
   - 只有PPO的标准损失（policy + value + entropy）
   - 没有"improvement across attempts"的损失
   - 没有"context influence"的正则化

---

### 2. 训练循环 (`agl/train/train.py`)

**当前流程**：
```python
Trainer.collect():
  for t in rollout_len:
    - 如果pending: 重置hidden state
    - policy.step(obs, hidden) → action
    - env.step(action) → next_obs, reward, done
    - 存储到rollout buffer
    - 如果done或end_event: pending=True
```

**关键发现**：

✅ **episode内多次尝试已支持**：
- `attempts`: 尝试次数计数
- `in_attempt`: 当前是否在尝试中
- `end_event`: 尝试结束事件
- `end_outcome`: 尝试结果 (abort/success/collision/cutoff)

✅ **hidden state重置机制**：
- `reset_between_attempts`: 可选择是否在尝试间重置GRU
- 当前默认：`False` (跨尝试保留记忆)

⚠️ **关键问题**：
1. **训练不是按episode组织的**：
   - 当前是按时间步收集rollout
   - 没有"trial 1 → trial 2 → trial 3"的结构化组织
   - 无法计算"trial_2比trial_1好了多少"

2. **奖励设计不鼓励快速适应**：
   - `attempt_cost: -0.3` (惩罚多次尝试)
   - `abort_bonus: +0.3` (鼓励安全中止)
   - 但没有"第2次比第1次好"的显式奖励

3. **统计指标不足**：
   - 只记录`n_attempts`的平均值
   - 没有"第1次vs第2次成功率"对比
   - 没有"improvement rate"指标

---

### 3. 仿真环境 (`agl/sim/env.py`)

**当前设计**：
```python
GapEnv:
  - 向量化环境 (n_envs=3072)
  - 支持多次尝试 (retry_x=1.2作为分界线)
  - 自动重置 (done时)
  - 记录详细信息 (clearance, attempts, outcomes)
```

**关键发现**：

✅ **基础设施完善**：
- 碰撞检测精确 (substep级别)
- 间隙参数化完整 (width, height, roll, position)
- 动力学随机化全面 (mass, thrust, delay, drag, wind)
- 传感器噪声真实 (gyro, acc, vision, VIO)

✅ **尝试机制已实现**：
- `retry_x`: 尝试区边界
- `crossed_in/out`: 进入/退出尝试区
- `aborted`: 安全中止检测
- `attempt_depth`: 尝试深度记录

⚠️ **关键问题**：
1. **info不足以支持context学习**：
   - 只提供`clearance`标量
   - 没有"左间隙"、"右间隙"分离信息
   - 没有"碰撞位置"、"最近点"等细节
   - 缺少"可以往哪个方向调整"的提示

2. **episode记录粗糙**：
   - `records`只在episode结束时记录
   - 没有per-attempt的详细记录
   - 无法分析"第1次做了什么，第2次怎么调整的"

---

## 📋 与目标架构的差距分析

### 目标架构需求（来自之前讨论）

1. **Context-Aware Policy**：
   - Evidence Extractor (Transformer/GRU)
   - Base Policy (基础决策)
   - Context Adapter (基于evidence的调整)

2. **三阶段训练**：
   - Phase 1: 基础技能 (允许碰撞)
   - Phase 2: 上下文学习 (限制碰撞)
   - Phase 3: 零碰撞强化 (强制安全)

3. **历史替换对照**：
   - policy(obs, context=[trial_1]) vs. policy(obs, context=[])
   - 验证改进来自经历而非随机

### 当前代码库状态

| 功能模块 | 当前状态 | 差距评估 | 实施难度 |
|---------|---------|---------|---------|
| **GRU记忆** | ✅ 已有 | 🟡 可能未充分利用 | 🟢 低 (验证+调优) |
| **Episode内多次尝试** | ✅ 已有 | 🟡 数据收集不完整 | 🟢 低 (增强info) |
| **隐藏状态重置控制** | ✅ 已有 | 🟢 可直接使用 | 🟢 低 |
| **Context-Aware架构** | ❌ 没有 | 🔴 完全缺失 | 🟡 中 (新增模块) |
| **Evidence Extraction** | ❌ 没有 | 🔴 完全缺失 | 🟡 中 (新增head) |
| **Improvement Loss** | ❌ 没有 | 🔴 完全缺失 | 🟡 中 (修改训练) |
| **Historical Replacement** | ❌ 没有 | 🔴 完全缺失 | 🟡 中 (评估脚本) |
| **Safety Layer (CBF)** | ❌ 没有 | 🔴 完全缺失 | 🔴 高 (新系统) |
| **Per-attempt Analytics** | 🟡 部分 | 🟡 不够详细 | 🟢 低 (增强记录) |

---

## 🎯 快速验证路径（最小改动）

### 实验1：验证GRU是否已经在用历史

**目标**：确认当前GRU hidden state是否已经影响决策

**方法**：
```python
# 在评估脚本中
for gap in test_gaps:
    # 带历史
    hidden = init_hidden()
    obs_1 = observe(gap)
    action_1, hidden = policy(obs_1, hidden)
    result_1 = execute(action_1)
    
    obs_2 = observe(gap)  # 重新观测
    action_2_with_hist, _ = policy(obs_2, hidden)  # 用更新后的hidden
    
    # 不带历史
    hidden_fresh = init_hidden()
    action_2_no_hist, _ = policy(obs_2, hidden_fresh)  # 用新的hidden
    
    # 对比
    diff = (action_2_with_hist - action_2_no_hist).norm()
    print(f"Action difference: {diff}")
```

**预计耗时**：1天  
**如果diff很小** → GRU没被充分利用，需要改进训练  
**如果diff很大** → GRU已经在用，可以直接构建上层机制

---

### 实验2：分析当前n_attempts分布

**目标**：了解当前策略的尝试行为

**方法**：
```python
# 统计100个episode
n_attempts_list = []
success_by_attempt = {1: 0, 2: 0, 3: 0, 4: 0, 5+: 0}

for ep in range(100):
    result = run_episode()
    n_attempts_list.append(result['n_attempts'])
    if result['success']:
        attempt_num = min(result['n_attempts'], 5)
        success_by_attempt[attempt_num] += 1

# 分析
print(f"平均尝试次数: {mean(n_attempts_list)}")
print(f"第1次就成功: {success_by_attempt[1]}")
print(f"第2次成功: {success_by_attempt[2]}")
print(f"需要3次以上: {success_by_attempt[3]+success_by_attempt[4]+success_by_attempt['5+']}")
```

**预计耗时**：半天  
**如果大部分第1次就成功** → 任务太简单，需要提升难度  
**如果需要很多次** → 策略还没学会快速适应

---

## 🚧 实施路线图建议

### 阶段0：基线验证（1-2周）

1. ✅ **运行当前最佳checkpoint**
   - 加载`recipe_v3`训练的模型
   - 评估当前性能基线
   - 分析失败案例

2. ✅ **执行快速验证实验**
   - 实验1：GRU历史影响验证
   - 实验2：n_attempts分布分析
   - 决策：继续当前架构 vs. 重构

### 阶段1：最小改动原型（2-4周）

**如果GRU已经有效**：
1. 增强info记录（左右间隙、碰撞位置）
2. 修改训练循环，结构化episode内尝试
3. 添加improvement loss
4. 验证：第2次是否明显比第1次好

**如果GRU未充分利用**：
1. 改进训练目标（context influence regularization）
2. 显式evidence extraction head
3. 重新训练Phase 2
4. 验证：GRU影响增强

### 阶段2：完整Context-Aware架构（1-2个月）

1. 实现Evidence Extractor
2. 实现Context Adapter
3. 三阶段训练流程
4. 历史替换对照验证

### 阶段3：Safety Layer（2-3个月）

1. CBF实现
2. Reachability analysis
3. 真机前仿真验证

### 阶段4：真机验证（3-6个月）

1. 真机安全协议
2. 渐进式测试
3. 零碰撞验证

---

## 📊 风险评估

### 高风险项

🔴 **GRU可能未学会利用历史**
- **影响**: 整个方法论基础动摇
- **概率**: 中等 (40%)
- **缓解**: 快速验证实验1，2天内知道结果

🔴 **当前训练数据不支持context学习**
- **影响**: 需要重新收集数据，延期2-4周
- **概率**: 高 (60%)
- **缓解**: 渐进式修改，保留现有能力

🔴 **真机硬件问题未解决**
- **影响**: 无法进行最终验证
- **概率**: 未知 (参考HANDOFF.md的硬件问题)
- **缓解**: 并行开发仿真验证，等待硬件修复

### 中风险项

🟡 **三阶段训练可能需要大量调参**
- **影响**: 延期1-2个月
- **概率**: 中等 (50%)
- **缓解**: 从Phase 1开始，渐进式验证

🟡 **Safety Layer实现复杂度高**
- **影响**: 延期1个月
- **概率**: 中等 (40%)
- **缓解**: 先用简单规则，后续升级CBF

---

## 🎯 下一步行动（M0.2）

**立即可执行的任务**：

1. ✅ **创建项目管理基础设施** (本任务)
   - [x] .project/目录
   - [x] status.json
   - [x] M0_codebase_analysis.md

2. 🔄 **准备验证实验脚本** (下一个任务)
   - [ ] 实验1：GRU历史影响验证脚本
   - [ ] 实验2：n_attempts分布分析脚本
   - [ ] 加载当前最佳checkpoint

3. ⏳ **运行基线评估** (M1.1)
   - [ ] 在测试集上评估当前模型
   - [ ] 记录详细性能指标
   - [ ] 识别典型失败案例

---

## 📝 技术债务和待改进项

1. **文档不足**：
   - 缺少API文档
   - 缺少数据流程图
   - 缺少训练配方详细说明

2. **测试覆盖**：
   - 没有单元测试
   - 没有集成测试
   - 缺少回归测试套件

3. **代码质量**：
   - 部分hardcoded参数
   - 缺少类型注解
   - 缺少日志系统

---

## ✅ 结论

**当前代码库评估**: 🟢 **良好基础，需要渐进式增强**

**关键优势**：
- GRU记忆机制已存在
- Episode内多次尝试已实现
- 仿真环境完善
- 配置系统灵活

**主要差距**：
- Context-Aware设计缺失
- 训练目标单一
- Safety Layer未实现
- 评估指标不足

**建议策略**：
1. **先验证后重构**：快速实验确认GRU状态
2. **渐进式改进**：保留现有能力，逐步增强
3. **并行开发**：仿真+真机准备同时进行
4. **持续验证**：每个阶段都有明确验收标准

**预计时间线**：
- 基线验证：2周
- 最小原型：1个月
- 完整架构：3个月
- 真机验证：6个月

---

**报告状态**: ✅ 完成  
**下一步**: M0.2 - 准备验证实验脚本
