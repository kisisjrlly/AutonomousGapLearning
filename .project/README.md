# AutonomousGapLearning 项目管理系统

**项目启动日期**: 2026-09-17  
**项目目标**: 实现零碰撞、2-3次试探的类人快速适应能力

---

## 📁 项目管理文件结构

```
.project/
├── README.md                      # 本文件 - 项目管理系统说明
├── status.json                    # 实时项目状态（机器可读）
├── milestones.json                # 里程碑进度追踪
├── decisions.md                   # 关键决策记录
├── blockers.md                    # 风险和阻塞追踪
├── NEXT_TASK.md                   # 下一个任务执行指南
├── M0_codebase_analysis.md        # M0.1 代码库分析报告
└── M0_progress.md                 # M0 阶段进度报告
```

---

## 🎯 如何使用这个系统

### 作为用户

#### 继续项目工作
只需输入：
```
请你继续
```
Claude会自动：
1. 读取 `NEXT_TASK.md` 了解当前任务
2. 执行任务
3. 更新进度
4. 完成后等待下一个"请你继续"

#### 查看项目状态
```
请告诉我当前项目进度
```
Claude会读取 `status.json` 和 `milestones.json` 生成报告。

#### 调整优先级
```
跳过M2，先做M3
```
Claude会更新 `NEXT_TASK.md` 和依赖关系。

#### 记录决策
当你做出重要决策时，Claude会自动更新 `decisions.md`。

### 作为Claude

#### 每次"请你继续"时
1. ✅ 读取 `.project/NEXT_TASK.md`
2. ✅ 执行当前任务
3. ✅ 更新 `.project/status.json`
4. ✅ 记录进度到对应的 `MX_progress.md`
5. ✅ 遇到决策时更新 `decisions.md`
6. ✅ 遇到阻塞时更新 `blockers.md`
7. ✅ 完成任务后更新 `NEXT_TASK.md` 指向下一个任务

#### 标准工作流
```
用户: "请你继续"
  ↓
读取 NEXT_TASK.md → 了解当前任务
  ↓
执行任务（代码、实验、分析）
  ↓
更新项目状态文件
  ↓
生成任务完成报告
  ↓
向用户汇报 + 等待下一个"请你继续"
```

---

## 📊 项目状态文件说明

### status.json
**作用**: 实时项目状态快照（机器可读）

**内容**:
```json
{
  "project_name": "...",
  "current_milestone": "M0",
  "current_task": "M0.2",
  "progress_percent": 5,
  "status": "in_progress",
  "last_update": "2026-09-17T00:00:00Z",
  "next_action": "..."
}
```

**更新频率**: 每次任务完成

---

### milestones.json
**作用**: 8个里程碑的详细进度（机器可读）

**内容**:
```json
{
  "milestones": [
    {
      "id": "M0",
      "name": "Project Initialization",
      "status": "in_progress",
      "progress_percent": 60,
      "deliverables": [...],
      "risks": ["R001", "R002"]
    },
    ...
  ]
}
```

**更新频率**: 每个任务完成、每个里程碑完成

---

### decisions.md
**作用**: 记录所有重要决策及其理由（人类可读）

**格式**:
```markdown
## 决策 #001 - 渐进式改进策略
**日期**: 2026-09-17
**决策者**: Claude + User
**类型**: 🟢 已确认

### 背景
...

### 决策
...

### 理由
...
```

**更新时机**: 遇到需要决策的分叉点

---

### blockers.md
**作用**: 追踪风险、阻塞和缓解措施（人类可读）

**格式**:
```markdown
### 风险 #R001 - GRU可能未充分利用
**状态**: 🟡 待验证
**影响**: 🔴 高
**概率**: 🟡 中

**缓解措施**:
- 快速验证实验
- 准备两套方案
```

**更新频率**: 
- 发现新风险时添加
- 每周审查
- 风险解决时更新

---

### NEXT_TASK.md
**作用**: Claude的"任务执行手册"

**内容**:
- 当前任务的详细说明
- 执行步骤清单
- 验收标准
- 遇到问题时的处理方案

**更新时机**: 任务完成后，更新为下一个任务

---

## 🎯 8个核心里程碑

```
M0: 项目启动 (2周)
  → 代码分析、验证实验准备、基线评估

M1: 基线验证 (2周)
  → 运行实验、分析GRU、确定改进路径

M2: Context-Aware架构 (2周)
  → Evidence Extractor + Context Adapter

M3: Phase 1训练 - 基础技能 (1个月)
  → 允许碰撞，学会飞行、判断、中止

M4: Phase 2训练 - 上下文学习 (1个月)
  → 限制碰撞，学会从试探中快速适应

M5: Phase 3训练 - 零碰撞 (1个月)
  → 强制安全，证明零碰撞下仍能完成

M6: 仿真全面验证 (1个月)
  → Held-out测试、消融实验、论文初稿

M7: 真机验证 (1个月, 阻塞中)
  → 零碰撞真机实验（等待硬件修复）
```

---

## 🔄 典型交互示例

### 示例1: 正常推进
```
用户: 请你继续

Claude:
  [读取 NEXT_TASK.md]
  [执行 M0.2: 创建验证脚本]
  [创建 verify_gru_context.py]
  [创建 analyze_attempts.py]
  [创建 load_checkpoint.py]
  [更新 status.json: M0.2 → 100%]
  [更新 NEXT_TASK.md → M1.1]
  
  ✅ M0.2 完成！
  
  已创建：
  - agl/eval/verify_gru_context.py
  - agl/eval/analyze_attempts.py
  - agl/eval/load_checkpoint.py
  
  下一步：M1.1 - 运行基线评估
  
  准备好后，请说"请你继续"。
```

---

### 示例2: 遇到阻塞
```
用户: 请你继续

Claude:
  [读取 NEXT_TASK.md]
  [尝试加载checkpoint]
  ❌ 错误：找不到checkpoint文件
  
  🔴 阻塞 #B001 记录到 blockers.md
  
  问题：未找到训练好的checkpoint
  路径已检查：
  - ./runs/*/*.pt
  - ./checkpoints/*.pt
  
  解决方案：
  1. 提供checkpoint路径
  2. 使用随机初始化权重（仅验证结构）
  3. 先训练一个简单checkpoint
  
  请选择方案，或提供checkpoint路径。
```

---

### 示例3: 需要决策
```
用户: 请你继续

Claude:
  [读取 NEXT_TASK.md]
  [分析实验结果]
  
  🟡 决策点 #D004
  
  实验1结果：GRU历史影响很小（action差异 < 0.01）
  
  这意味着当前GRU没有被充分利用。
  
  两种路径：
  
  A. 改进训练目标（2-4周）
     + 添加context influence loss
     + 重新训练Phase 2
     - 需要重新训练
  
  B. 重构为显式Context-Aware（4-6周）
     + Evidence Extractor + Context Adapter
     + 更清晰、可解释
     - 实施时间更长
  
  我的建议：路径A（更快，风险更低）
  
  你的决策？
```

---

## 📈 进度查看

### 快速状态
```bash
cat .project/status.json
```

### 详细进度
```bash
cat .project/milestones.json | jq '.summary'
```

### 当前任务
```bash
cat .project/NEXT_TASK.md
```

### 风险列表
```bash
grep "^### 风险" .project/blockers.md
```

---

## 🛠️ 维护指南

### 每个任务完成后
1. 更新 `status.json`
2. 更新 `milestones.json`
3. 创建任务完成报告 `MX_Y_completed.md`
4. 更新 `NEXT_TASK.md` 指向下一个任务

### 每周审查
1. 检查 `blockers.md` 中的风险状态
2. 更新风险概率和缓解措施
3. 检查项目时间线是否需要调整

### 里程碑完成后
1. 标记里程碑为完成
2. 生成里程碑总结报告
3. 更新总体进度
4. 审查下一个里程碑的准备情况

---

## 🎓 设计原则

### 1. 自动化优先
用户只需说"请你继续"，Claude处理所有细节。

### 2. 透明度
所有决策、风险、进度都记录在案，用户随时可查。

### 3. 灵活性
用户可以随时调整优先级、跳过任务、改变方向。

### 4. 持久性
跨会话工作：下次对话时，Claude读取状态继续工作。

### 5. 可追溯性
所有决策有记录，所有变更有理由，所有风险有缓解。

---

## ✅ 系统状态

**系统启动日期**: 2026-09-17  
**当前版本**: 1.0  
**状态**: 🟢 运行中

**已完成的工作**:
- ✅ 项目管理基础设施建立
- ✅ M0.1 代码库分析完成
- ✅ 8个里程碑定义清晰
- ✅ 风险识别完成
- ✅ 决策框架建立

**下一步**:
等待用户输入"请你继续"，执行 M0.2 任务。

---

## 📞 快速参考

| 需求 | 命令 |
|-----|------|
| 继续工作 | "请你继续" |
| 查看进度 | "当前进度如何？" |
| 跳过任务 | "跳过X，做Y" |
| 调整优先级 | "先做X再做Y" |
| 查看风险 | "当前有哪些风险？" |
| 查看决策 | "我们做了哪些决策？" |

---

**文档版本**: 1.0  
**最后更新**: 2026-09-17  
**维护者**: Claude (Project Manager)
