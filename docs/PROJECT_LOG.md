# PROJECT LOG — 项目时间线与决策记录

> 供任何后续 AI / 协作者理解"为什么是现在这样"。按时间顺序追加。
> 格式：日期 | 事件 | 关键决策/产出 | 影响。

## 2026-07-26（研究启动与基建）

- **11:26** 用户设定总体目标：从 README 出发，自主完成"设想→设计→开发→测试→运行→校验→
  分析→写作"，最终交付 Science Robotics 级别论文。
- **技术选型**（docs/tech-selection.md）：**自研 GPU 向量化 PyTorch 仿真器**（拒绝 Isaac Sim/
  MuJoCo/Genesis——16GB 单卡需数千视觉环境并行，只能全 GPU 零拷贝）；低分辨率平面着色 RGB
  （32×24）+ IMU + 本体状态；CTBR 动作空间；CNN+GRU；循环 PPO + 约束优先多目标奖励 + 自动课程。
- **代码实现**：sim/maths、dynamics、scene、collision（SDF 墙带孔）、render（光线投射）、
  env（多尝试状态机）、models/policy、train/ppo（BPTT）、train/train。**17 项 pytest 全过**。
- **吞吐基准**：rollout 133k env-steps/s @3072 envs；渲染可视化抽查。
- **多智能体评审**（4 视角×对抗验证）：确认 5 个缺陷并修复——
  ① 40Hz 碰撞检查可穿墙/擦边漏检 → 改 **200Hz 子步碰撞检查**（+回归测试）；
  ② 辅助碰撞标签的 valid 掩码死代码丢弃负例 → 重写窗口掩码；
  ③ 特权向量在 reset 前计算 → 对齐到 reset 后状态；
  ④ 截断 episode 的 GAE 把终止/截断混淆 → **时限截断自举**；
  ⑤ CSV 列首行冻结丢失晚出现列 → 静态 schema。
- **冒烟调优**：课程底部加宽（width_lo_easy 0.55→0.80 等）、碰撞罚课程化（λ<0.5 退火）。
  学习信号 10× 提速（iter 10 成功率 0.1% → iter 40 达 20%）。
- **评估管线**：evaluate.py（三划分固定种子任务库）+ metrics.py（README §12 全指标）+
  stats.py（bootstrap/检验）。
- **引用核实**：8 代理 × WebSearch，26 条参考文献确认 + 18 条直接相关新发现
  （recovery/abort 类：Recovery RL、To Err is Robotic、RACER…）→ references_verified.json。

## 2026-07-26 夜（首次战役 + 硬件故障）

- **22:10** 首次启动 `run_campaign.sh`（full×3 + 4 消融，各 3 亿步）。CPU 线程限 4。
- **硬件死机**：用户报告"GPU 训练中整机冻结、只能强制重启"。诊断为 **i9-13900KF 硅片不稳定**
  （证据：7-18 soft lockup 记录"CPU#5 卡死 121938s"、历次开机随机 segfault、BIOS 1010 早于
  Intel 全部缓解措施）。GPU 驱动全程无 Xid → 排除驱动。
- **止损**：用户升级 BIOS 至 **1836（2026-04-16，含 0x12B+ 微码）**+ Intel Default Settings，
  XMP 关闭。验证：BIOS 1836、微码仍 0x133（Linux 热更新上限）。
- **22:10 重启用**：降险配置（OMP=4）重启战役；full_s1 跑到 iter 120（35.4M 步，难度 0，
  成功率 0.64 上升中）后机器再次重启（7-27 09:21 / 11:50），战役中断。内核 124→136（系统更新）。

## 2026-08-09（当前：恢复与交接）

- **11:07** 测量真实单迭代耗时 **5.3 s**（此前误判为分钟级）→ 3 亿步/run ≈1.5–2 h，
  全战役 ≈10–12 h，**预算可行**。
- **11:07** 重写 `run_campaign.sh`：**断点续跑 + 崩溃自恢复 + 训练后自动评估**；
  ckpt_every 100→50（崩溃最多丢 50 迭代）。
- **11:22** full_s1 从 iter 100（29.5M 步）**续跑成功**，战役后台运行。
- 建立交接文档：HANDOFF.md / docs/PIPELINE.md / docs/PROJECT_LOG.md。
- **尚未产出**：评估结果、聚合数据、统计检验、图表、论文正文。

## 待追加（后续 AI 负责）

- [ ] 各 run 完成/评估的时间点与结果
- [ ] summary.json 的关键数字（k1/k2 成功率、wipe 对比 p 值、OOD、消融差距）
- [ ] 图表与论文完成记录
- [ ] 硬件稳定性结论（监视器若报 segfault/MCE 需记录）

## 2026-08-09（下午：关键行为突破 + 奖励修正 + 干净重启）

- **发现核心行为缺失**：full_s1（旧奖励）跑至 iter 700–870，难度卡死 0.68，
  **n_attempts 恒为 1.0**——策略要么一次穿过要么撞，从不"掉头重试"。
  诊断根因：对称进度奖励（0.6·Δdist）**惩罚后退**，直接压制了安全掉头行为的探索空间。
- **奖励修正（capability-first reward）**，commit `feat: capability-first reward`：
  - `progress_asymmetric: true`：只奖前进、不罚后退（让撤退探索免费）；
  - `abort_bonus 0.2→0.3` + 新增 `abort_depth_bonus 1.2`（按尝试深度缩放，深尝试才拿高额，
    防止浅尝辄止刷分）；
  - env 新增 `attempt_depth` 追踪本次尝试最深点。
- **验证（phase1 微调）**：对 iter-800 checkpoint 用新奖励续跑，**n_attempts 从 1.01 爬升至
  1.32**（iter 810→870）——策略开始学会"过不去就掉头再试"。成功率 ~0.62、碰撞率降至 0.21。
- **决策：干净重启全战役**。full_s1 混合配方（209M 旧+91M 新）与即将从零开始的 full_s2/s3
  不一致；论文主模型必须单一一致配方。已将旧 full_s1 存档为 `runs/full_s1_phase1`（数据点），
  16:56 以新奖励从零重启全 7 run（每 run ~90 min，cron 断点续跑兜底）。
- **硬件**：今日已 6+ 次硬冻结重启（GPU 满载数分钟→整机冻结→断电重启），@reboot cron
  持续自动接管训练。等用户执行 `sudo nvidia-smi -pl 220`（限 GPU 功耗，需 root）。

## 2026-08-10（凌晨–上午：干净重启的验证结果 + 决定性干预）

- **干净重启（新奖励从零）full_s1 结果**：课程爬到难度 1.0（iter 700），但
  **n_attempts 恒为 1.0 直至 iter 970（300M 步终点）**——从零训练的策略在难度 1.0 上
  学到 71% 首尝试成功率（确定性评估 78.8%），但**从未发现"掉头"动作**，失败全是
  "全速撞墙"（碰撞瞬间速度 5.5–6.5 m/s，无减速）。
- **诊断**：与 phase1（成熟策略+新奖励微调 → 60 迭代内 n_attempts→1.32）对比，
  从零训练陷入"快冲穿过"局部最优，掉头动作从未被探索。失败模式分析（eval 512 任务）：
  成功穿越最小净空均值 0.075m、20 分位 0.032m；碰撞 100% 为高能碰撞。
- **干预（capability-first reward v2）**：新增 `retreat_reward_k`（尝试区内后退每米 +0.4，
  默认 0.0），直接让"掉头后退"动作可发现。预案：
  ① 在 full_s1 收敛 checkpoint 上开启撤退奖励微调，验证 n_attempts 是否突破 1；
  ② 若生效 → 最终配方 = 新奖励 + retreat_reward，全 run 从零重启（每 run ~90–100 min）；
  ③ 若失效 → 加大 incentive 或改课程节奏（step_up 0.01→0.005 让中等难度停留更久）。
- **full_s1（300M，无 abort）作为"无撤退奖励基线"数据点保留**；其结果用于对照。
- **硬件**：机器继续频繁硬冻结重启（8-9 至今 ≥10 次）。用户已移除 @reboot 自动训练
  （手动恢复命令见 HANDOFF §5）；限 GPU 功耗命令已给出待用户执行（sudo nvidia-smi -pl 220）。
