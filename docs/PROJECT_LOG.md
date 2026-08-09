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
