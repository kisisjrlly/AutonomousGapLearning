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

## 2026-09-16（研究主线调整：真机无碰撞在线适应）

- 用户明确最终目标：无人机必须在真机运行时通过无碰撞安全试探学习尝试与重试；仿真训练不能作为最终成果。
- 文档统一调整：仿真定位为基础技能训练、风险筛选和反事实验证；真机无碰撞在线适应成为最终验收。
- 新安全门槛：碰撞、擦碰和保护罩接触不得作为真机学习反馈；所有试探动作必须经过独立可恢复性检查。
- 新验收证据：第一次真实试探产生任务相关信息，第二次方案发生可重复且与该信息对应的改变；历史替换对照用于排除普通状态估计或随机波动。
- 首选路径：冻结基础策略 + 可审计任务记忆 + 独立安全执行层；暂不把真机在线更新完整网络作为第一阶段方案。
- **仿真先行验证原则**：仿真与真机共享验收标准（未知窄缝、零接触、证据收益、历史对照），区别仅在物理真实性。
  仿真必须先按真机标准验证完整闭环通过，才能进入真机阶段。仿真不是"降低标准的训练场"，而是"可控环境中的真机标准预演"。

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

## 2026-08-10（下午：校准诊断 + 风险反馈干预 recipe_v2）

- **撤退奖励微调（abort_test_v2，retreat_reward_k=0.4，90 迭代）结果**：n_attempts 仍 ~1.0——
  **单独撤退奖励不足以让收敛的快冲策略掉头**。回收该路径，转从根本上补"预测信号"。
- **辅助头碰撞预测校准测量**（确定性评估 192 环境）：20 步前瞻 AUROC≈0.50（全局），
  但**碰撞前预测单调上升**（20 步前 0.22 → 15:0.48 → 10:0.60 → 5:0.68 → 3:0.71 → 1:0.73）。
  结论：**策略有可用的风险信号，但 actor 从未被奖励去"据其行动"**（actor/aux 头分离，
  PPO 梯度不激励"风险高就掉头"）。
- **干预（recipe_v2）**：把辅助头的碰撞概率作为"风险仪表盘"显式喂给 actor
  （`use_risk_feedback`），配撤退奖励 0.6 + 慢课程（step_up 0.006）。这正是
  evidence-validity 框架的落地：策略基于"可穿越性是否已验证"行动。从零 120M 步实验运行中。
- **论文概念框架**：新增 paper/evidence-validity-framing.md（command-measurement-action
  的单命题实例化：测量查询=进近→commit/abort；假确定性=把影子当裂缝；评估=避免假确定性）。
- **已知问题**：test_attempt_state_machine 偶发失败（CUDA 时序），已加固定种子去抖。

## 2026-08-10（文档：明确记录 LLM 思维链为候选技术方案）

- 应用户要求，在项目文档中**明确记录"大语言模型/视觉-语言模型思维链（LLM chain-of-thought）
  可作为实现本项目的技术方案之一"**（能力为本、方法可替换）：
  - README §4：能力-手段候选途径清单加入 LLM 思维链（command-measurement-action 分层）；
  - README §11：训练数据来源加入"LLM 高层决策轨迹可作为示范/先验，或直接作分层控制高层"；
  - docs/tech-selection.md §11：LLM 思维链方案的定位/接口一致性/已知挑战/与 RL 路径关系；
  - paper/evidence-validity-framing.md：补充两条候选实现路径（端到端 RL / LLM 思维链）；
  - HANDOFF.md 北极星注记同步。
- **训练状态**：应用户要求暂停所有训练（recipe_v3 已启动但立即停止，未继续；用户明确
  不希望启动训练命令——机器在 GPU 训练下频繁硬死机）。恢复训练需用户明确指示。

## 2026-09-18（最小安全试探协议与 CPU 参考实现）

- 新增 `docs/MINIMAL_SAFE_PROBE_PROTOCOL.md` 和独立 `safe_probe_reference.py`：抽象视点门控、几何真值安全层、真实/移除/替换信息配对对照；不是视觉或刚体飞行成果。
- 修正 GRU 诊断为实际安全中止后逐任务采样，排除终止/自动重置样本，比较限幅动作；删除阈值选型和虚假显著性判断。
- 修复尝试分析 OUTCOME 字典接口、配置浅复制、累计成功率分母和无重试样本语义；执行脚本默认 CPU。
- 旧架构选型与宽松无碰撞验收标记废止；不改训练配方、不启动训练、不改已有实验数据。
- CPU 参考运行 24 个任务：real 成功 16 / 放弃 8，removed 放弃 24，swapped 介入 16 / 放弃 8，各分支无接触。仅为脚本化协议自检。
- 新增 6 项回归测试通过；现有仿真/环境非训练测试通过，训练冒烟测试明确排除。
- 更新 `.project/NEXT_TASK.md`，移除等待旧 GRU 两指标的阻塞；新增 36 组合的抽象安全裕度扫描器，作为视觉/刚体验证前的回归门。
- 新增 `agl/sim/safety.py`：独立保守刹停/退出门及 `docs/RIGID_BODY_RECOVERY_GATE.md`；当前仅诊断，不接入训练或宣称形式化安全。
- 运行安全/参考/仿真回归测试：25 passed，2 个训练冒烟测试排除；未启动训练。
- 新增 `replay_trace` 与 `replay_safety_gate.py`，将独立安全门与评估轨迹的同时刻接触/拒绝事件分开统计；明确不能把它解释为未来碰撞的漏报/误报率，且缺少动力学制动下界时拒绝回放。未接入控制或训练。

## 2026-08-13（恢复研究管线：recipe_v3 决定性实验续跑）

- **用户明确恢复指令**："请你继续原先的 research pipeline，先不用管 hang 死的问题了"。
  此前（08-10~08-12）因机器硬死机暂停训练、投入硬件排查（HANG_PROBLEM_SUMMARY.md 交接，
  挂起 passive 监视器 hang_logger + boot_report.sh）。现按用户指令恢复研究主线。
- **recipe_v3 决定性实验续跑**（runs/recipe_v3）：风险反馈 actor + 撤退奖励 0.6 +
  刹停余量惩罚 brake_k 0.3 + 风险时域 coll_horizon 40（1.0s）+ 慢课程 step_up 0.006。
  checkpoint 完好（iter 50 / 14.7M 步 / seed 1），断点续跑至 150M 总步。
  判定标准：**n_attempts 是否从 ~1.0 突破**（abort 行为从零涌现）。
  - 若涌现 → 用 recipe_v3 重跑 full×3 + 4 消融全战役 → eval → summary → 图表 → 论文正文。
  - 若仍 ~1.0 → 从零涌现路线证伪，转"分层/示范/课程注入退避行为"备选路径（能力为本、方法可替换）。
- **兼容性确认**：run 创建于 265e79b 但训练时代码含未提交的 brake_k 改动；当前 HEAD
  （ceb61d5）已含全部改动且 agl/ 工作区干净 → ckpt 可安全续跑。log.csv 历史已备份为
  runs/recipe_v3/log_resume_backup.csv（trainer resume 会截断重写 CSV 头）。

- 2026-09-22：校正真实 GapEnv 三维回放，加入 `gap_roll` 旋转轮廓；生成 `artifacts/progress/full-s1-task0-rotated.gif`。该动画使用真实评估 NPZ 的位置/速度/净空/碰撞记录，仍属于离线诊断。


## 2026-09-22（研究主线落地：information-gated GapEnv v2）

- 将旧 7-run recurrent-PPO campaign 降级为 legacy baseline；脚本默认阻止误启动，复现时需显式设置
  `AGL_ALLOW_LEGACY_CAMPAIGN=1`。
- TaskCfg 新增默认关闭的 information gate：隐藏局部横风只在近墙 probe zone 平滑激活，
  用真实刚体/IMU/VIO 交互制造“安全接近后才出现的新信息”，旧任务默认行为不变。
- 新增 `scene.paired_information_tasks()`：每对任务除隐藏横风符号外，几何、视觉、基础动力学和传感器偏置完全相同。
- 新增 `agl.eval.verify_info_gate` CPU smoke check，验证远处 pair 等价、近墙隐变量激活及相同控制下的刚体响应分叉。
- 修复 safety replay 时序对齐：评估新增 `rec_clear_pre`；门控使用动作前 clearance/velocity，
  并与同一动作随后控制周期内的 `rec_collision` 对齐。旧 NPZ 禁止继续用于该统计。
- 去除 reward shaping 中凭空的 3 m/s² 最小制动下界；该项仍只是训练 proxy，不是安全保证。
- 下一步：实现低速刚体 probe→stop→retreat，保存 post-retreat snapshot，并从同一物理状态执行
  correct / removed / swapped history intervention。
