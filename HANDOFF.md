# HANDOFF — 项目交接索引（给任何后续 AI / 协作者）

> 本文件是**唯一入口**。任何 AI（codex、另一 Claude 会话等）接手本项目时，请先完整阅读本文件，
> 再按需阅读 `docs/PIPELINE.md`（运行手册）、`docs/PROJECT_LOG.md`（历史记录）、
> `README.md`（研究纲领）、`docs/tech-selection.md`（选型依据）、`paper/`（论文素材）。
> 最后更新：2026-09-16（研究主线调整为真机无碰撞在线适应）。

## 1. 一句话是什么

在**未知窄缝**上实现"尝试→安全中止→退出→利用真实经历调整→重试→穿越"完整闭环。
仿真阶段按真机标准验证机制可行性（零接触约束、证据收益、历史对照）；真机阶段在真实物理约束下复现该能力。
适应机制是**跨尝试保留的 GRU 记忆**（权重冻结的"上下文适应"，README §5 方案A），
训练数据核心是**无碰撞的安全中止尝试**而非碰撞（README §2/§9）。

> **北极星（任何接手者必须理解，勿偏离）**：最终交付物是一个**真机能力**——"能过就调整姿态穿过、
> 过不去就在接触前安全掉头、掉头后改起点/姿态/策略再试"。**能力是目标，方法（RL、模仿、
> 规划+学习、LLM 思维链分层控制等）是可替换的手段**（README §4 新增段落、§17；
> LLM 思维链候选方案的接口定义见 docs/tech-selection.md §11）。论文贡献第一序是
> "无碰撞真机能力 + 可复现的获取路径 + 与获取方法无关的评估协议"。**不要**把"必须用 RL"当成项目前提；
> 当前尚无真机结果；真机是最终验收阶段，仿真只能提供候选机制与安全筛选证据。

## 2. 仓库地图

```
README.md                 研究纲领（§1–§17，必读）
docs/tech-selection.md    技术选型与设计决策（仿真器/传感器/动作空间/模型/训练/消融）
docs/PIPELINE.md          各阶段运行手册（本文件的下游）
docs/REAL_FLIGHT_ADAPTATION.md  真机无碰撞在线适应与验收规范
docs/PROJECT_LOG.md       时间线历史记录
paper/outline.md          论文结构大纲
paper/intro-draft.md      Introduction 初稿
paper/methods-draft.md    Methods 素材草稿
paper/references_verified.json  26 条已核实的参考文献 + 18 条相关发现
paper/hyperparams_snapshot.json 配置快照

agl/config.py             全部超参数（dataclass 默认值，YAML 覆盖）
agl/sim/                  GPU 向量化仿真器：maths(四元数)/dynamics(刚体+执行器)/
                          scene(任务分布+域随机化)/collision(SDF碰撞/净空)/render(光线投射RGB)/env(多尝试环境)
agl/models/policy.py      CNN+GRU 策略、非对称批评家、辅助头
agl/train/ppo.py          循环 PPO（BPTT、GAE、辅助标签扫描）
agl/train/train.py        训练入口（rollout 收集 + 课程 + 日志/存档）
agl/eval/evaluate.py      评估：固定种子任务库、三划分、轨迹记录、上下文清空
agl/eval/metrics.py       README §12 全指标（离线的 attempt 分段 + 统计）
agl/analysis/stats.py     bootstrap CI / 比例检验 / 配对检验
agl/analysis/make_paper_data.py  聚合所有 eval → results/paper/summary.json（论文数据源）
agl/analysis/figures.py   论文图表（已验证调色板）
agl/analysis/latency.py   部署延迟基准

configs/*.yaml            full + 4 消融的配置
scripts/run_campaign.sh   训练+评估一体化战役（断点续跑+崩溃自恢复）
scripts/run_evals.sh      独立评估（campaign 已含评估时此脚本备用）
tests/                    pytest：仿真/训练/安全回放/信息门控/可视化回归测试

runs/<run>/               log.csv(训练曲线) config.yaml ckpt_latest.pt ckpt_final.pt tb/
results/<run>/eval_*.npz  评估轨迹（每 episode 逐步记录）
results/paper/summary.json 论文引用数据的单一来源
```

## 3. 当前状态（2026-09-16）

2026-09-22 补充：抽象协议开始迁入真实刚体环境。GapEnv v2 新增默认关闭的 information gate：
隐藏局部横风只在近墙 probe zone 激活；可生成仅 latent wind 符号不同的严格配对任务。
旧 PPO campaign 已降级为 legacy baseline。下一步优先完成刚体 probe/retreat 与 same-state history intervention。

- **当前阻塞**：机器在 GPU 满载下频繁硬死机；更关键的是，**仿真策略尚未按真机标准证明零接触安全试探闭环**——
  这是真机实验的必要前提，必须在仿真中先验证通过（未知窄缝、零接触约束、证据收益、历史替换对照）。
  恢复训练需用户明确指示。恢复命令见 §5。
- **硬件问题（当前主阻塞）**：i9-13900KF 负载下整机硬冻结（无日志、需断电重启），疑似 Vmin-shift
  硅片退化或电源预算不足。已做：BIOS 1836 + Intel Default Settings、XMP 关、GPU 功耗限 250W。
  已架 **被动 GPU 功耗监视器**（agl/analysis/gpu_watch.py，零负载，每秒记录到 runs/gpu_watch.log，
  @reboot 自动启动）——下次死机时最后一条记录即死机瞬间功耗，用于区分电源 vs 硅片。
  已整理 **docs/HARDWARE_ISSUE.md**（RMA 证据包，建议走 Intel 5 年质保换 CPU）。
- **旧待验证干预（现为 legacy）**：recipe_v3（risk 反馈 + 撤退奖励 + 慢课程 + 刹停 shaping）。
  不再把“从零 PPO 涌现 retry”作为默认关键路径；配置保留用于基线和对照。
- **当前关键路径**：information-gated GapEnv v2 → 刚体安全试探/退出 → paired latent task →
  same-state correct/removed/swapped history intervention → 再选择模仿/离线 RL/分层策略或 PPO 微调。
- **已确认基线（full_s1 300M）**：难度 1.0，首尝试 80.7%，碰撞 23%，n_attempts≈1.0（无 abort）。
- **已就绪（等数据）**：评估管线（含 risk 校准 AUROC 0.978 基线）、聚合 make_paper_data（已验证）、
  图表管线（training/adaptation/bars/episode/overview/risk）、论文占位符（abstract/results/
  intro/related-work/methods）、evidence-validity 概念框架、LLM 思维链候选方案记录。
- **尚未产出**：仿真最终评估、统计检验、图表、论文正文，以及真机安全闭环和真机在线适应证据。

## 4. 快速上手（完整命令在 docs/PIPELINE.md）

```bash
PY=/home/zhaoguodong/miniconda3/bin/python3   # PATH 里默认 python3 没有 torch！

# 1) 先验证当前代码、信息门控和安全回放
$PY -m pytest tests/ -q

# 2) 旧 7-run PPO 战役仅作复现基线，默认被脚本阻止
# AGL_ALLOW_LEGACY_CAMPAIGN=1 nohup bash scripts/run_campaign.sh > runs/campaign.out 2>&1 &

/home/zhaoguodong/miniconda3/bin/python3 - <<'EOF'
import csv; r=list(csv.DictReader(open('runs/full_s1/log.csv')))[-1]
print({k:r[k] for k in ('iter','steps','difficulty','succ_ema','ep/success','ep/coll_high')})
EOF

# 3) 训练全部完成后，聚合出论文数据
$PY -m agl.analysis.make_paper_data --results results --out results/paper/summary.json

# 4) 生成论文图表（写入 paper/figures/）
$PY -m agl.analysis.make_figures

# 5) 单元测试（改代码后必须跑）
$PY -m pytest tests/ -q
```

## 5. 崩溃/重启后的恢复（重要）

- **训练进程死了但机器没死**：campaign 脚本自带重试循环，会自动从 `ckpt_latest.pt` 续跑。
- **机器硬死机/重启**：campaign 脚本随进程而死。重启后**手动执行**以下命令恢复
  （用户已移除 @reboot 自动拉起——机器稳定性问题解决前不要重新加回）：

  ```bash
  cd /home/zhaoguodong/work/code/AutonomousGapLearning
  nohup bash scripts/run_campaign.sh > runs/campaign.out 2>&1 &
  ```

  脚本会：① 跳过已有 `ckpt_final.pt` 的 run；② 无 final 的 run 自动从 `ckpt_latest.pt` 续跑；
  ③ 各 run 训练完自动评估。
- **单 run 最多丢失 50 迭代**（ckpt_every=50）。

## 6. 数据溯源铁律（写论文/报告时必须遵守）

- **任何数字必须可追溯到 `results/paper/summary.json`、`runs/*/log.csv` 或评估 npz**，
  禁止编造。summary.json 里每个池化指标自带 bootstrap CI。
- 当前仍无真机结果；论文必须明示这一点，不得把仿真结果写成真机在线学习证据。
- 真机实验的硬约束是无碰撞；碰撞、擦碰或保护罩接触不得作为学习反馈。
- **主动中止样本绝不标注为"必然碰撞"**（辅助头标签仅用真实碰撞的事后回溯）。
- **不声称零碰撞保证**；影子刹停包络只适用于仿真评估，真机安全层必须实际否决不可恢复动作。

## 7. 关键设计决策（改动前先读）

| 决策 | 位置 | 若改动的连锁 |
|---|---|---|
| 无全局位置/缝位姿/尝试计数进观测 | env.py observe() | 泄漏=论文核心主张失效 |
| attempt 分段仅测量侧 | env.py / metrics.py | 不许进观测 |
| 特权信息仅训练批评家/辅助头 | policy.py priv_fc | 不许进动作路径 |
| 碰撞在 200Hz 子步检查 | env.py step 的 _cb | 防穿墙（有回归测试） |
| GAE 时限截断自举 | ppo.py compute_gae | 已修，勿回退 |
| 尝试间清空训练(消融)仅 reset_attempts | config reset_attempts.yaml | |
| 课程 λ 由可行任务成功率 EMA 驱动 | train.py curriculum | 目前 up_thresh=0.70 |

## 8. 已知问题与注意

- **机器硬件不稳定历史**：i9-13900KF 曾在旧 BIOS 下发生整机死锁（内核 soft lockup、
  随机 segfault）。2026-07-26 已升级 BIOS 1836 + Intel Default Settings。此后稳定性待验证
  ——**系统监视器会盯 segfault/MCE/soft lockup**，若复发需重新评估（降 GPU 功耗
  `nvidia-smi -pl 250`、更保守的训练预算、或走 Intel RMA，证据在 journalctl）。
- **PATH**：系统默认 `python3` 是 /usr/bin（无 torch）；必须用
  `/home/zhaoguodong/miniconda3/bin/python3`。脚本里已写死。
- **OMP_NUM_THREADS=4** 已设（降 CPU 压力，吞吐几乎无损）。
- **Markdown lint 警告**（docs/tech-selection.md 表格风格）为样式问题，不影响内容。
- **eval 的 oob/gave_up 判定**已改为读取环境标志而非推断（metrics.py），outcome 五分类严格互斥。

## 9. 剩余工作清单（完成判定标准）

- [x] 技术选型 / 仿真器 / 环境 / 策略 / PPO / 17 测试
- [x] 多智能体评审 + 5 缺陷修复
- [x] 冒烟调优（课程 + 碰撞罚课程化）
- [x] 引用核实（26 条）
- [ ] 仿真安全适应原型完成，并证明经历会改变后续决策
- [ ] **评估完成**（campaign 自动；全 run×3 划分 + full 的 wipe 条件）
- [ ] 聚合 summary.json + 统计检验（k2>k1、wipe 对比、OOD、消融）
- [ ] 论文图表（training/adaptation/bars/episode/overview）
- [ ] 论文正文（abstract/intro/results/discussion/methods/refs）+ 补充材料
- [ ] 真机无碰撞安全层、刹停/退出检查和急停通过独立验证
- [ ] 真机完成“安全试探→退出→基于真实经历改变→再次试探”闭环
- [ ] 本 HANDOFF 更新为"已完成"状态 + 最终提交

## 10. 给接手 AI 的嘱咐

1. 先跑通 `$PY -m pytest tests/ -q` 确认环境。
2. 用第 4 节命令恢复/查看训练；别动正在写的 runs/* 目录。
3. 别"优化"已冻结的核心不变量（第 7 节）——尤其别让特权/全局信息进入观测。
4. 论文写作只引用 summary.json / log.csv / npz 中真实存在的数字。
5. 每个改动：更新 docs/PROJECT_LOG.md 一行记录，commit 到 main。
