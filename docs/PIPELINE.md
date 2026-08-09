# PIPELINE — 各阶段运行手册

> 配合 `HANDOFF.md` 使用。所有命令在仓库根目录执行。
> 统一用 `PY=/home/zhaoguodong/miniconda3/bin/python3`（系统 python3 无 torch）。

## 阶段 0：测试与冒烟（每次改代码后）

```bash
$PY -m pytest tests/ -q          # 期望 17 passed
```
覆盖：四元数/刚体/悬停平衡、SDF 碰撞与净空、可行性标签、渲染可见性、
attempt 状态机、辅助标签扫描、PPO 端到端冒烟、**穿墙回归**（高速薄墙不隧道）。

## 阶段 1：训练（单卡 RTX 4080，GPU 向量化 3072 环境）

训练入口：`python3 -m agl.train.train --config <cfg> --run <name> [--total-steps N] [--resume <ckpt>]`

- 配置来源：`configs/*.yaml` 覆盖 `agl/config.py` dataclass 默认值。
- 每迭代 = 3072 环境 × 96 步 rollout = 294,912 步 + 2 epoch PPO/BPTT 更新，≈5.3 s。
- 产出：`runs/<name>/log.csv`（每 10 迭代一行：iter/steps/time/sps/difficulty/succ_ema/
  rew_mean/各 loss/ep/* 指标）、`ckpt_latest.pt`（每 50 迭代）、`ckpt_final.pt`、`config.yaml`、
  `GIT_COMMIT`、`tb/`。

**推荐做法**：不要单独调 train.py，直接跑 `scripts/run_campaign.sh`（它按序跑 7 个 run，
每 run 训练完自动评估，带断点续跑与崩溃重试）。自定义预算：`bash scripts/run_campaign.sh 200000000`。

### 关键训练语义（勿改坏）
- **课程 λ**：`train.py curriculum()` —— 可行任务成功率 EMA（ema=0.98）> up_thresh(0.70) 则
  λ+=0.01，< dn_thresh(0.40) 则 λ−=0.01。λ 控制缝尺寸/滚转/风/不可行比例/碰撞罚退火。
  目标难度 1.0 与评估一致。
- **记忆语义**：GRU 隐状态跨 rollout 持续；episode 终止（done）时清零；若
  `reset_between_attempts` 为真则尝试中止时也清零（消融变体）。
- **碰撞罚课程化**：λ<0.5 时碰撞罚从 −3/−1.5 退火到 −10/−4（探索脚手架）。
- **辅助标签**：0.5 s 内碰撞（掩码：窗口内确知才标负例）；本次尝试成败（中止=未成功，绝不=必然碰撞）。

## 阶段 2：评估

```bash
$PY -m agl.eval.evaluate --ckpt runs/<name>/ckpt_final.pt --out results/<name> \
    --n 512 --splits id,ood_geom,ood_dyn [--wipe-context]
```
- **划分**：`id`（训练分布难度 1 的保留实例）、`ood_geom`（宽度−15%、滚转+15%、厚度+15% 外推）、
  `ood_dyn`（风/质量/推重比/τ_ω ±15%）。任务库种子固定（7001/7002/7003）。
- **--wipe-context**：每次尝试中止时把 GRU 隐状态清零（仅评估时）→ 隔离跨尝试记忆因果效应。
- 产出 `eval_<tag>.npz`：逐 episode 逐步轨迹（位置/速度/四元数/动作/净空/尝试 id/事件）。

**评估已集成进 `scripts/run_campaign.sh`**（每 run 训练完自动跑；full_s* 额外跑 wipe）。
独立跑也可用 `scripts/run_evals.sh`。

## 阶段 3：指标与统计

```bash
$PY -m agl.eval.metrics results/<run>/eval_id.npz ...   # 单文件摘要
$PY -m agl.analysis.make_paper_data --results results --out results/paper/summary.json
```
- `agl/eval/metrics.py`：离线 attempt 分段 → README §12 全指标（首尝试成功率/条件成功率/
  中止率与中止净空/中止恢复率/接触率/放弃率/最小净空/饱和率/影子刹停包络违约率…）。
- `make_paper_data.py`：**论文数据唯一来源**。池化 3 个 full 种子的 episode（簇 bootstrap CI）、
  计算 k2 vs k1 的 z 检验、full vs wipe 的 k2 对比、消融各划分。
- `agl/analysis/stats.py`：bootstrap CI、簇 bootstrap（attempt 按 episode 聚类）、
  双比例 z 检验、配对 bootstrap。

## 阶段 4：图表

```bash
$PY -m agl.analysis.make_figures --summary results/paper/summary.json --out paper/figures
```
图表清单（见 paper/outline.md）：
- fig_training：4 面板训练曲线（success/difficulty/collision/attempts × 全部变体）
- fig_adaptation：按尝试序号的条件下成功率（full/wipe/reset/no-memory，含 CI）
- fig_bars：消融 & OOD 对比、安全指标
- fig_episode：单 episode 行为解剖（俯视+侧视轨迹、净空/速度时序、中止标记）
- fig1_overview：系统/任务/观测示意图
调色板用已验证的 8 色分类表（figures.py 顶部），实体固定颜色、条件用线型。

## 阶段 5：论文与交付

- `paper/` 下素材：outline.md / intro-draft.md / methods-draft.md / references_verified.json。
- 写作纪律：所有数字来自 summary.json/log.csv/npz；仿真定位；不违反 HANDOFF 第 6 节铁律。
- 交付物：论文正文 + 补充材料 + 图表 + 本套文档更新为完成态 + 提交。

## 诊断工具

```bash
$PY -m agl.analysis.latency --ckpt runs/full_s1/ckpt_final.pt   # 部署延迟/参数量
nvidia-smi -pl 250                                              # 降 GPU 功耗（若死机复发）
tail -f /var/log/kern.log | grep -iE "segfault|mce|soft lockup|Xid"  # 硬件稳定性
```
