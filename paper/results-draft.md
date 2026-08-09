# Results draft（v1 骨架 — 所有 {{PLACEHOLDER}} 待从 results/paper/summary.json / runs/*/log.csv 填充）

> 填充纪律：每个数字必须来自真实数据；CI 为 95% bootstrap；比例检验为双比例 z 检验。

## R1. 单一循环策略涌现出"试探—中止—重试"的多尝试行为

在约束优先的多目标奖励与自动课程下，单一循环策略在无任何任务状态机的前提下
学会了窄缝穿越。训练动力学（图 2）显示：可行任务成功率从随机初始化爬升至
{{difficulty1_success}}（难度 λ=1 的保留任务，池化 3 种子 ±{{ci}}），同时
高能碰撞率收敛到 {{coll_high}}。课程的自动爬坡将任务难度从易到难推进（图 2b），
而策略没有在更难的几何上崩溃。

行为统计（保留测试，图 3/表 1）：
- 限 5 次尝试内最终成功率：{{final_success}}（可行实例，n={{n_feasible}}）
- 首次尝试成功率：{{first_success}}
- 每次 episode 平均尝试数：{{n_attempts_mean}}；至少一次中止的 episode 比例：
  {{ep_with_abort}}
- 中止后成功返回重试区比例：{{abort_recovery}}
- 任务不可行实例上放弃率：{{giveup_infeasible}}（有限次尝试后合理放弃，
  对应 README §15 问题10）

## R2. 第二次尝试更好——而且只在有记忆时（核心主张）

按尝试序号的条件成功率（图 3，可行实例，簇 bootstrap CI）：
- 第 1 次尝试：{{k1}}（n={{n_k1}}）
- 第 2 次尝试：{{k2}}（n={{n_k2}}）
- 第 3 次尝试：{{k3}}
- k2 vs k1 双比例 z 检验：z={{z_k2k1}}, p={{p_k2k1}}

**因果控制（图 3，免训练）**：同一模型在评估时于每次尝试中止后清零 GRU 隐状态
（full_wipe），第 2 次尝试成功率降至 {{k2_wipe}}，与第 1 次 {{k1_wipe}} 无显著
差异（p={{p_wipe}}）。即：同样的权重、同样的观测，只去掉跨尝试记忆，适应增益即
消失——证明该增益确实由尝试历史驱动，而非任务间偶然性。

**训练态对照**：显式在尝试间清零记忆的变体（reset_attempts）不出现该增益
（k2={{k2_reset}}）；无记忆前馈变体（no_memory）同样没有（k2={{k2_nomem}}）。
含辅助头/前一动作的消融见消融表。

## R3. 记忆携带了什么：中止后的对准修正

配对分析（中止 → 下一次尝试，n={{n_align_pairs}}）：
- 深度点处相对缝心的横向对准误差：中止前 {{align_before}} m → 中止后
  {{align_after}} m（配对 bootstrap 差值 {{align_diff}}，CI {{align_ci}}；
  符号检验 p={{align_p}}）；改善比例 {{align_improve_frac}}。
- 中止前的平均最小净空：{{abort_min_clear}} m（中止时刻仍保留的安全边距）。
这表明策略在失败尝试中提取了可迁移的对准信息并用于下一次尝试——上下文内
系统辨识的直接体现（含隐藏的质量/推重比/风，见方法）。

## R4. 安全：中止让它不碰墙

- 高能碰撞率（episode 级）：{{coll_high_ep}}；软接触率：{{coll_soft_ep}}
- 碰撞随几何余量的分布（图 5）：碰撞集中于低余量实例，接近零余量处才出现
- 影子刹停包络违约率（测量侧，不否决动作）：{{stop_viol}}
- 与"学习靠撞墙"的对照：碰撞作为主要失败信号在训练后期被奖励结构持续抑制，
  碰撞数据仅作为探索脚手架的早期退火（λ<0.5 时罚 -3/-1.5→-10/-4 退火）。

## R5. 泛化：超出训练分布

| 划分 | 成功率(可行) | Δvs ID |
|---|---|---|
| ID（保留，λ=1） | {{id}} | — |
| OOD-geom（宽-15%、滚转+15%、厚+15%） | {{ood_geom}} | {{d_geom}} |
| OOD-dyn（风/质量/推重比/延迟 +15%） | {{ood_dyn}} | {{d_dyn}} |

结论句：{{ood_conclusion}}

## R6. 部署代价

单流策略推理延迟：{{latency_gpu}} ms/步（RTX 4080 参考）/ {{latency_cpu}} ms/步
（CPU 参考）；40 Hz 控制需求余量 {{latency_headroom}}×；部署路径参数量
{{params_deploy}}M（总量 {{params_total}}M）。详见 agl/analysis/latency.py 输出。

## 消融表（表 2，全部评估在保留 ID 划分）

| 变体 | 改动 | 最终成功率 | k1→k2 | 碰撞率 |
|---|---|---|---|---|
| full | — | {{abl_full}} | {{abl_full_k}} | {{abl_full_c}} |
| no_memory | 前馈(仅当前帧) | {{abl_nomem}} | {{abl_nomem_k}} | {{abl_nomem_c}} |
| reset_attempts | 尝试间清空记忆(训练+评估) | {{abl_reset}} | {{abl_reset_k}} | {{abl_reset_c}} |
| no_prev_action | 无上一动作观测 | {{abl_npa}} | {{abl_npa_k}} | {{abl_npa_c}} |
| no_aux | 无辅助头 | {{abl_noaux}} | {{abl_noaux_k}} | {{abl_noaux_c}} |
| full+wipe(评估时) | 仅评估时尝试间清空 | {{abl_wipe}} | {{abl_wipe_k}} | {{abl_wipe_c}} |
