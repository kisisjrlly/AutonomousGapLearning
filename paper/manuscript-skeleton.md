# Manuscript skeleton（Science Robotics 格式，论文汇编索引）

> 论文 = 下列章节按序拼接。各章节草稿文件在 paper/ 下；数据到位后按 §0 填充顺序
> 把 {{PLACEHOLDER}} 换成 results/paper/summary.json / runs/*/log.csv / latency 输出中的真实数字。
> 生成：2026-08-10。

## 章节与草稿映射

| 论文章节 | 草稿文件 | 状态 |
|---|---|---|
| Title / 作者 | 见下 | 待定（可随时补） |
| Abstract | paper/abstract-draft.md | 骨架+占位符就绪 |
| Introduction | paper/intro-draft.md | 骨架+占位符就绪 |
| Related Work | paper/related-work-draft.md | 正文基本定稿 |
| Results R1–R6 | paper/results-draft.md | 骨架+占位符就绪 |
| Discussion | paper/discussion-draft.md | 正文定稿 |
| Materials and Methods | paper/methods-draft.md | 正文定稿（含实测部署代价） |
| References | paper/references_verified.json | 26 条已核实 |
| 概念框架（并入 Intro/Discussion） | paper/evidence-validity-framing.md | 定稿 |
| 候选方案记录（并入 Discussion） | docs/tech-selection.md §11 | 定稿 |

## 拟用标题

**Learning to try again: safely aborted attempts drive evidence-based re-attempt
for autonomous flight through unknown narrow gaps**

（副题可选：*an end-to-end aerial policy that verifies 'passable' before committing*）

## 图表清单（对应 agl/analysis/figures.py + make_figures.py）

- Fig 1 Overview：系统示意图 + 任务分布 + 机载视角 filmstrip
- Fig 2 Training dynamics：成功率/难度/碰撞/尝试数 × 全部变体（4 面板）
- Fig 3 Adaptation：按尝试序号条件下成功率（full/wipe/reset/no_memory，含 CI）
- Fig 4 Episode anatomy：单 episode 俯视/侧视轨迹 + 净空/速度时序 + 中止标记
- Fig 5 Safety & OOD：碰撞 vs 几何余量；放弃率；ID vs OOD 柱状
- Fig 6 Risk calibration：可靠性图 + 碰撞前风险上升 + abort 时风险
- Table 1：README §12 全指标表（full 池化 3 种子 ± CI）
- Table 2：消融表（full/no_memory/reset/no_prev_action/no_aux/full+wipe）

## §0 填充顺序（数据到手后的机械步骤）

1. 训练全部完成 → `results/paper/summary.json`（make_paper_data.py 已就绪，验证过）。
2. `python -m agl.analysis.make_figures --summary results/paper/summary.json --out paper/figures`
   （figures.py 全部函数已实现并通过 dummy 预演；图4/图1 需 eval 含 frames 的 npz）。
3. 按占位符逐个填入（{{FINAL_SUCCESS}}、{{K1}}、{{K2}}、{{P}}、{{K2_WIPE}}、
   {{COLL_HIGH}}、{{GIVEUP_INFEASIBLE}}、{{ID}}、{{OOD}}、{{LATENCY}}、{{AUROC}}…）。
4. 撰写补充材料（超参表 paper/hyperparams_snapshot.json、完整指标表、附加消融）。

## 诚实清单（提交前逐项核对）

- [ ] 每个数字可溯源到 summary.json / log.csv / latency 输出（禁止编造）
- [ ] 明确"纯仿真研究"，不宣称真机结果
- [ ] 无"零碰撞保证"声称（README §9）
- [ ] 主动中止从不标注为"必然碰撞"（辅助头标签只来自真实碰撞回溯）
- [ ] 引用全部来自 references_verified.json（已核实），无未核实引用
- [ ] evidence-validity 叙事与"能力为本、方法可替换"一致
