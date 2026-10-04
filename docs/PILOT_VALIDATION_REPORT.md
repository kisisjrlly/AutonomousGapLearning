# 本轮验证工具交付与测试记录

日期：2026-10-04。基线：`5c77caf8bce8aef2e00ba280a343ad2f7a6f6502`。

## 这次完成的内容

本轮将整体研究评审落地为“共同候选、重复试验、简单传感器上下文基线、独立的预测与决策指标”。没有再次改写项目终极目标，也没有添加大型世界模型或 CEM。

- 新增 `agl/research/collect_pilot.py`：固定任务种子、共享候选库、独立 rollout 种子与无覆盖清单。
- 新增 `agl/research/pilot.py`：三个问题的离线诊断；物理组划分与配对 bootstrap；岭回归低成本基线。
- 新增 `configs/research_pilot.json`：预先固定探测与候选菜单。
- 在现有生成器只增加可选 `fixed_specs` / `rollout_seed`，默认随机采样调用保持原语义。
- 共享已获得的当前观测，禁止预先读取所有候选起点的观测。
- 保留未恢复探测、接触、无效样本的分母；将条件收益与完整任务净收益分开。
- 添加整体实验协议，更新 HANDOFF 的下一步指向。

## 实际运行

环境：Python 3.13.5，NumPy 2.3.5，CPU。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/test_research_pilot.py -q
```

结果：**18 passed**。

另运行两个 CLI 的 `--help` 和新增模块／修改生成器的编译检查。

测试覆盖：有信息的分析型样例、无适应需求样例、跨重复不稳定排序、失败探测阻断、查询标签不进入特征、候选特定观测不可预先使用、测试标签不影响拟合和选择、重复输入拒绝、执行协议混用拒绝、物理组 bootstrap、无覆盖报告、采集失败清单等。

**所有用于这18项测试的数值数据都是单元测试构造的分析型样例。它们不是 GapEnv 运行结果，更不是无人机学习效果。**报告读取这类数据时会明确标记 `synthetic_logic_test_NOT_drone_result`。

## 源码来源与验证边界

通过 GitHub 连接读取锁定版本文件。以下五个未修改依赖的本地副本均与 GitHub blob SHA 匹配：

- `agl/attempt/__init__.py`: `8b98440898ff1941c0bdee36a32dc76ae1b35b69`
- `agl/attempt/spec.py`: `bca67892e5025de646fe9eee6a2812d24b882c61`
- `agl/attempt/outcome.py`: `fba02968f020a4ccf35d06b4866f75993d566c87`
- `agl/data/__init__.py`: `74a0f472783ed903f5120ad36fc9c4319cefdc4e`
- `agl/data/attempt_dataset.py`: `77566b63894c867fef5dc7309b8b4f83e7d23e95`

修改前的 generator 和 HANDOFF 副本也分别匹配 `1776d629c5cf6cd169fb50297d42bd289d97a1f1` 与 `cb5359e2073cd94113e6e79b45ab4eef4ac306de`。

没有运行：现有仓库完整测试集、真实 GapEnv 共同候选采集、GPU 基准、神经网络训练、图形 Viewer、真机。采集器调用的单元测试使用可注入的假生成器，仅验证参数、种子和文件发布协议。

因此本轮交付的是**能检验研究前提的代码与协议**，不是“三个前提均已成立”的结论。尤其执行器的强度、真实模型失配、采样碰撞检查和连续恢复能力仍是已知限制。

## GitHub 发布状态更新（2026-10-04）

上一轮制作补丁时未能提交远程，因此先交付了补丁包。本次用户明确要求提交后，重新检查发现 GitHub 写入接口可用，已通过该接口发布本轮九个变动文件。具体提交号以包含本文件的 GitHub 提交为准；目标仅为 `kisisjrlly/AutonomousGapLearning` 的 `main`，不创建上游 PR，不强制更新分支。

发布前重新核对了补丁包 SHA256、补丁基线、正向/反向应用、九个文件内容和 Python 语法。除本节更新提交状态外，代码和实验方案与已交付补丁一致。上述 18 passed 是上一轮测试记录，本次发布没有重新运行这些测试，也没有运行全量测试、共同候选 GapEnv 采集、模型训练或真机。

本地拉取后先运行针对性测试和原仓库全量回归，再按 `docs/RESEARCH_VALIDATION_PROTOCOL.md` 生成一组小型真实仿真诊断数据。不要跳过这个阶段直接启动大训练。
