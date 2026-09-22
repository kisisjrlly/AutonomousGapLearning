# Next Task Guide

## Current execution (2026-09-21)

The old M0.2 checkpoint-result workflow is retired. Do not wait for `mean_diff` or
`improvement_rate`, and do not choose an architecture from those values. Continue
with `docs/MINIMAL_SAFE_PROBE_PROTOCOL.md`: reference test and safety sweep are
complete; the independent rigid-body stopping gate is now implemented as a
diagnostic. Next implement shadow replay against actual `GapEnv` substep traces,
then the visual-identifiability scenario. Do not start a campaign or modify
`runs/` without an explicit request.

> 2026-09-18 覆盖旧行动：先执行 `docs/MINIMAL_SAFE_PROBE_PROTOCOL.md`。
> 已有抽象几何自检；下一关是视觉可辨识性和刚体安全退出验证。不要用两个标量决定网络重构，不自动恢复训练。

**Current Status**: 🟡 **Waiting for User Action**  
**Last Completed**: M0.2 - Verification experiment scripts  
**Progress**: M0 is 85% complete  

---

## 🚦 Current Situation

**M0.2 has been completed successfully.** All verification experiment scripts are ready to run.

However, **we cannot proceed to the next automated task** until the user runs the experiments and reports the results. This is because the experimental results will determine which implementation path we take in M1 and M2.

---

## 📋 What the User Needs to Do

### Step 1: Run the Experiments (15-30 minutes)

```bash
cd /home/zhaoguodong/work/code/AutonomousGapLearning
./agl/eval/run_verification.sh
```

**Prerequisites**:
- Python environment with PyTorch must be activated
- CUDA device recommended (can fallback to CPU)
- ~100 episodes will be evaluated

**What will happen**:
1. Script loads the latest checkpoint (`runs/recipe_v3/ckpt_latest.pt`)
2. Runs GRU context verification experiment
3. Runs attempt distribution analysis experiment
4. Generates plots and statistics
5. Saves everything to `results/verification_<timestamp>/`

---

### Step 2: Report Two Key Numbers

After experiments complete, open these two files and report the numbers:

**File 1**: `results/verification_<timestamp>/gru_verification/gru_verification_stats.json`
```json
{
  "mean_diff": 0.XXX,  ← Report this number
  ...
}
```

**File 2**: `results/verification_<timestamp>/attempt_analysis/attempt_analysis_stats.json`
```json
{
  "improvement_rate": XX.X,  ← Report this number
  ...
}
```

**Just say**:
> "mean_diff: 0.XXX, improvement_rate: XX.X%"

---

### Step 3: Say "请你继续"

Once you've reported the numbers, say "**请你继续**" and I will automatically:
1. Analyze the results
2. Update the decision log with the chosen path
3. Update risk assessment based on findings
4. Complete M0.3 documentation
5. Start M1 tasks based on the optimal path

---

## 🎯 Why This Matters

The experimental results determine the **entire implementation strategy**:

### If GRU shows strong influence (mean_diff > 0.1):
✅ **Fast Path**: Skip architecture refactoring  
→ Save 2 weeks  
→ Go directly to training recipe improvements  
→ Focus on safety layer integration  

### If GRU shows weak influence (mean_diff < 0.05):
🔧 **Full Path**: Complete architecture redesign  
→ Follow original plan  
→ Implement Context-Aware Policy from scratch  
→ Add Evidence Extractor and Context Adapter  

### If results are mixed:
🔄 **Enhancement Path**: Augment existing architecture  
→ Save 1 week  
→ Add Attention mechanism to GRU  
→ Enhance training objectives  

**This one experiment determines which of these 3 paths we take!**

---

## 🚫 What I Cannot Do Right Now

I **cannot** proceed with these tasks until experimental results are in:

- ❌ Cannot choose implementation architecture (depends on GRU effectiveness)
- ❌ Cannot design training recipe (depends on improvement rate)
- ❌ Cannot finalize M1 task breakdown (depends on chosen path)
- ❌ Cannot update risk assessment (depends on validation results)
- ❌ Cannot complete M0.3 documentation (needs experimental findings)

**All of these are blocked on your experimental results.**

---

## 📊 Alternative: If You Cannot Run Experiments Now

If you cannot run the experiments right now (e.g., Python environment issues, hardware unavailable), you can:

### Option 1: Skip to Documentation Review
Say: "**跳过实验，先做其他任务**"

I can work on:
- Code refactoring tasks
- Documentation improvements
- Tool development
- But I still cannot make architecture decisions without experimental data

### Option 2: Manual Inspection
Say: "**我先手动检查代码，稍后运行实验**"

You can manually inspect:
- Current policy architecture in `agl/policy/`
- Training loop in `agl/train/`
- Existing checkpoints and logs

Then run experiments when ready.

### Option 3: Tell Me the Environment Issue
Say: "**运行实验遇到问题: [描述问题]**"

I can help debug:
- Python environment setup
- CUDA/PyTorch issues
- Checkpoint loading problems
- Any other technical blockers

---

## 📈 Current Project State

```
Overall Progress: 12% (M0: 85%)

M0: Project Initialization ████████████████████░░░░ 85%
  ✅ M0.1: Codebase analysis (DONE)
  ✅ M0.2: Experiment scripts (DONE)
  ⏸️  M0.3: Documentation (BLOCKED - waiting for experiment results)

M1: Baseline Validation ░░░░░░░░░░░░░░░░░░░░░░░░░░ 0%
  ⏸️  Cannot start until M0.3 completes
  ⏸️  Path depends on experimental results

Timeline:
  Target M0 completion: 2026-09-19 (tomorrow)
  Blocked since: 2026-09-18 (today)
  Blocking duration: TBD (waiting for user)
```

---

## ✅ What Has Been Done (M0.2 Deliverables)

All of these files are ready and waiting:

1. ✅ `agl/eval/verify_gru_context.py` - GRU verification experiment
2. ✅ `agl/eval/analyze_attempts.py` - Attempt distribution analysis
3. ✅ `agl/eval/load_checkpoint.py` - Checkpoint inspection tool
4. ✅ `agl/eval/run_verification.sh` - One-command automation (executable)
5. ✅ `agl/eval/QUICKSTART.md` - Complete user guide
6. ✅ `.project/M0.2_completion_report.md` - Technical completion report
7. ✅ `.project/M0.2_summary_zh.md` - Chinese summary for user

**Everything is ready. We're just waiting for you to run the experiments.**

---

## 🎯 Recommended Next Action

**Recommended**: Run the experiments now (15-30 minutes)

```bash
cd /home/zhaoguodong/work/code/AutonomousGapLearning
./agl/eval/run_verification.sh
```

Then report the two numbers and say "**请你继续**".

This unblocks the entire M1-M2 phase and ensures we take the optimal implementation path.

---

**Status**: 🟡 **Waiting for User**  
**Action Required**: Run experiments and report results  
**Estimated Time**: 15-30 minutes  
**Next Claude Action**: Automatic upon receiving "请你继续" + results
