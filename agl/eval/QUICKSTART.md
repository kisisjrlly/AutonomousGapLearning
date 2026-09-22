"""Quick Start Guide for Verification Experiments

2026-09-18 correction: the GRU diagnostic now samples actual safe-abort events,
uses clamped CTBR actions, and reports missing samples as null. It emits JSON/NPZ,
not a GRU PNG. Ignore all older significance thresholds and architecture decision
trees below. The runner defaults to CPU and the documented Miniconda interpreter.
Current protocol: docs/MINIMAL_SAFE_PROBE_PROTOCOL.md.

This guide helps you run the GRU context verification and attempt analysis experiments.

## Prerequisites

The experiments require:
- A trained checkpoint (e.g., runs/recipe_v3/ckpt_latest.pt)
- PyTorch environment with CUDA (if available)
- The agl package installed

## Available Checkpoints

Current checkpoints found in the repository:
- runs/recipe_v3/ckpt_latest.pt (20.1 MB, most recent: Aug 13)
- runs/recipe_v2/ckpt_latest.pt (20.1 MB, Aug 10)
- runs/full_s1/ckpt_latest.pt (20.1 MB, Aug 10)
- runs/full_s1/ckpt_final.pt (20.1 MB, Aug 10)
- runs/abort_test_v2/ckpt_latest.pt (20.1 MB, Aug 10)
- runs/full_s1_phase1/ckpt_latest.pt (20.1 MB, Aug 9)

**Recommendation**: Use `runs/recipe_v3/ckpt_latest.pt` (most recent)

## Experiment 1: GRU Context Verification

**Purpose**: Test whether the GRU hidden state actually influences decision-making

**Command**:
```bash
python3 -m agl.eval.verify_gru_context \\
    --ckpt runs/recipe_v3/ckpt_latest.pt \\
    --n 100 \\
    --out results/gru_verification \\
    --device cuda
```

**What it does**:
1. Runs 100 episodes with the policy
2. At trial 2, compares actions with vs. without GRU history
3. Computes action difference statistics
4. Generates visualization plots

**Expected output**:
- `results/gru_verification/gru_verification_data.npz` - Raw data
- `results/gru_verification/gru_verification_stats.json` - Statistics
- `results/gru_verification/gru_context_verification.png` - Plots

**Interpretation**:
- Mean diff > 0.1: GRU is significantly influencing decisions ✓
- Mean diff 0.05-0.1: GRU has moderate influence ~
- Mean diff < 0.05: GRU has minimal influence ✗

**Time estimate**: 5-10 minutes (depends on GPU)

---

## Experiment 2: Attempt Distribution Analysis

**Purpose**: Analyze how many attempts are needed and whether performance improves

**Command**:
```bash
python3 -m agl.eval.analyze_attempts \\
    --ckpt runs/recipe_v3/ckpt_latest.pt \\
    --n 100 \\
    --out results/attempt_analysis \\
    --device cuda
```

**What it does**:
1. Runs 100 full episodes
2. Records all attempts per episode
3. Analyzes improvement from attempt 1 to attempt 2
4. Computes success rates by attempt number

**Expected output**:
- `results/attempt_analysis/attempt_analysis_episodes.json` - Detailed episode data
- `results/attempt_analysis/attempt_analysis_stats.json` - Statistics
- `results/attempt_analysis/attempt_distribution_analysis.png` - Plots

**Key metrics**:
- Mean attempts per episode
- Success rate by attempt number (1st, 2nd, 3rd)
- Improvement rate (2nd attempt > 1st attempt)
- Average clearance improvement

**Interpretation**:
- High 1st-try success: Policy already generalizes well
- Low improvement rate: Context isn't helping adaptation
- High improvement rate (>70%): Context enables learning ✓

**Time estimate**: 10-20 minutes (depends on episode length)

---

## Experiment 3: Checkpoint Information

**Purpose**: Inspect checkpoint contents without running experiments

**Command**:
```bash
python3 -m agl.eval.load_checkpoint \\
    --ckpt runs/recipe_v3/ckpt_latest.pt \\
    --info
```

**What it shows**:
- Training progress (steps, epoch)
- Model configuration (GRU settings, architecture)
- File size and parameter count
- Task configuration (gap width, etc.)

**Time estimate**: <1 second

---

## Running All Experiments (Batch)

Create a script `run_verification.sh`:

```bash
#!/bin/bash

CKPT="runs/recipe_v3/ckpt_latest.pt"
OUT_DIR="results/verification_$(date +%Y%m%d_%H%M%S)"

echo "Starting verification experiments..."
echo "Checkpoint: $CKPT"
echo "Output: $OUT_DIR"

# Experiment 1: GRU verification
echo "Running GRU context verification..."
python3 -m agl.eval.verify_gru_context \\
    --ckpt $CKPT \\
    --n 100 \\
    --out $OUT_DIR/gru_verification \\
    --device cuda

# Experiment 2: Attempt analysis
echo "Running attempt distribution analysis..."
python3 -m agl.eval.analyze_attempts \\
    --ckpt $CKPT \\
    --n 100 \\
    --out $OUT_DIR/attempt_analysis \\
    --device cuda

echo "All experiments complete!"
echo "Results saved to: $OUT_DIR"
```

Then run:
```bash
chmod +x run_verification.sh
./run_verification.sh
```

**Total time estimate**: 15-30 minutes

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution**: Activate your conda/venv environment first:
```bash
conda activate agl  # or whatever your environment is called
# OR
source venv/bin/activate
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size or use CPU:
```bash
python3 -m agl.eval.verify_gru_context \\
    --ckpt runs/recipe_v3/ckpt_latest.pt \\
    --n 50 \\  # Reduced from 100
    --out results/gru_verification \\
    --device cpu  # Use CPU instead
```

### Issue: "Checkpoint not found"

**Solution**: Verify checkpoint path:
```bash
ls -lh runs/recipe_v3/ckpt_latest.pt
```

### Issue: Experiments run but show no improvement

**This is actually valuable information!** It means:
- The current GRU may not be effectively used
- We need to implement context-aware architecture (as planned)
- This validates the need for the project improvements

---

## Next Steps After Running Experiments

### If GRU shows strong influence (mean diff > 0.1):
✓ Good news! The architecture is already using context
→ Focus on improving training objectives (cross-attempt improvement loss)
→ Skip to M2.2 (enhance training recipe)

### If GRU shows weak influence (mean diff < 0.05):
✗ Architecture isn't leveraging context effectively
→ Need to implement Context-Aware Policy (M1.1)
→ Follow the full implementation plan

### If improvement rate is low (<30%):
✗ Policy isn't learning from attempts
→ Need cross-attempt improvement loss
→ Need explicit evidence extraction

---

## Understanding the Results

Both experiments will generate JSON files with statistics. Here's how to read them:

### gru_verification_stats.json
```json
{
  "mean_diff": 0.087,        // Average action difference
  "pct_significant": 45.2,   // % of cases with diff > 0.1
  "conclusion": "MODERATE: GRU has some influence",
  "confidence": "medium"
}
```

### attempt_analysis_stats.json
```json
{
  "mean_attempts": 2.3,          // Average tries per episode
  "improvement_rate": 34.5,      // % improved on 2nd attempt
  "by_attempt": {
    "attempt_1": {"success_rate": 42.0},  // 1st try success
    "attempt_2": {"success_rate": 58.0}   // 2nd try success
  }
}
```

---

## Questions?

If experiments fail or results are unclear:
1. Check the generated plots (PNG files) for visual insights
2. Examine the raw data files (.npz, .json)
3. Run checkpoint info to verify model configuration
4. Report findings for next steps planning
