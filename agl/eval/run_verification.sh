#!/bin/bash
# Quick Verification Experiment Runner
# This script runs both verification experiments using the latest checkpoint

set -e  # Exit on error

# Configuration
CKPT="runs/recipe_v3/ckpt_latest.pt"
N_EPISODES=100
DEVICE="${AGL_DEVICE:-cpu}"
PYTHON="${AGL_PYTHON:-/home/zhaoguodong/miniconda3/bin/python3}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/verification_${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  Verification Experiments Runner"
echo "========================================"
echo ""
echo "Checkpoint: $CKPT"
echo "Episodes: $N_EPISODES"
echo "Device: $DEVICE"
echo "Output: $OUT_DIR"
echo ""

# Check if checkpoint exists
if [ ! -f "$CKPT" ]; then
    echo -e "${RED}Error: Checkpoint not found: $CKPT${NC}"
    echo "Available checkpoints:"
    ls -lh runs/*/ckpt*.pt 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"

# Experiment 1: GRU Context Verification
echo "========================================"
echo "Experiment 1: GRU Context Verification"
echo "========================================"
echo "Testing whether GRU hidden state influences decisions..."
echo ""

"$PYTHON" -m agl.eval.verify_gru_context \
    --ckpt "$CKPT" \
    --n "$N_EPISODES" \
    --out "$OUT_DIR/gru_verification" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ GRU verification complete${NC}"
    echo ""
else
    echo -e "${RED}✗ GRU verification failed${NC}"
    exit 1
fi

# Experiment 2: Attempt Distribution Analysis
echo "========================================"
echo "Experiment 2: Attempt Analysis"
echo "========================================"
echo "Analyzing multi-attempt behavior and improvement rates..."
echo ""

"$PYTHON" -m agl.eval.analyze_attempts \
    --ckpt "$CKPT" \
    --n "$N_EPISODES" \
    --out "$OUT_DIR/attempt_analysis" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Attempt analysis complete${NC}"
    echo ""
else
    echo -e "${RED}✗ Attempt analysis failed${NC}"
    exit 1
fi

# Summary
echo "========================================"
echo "All Experiments Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUT_DIR"
echo ""
echo "Key output files:"
echo "  - GRU Verification:"
echo "    * $OUT_DIR/gru_verification/gru_verification_data.npz"
echo "    * $OUT_DIR/gru_verification/gru_verification_stats.json"
echo ""
echo "  - Attempt Analysis:"
echo "    * $OUT_DIR/attempt_analysis/attempt_distribution_analysis.png"
echo "    * $OUT_DIR/attempt_analysis/attempt_analysis_stats.json"
echo ""

# Try to display key statistics if jq is available
if command -v jq &> /dev/null; then
    echo "Quick Results Summary:"
    echo "----------------------"

    if [ -f "$OUT_DIR/gru_verification/gru_verification_stats.json" ]; then
        echo ""
        echo "GRU Context Usage:"
        jq -r '"  Mean action diff: \(.mean_diff)\n  Conclusion: \(.conclusion)"' \
            "$OUT_DIR/gru_verification/gru_verification_stats.json"
    fi

    if [ -f "$OUT_DIR/attempt_analysis/attempt_analysis_stats.json" ]; then
        echo ""
        echo "Attempt Distribution:"
        jq -r '"  Mean attempts: \(.mean_attempts)\n  Success rate: \(.final_success_rate)%\n  Improvement rate: \(.improvement_rate)%"' \
            "$OUT_DIR/attempt_analysis/attempt_analysis_stats.json"
    fi
    echo ""
fi

echo -e "${GREEN}Done!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review the generated plots (PNG files)"
echo "  2. Check the statistics JSON files"
echo "  3. Sensitivity is NOT adaptation evidence; do not select architecture by mean_diff"
echo ""
