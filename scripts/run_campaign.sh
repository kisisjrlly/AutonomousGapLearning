#!/usr/bin/env bash
# Main training campaign: sequential runs on the single GPU.
# Usage: bash scripts/run_campaign.sh [TOTAL_STEPS]
set -u
STEPS=${1:-300000000}
cd "$(dirname "$0")/.."

run() {
  local name=$1 cfg=$2 seed=$3
  if [ -f "runs/$name/ckpt_final.pt" ]; then
    echo "[skip] $name already finished"; return
  fi
  echo "[start $(date +%H:%M:%S)] $name (seed $seed)"
  python3 - "$cfg" "$seed" <<'PYEOF'
import sys, yaml
cfg_path, seed = sys.argv[1], int(sys.argv[2])
d = yaml.safe_load(open(cfg_path)) or {}
d.setdefault("ppo", {})["seed"] = seed
yaml.safe_dump(d, open("/tmp/claude-1000/-home-zhaoguodong-work-code-AutonomousGapLearning/6824d512-47fc-4c76-be8f-cc4b18bbc0ef/scratchpad/campaign_cfg.yaml", "w"))
PYEOF
  python3 -m agl.train.train --config /tmp/claude-1000/-home-zhaoguodong-work-code-AutonomousGapLearning/6824d512-47fc-4c76-be8f-cc4b18bbc0ef/scratchpad/campaign_cfg.yaml \
    --run "$name" --total-steps "$STEPS" > "runs/$name.out" 2>&1
  echo "[done  $(date +%H:%M:%S)] $name exit=$?"
}

run full_s1 configs/full.yaml 1
run full_s2 configs/full.yaml 2
run full_s3 configs/full.yaml 3
run no_memory configs/no_memory.yaml 1
run reset_attempts configs/reset_attempts.yaml 1
run no_prev_action configs/no_prev_action.yaml 1
run no_aux configs/no_aux.yaml 1
echo "CAMPAIGN COMPLETE"
