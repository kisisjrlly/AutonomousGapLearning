#!/usr/bin/env bash
# Main training + evaluation campaign.
#
# Properties:
#  - Sequential per-run on the single GPU.
#  - Resumable: a run missing ckpt_final.pt resumes from ckpt_latest.pt.
#  - Crash-tolerant: the inner retry loop restarts the python process on a
#    non-zero exit, resuming from the latest checkpoint. (A hard machine hang
#    kills this script too; then just re-run `bash scripts/run_campaign.sh`
#    after reboot — it resumes everything.)
#  - After each run finishes, its evals run immediately (id + OOD splits;
#    full seeds additionally get the context-wipe condition).
#
# Usage: bash scripts/run_campaign.sh            # 300M steps per run
#        bash scripts/run_campaign.sh 200000000  # override budget
set -u
STEPS=${1:-300000000}
PY=/home/zhaoguodong/miniconda3/bin/python3
CFG_TMP=runs/campaign_cfg.yaml          # repo-relative: survives /tmp cleanup after reboot
cd "$(dirname "$0")/.."

run() {
  local name=$1 cfg=$2 seed=$3
  if [ -f "runs/$name/ckpt_final.pt" ]; then
    echo "[skip] $name already finished"; return
  fi
  echo "[start $(date +%H:%M:%S)] $name (seed $seed, ${STEPS} steps)"
  # risk reduction while validating post-BIOS-update stability:
  # cap CPU-side threading (workload is GPU-bound; full 32-thread fan-out
  # only adds heat/power transients on a CPU under stability observation)
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  # Optional GPU power cap (machine has hard-frozen under sustained 320 W GPU
  # load). Needs root; uncomment and pre-auth via visudo if you want it applied:
  #   sudo nvidia-smi -pl 220
  true
  $PY - "$cfg" "$seed" <<'PYEOF'
import sys, yaml
cfg_path, seed = sys.argv[1], int(sys.argv[2])
d = yaml.safe_load(open(cfg_path)) or {}
d.setdefault("ppo", {})["seed"] = seed
yaml.safe_dump(d, open("runs/campaign_cfg.yaml", "w"))
PYEOF
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    local resume=()
    if [ -f "runs/$name/ckpt_latest.pt" ]; then
      resume=(--resume "runs/$name/ckpt_latest.pt")
    fi
    $PY -m agl.train.train --config "$CFG_TMP" --run "$name" \
      --total-steps "$STEPS" "${resume[@]}" > "runs/$name.out" 2>&1
    local ec=$?
    if [ -f "runs/$name/ckpt_final.pt" ]; then
      echo "[done $(date +%H:%M:%S)] $name"; break
    fi
    if [ $ec -ne 0 ]; then
      echo "[retry] $name crashed (exit $ec) on attempt $attempt; resuming from latest ckpt"
      sleep 5; continue
    fi
    echo "[done $(date +%H:%M:%S)] $name (no final ckpt but clean exit)"; break
  done
  [ -f "runs/$name/ckpt_final.pt" ] || { echo "[FAIL] $name never produced ckpt_final.pt"; return 1; }
  eval_run "$name"
}

eval_run() {
  local name=$1 out="results/$name"
  echo "[eval $(date +%H:%M:%S)] $name"
  # always re-run (fast, ~1 min) — stale npz from older checkpoints must not be reused
  $PY -m agl.eval.evaluate --ckpt "runs/$name/ckpt_final.pt" --out "$out" \
    --n 512 --splits id,ood_geom,ood_dyn || echo "[fail] $name eval"
  if [[ "$name" == full_s* ]]; then
    $PY -m agl.eval.evaluate --ckpt "runs/$name/ckpt_final.pt" --out "$out" \
      --n 512 --splits id --wipe-context || echo "[fail] $name wipe eval"
  fi
  echo "[eval done $(date +%H:%M:%S)] $name"
}

run full_s1 configs/full.yaml 1
run full_s2 configs/full.yaml 2
run full_s3 configs/full.yaml 3
run no_memory configs/no_memory.yaml 1
run reset_attempts configs/reset_attempts.yaml 1
run no_prev_action configs/no_prev_action.yaml 1
run no_aux configs/no_aux.yaml 1
echo "CAMPAIGN COMPLETE $(date +%H:%M:%S)"
