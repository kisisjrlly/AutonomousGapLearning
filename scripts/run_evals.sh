#!/usr/bin/env bash
# Evaluate every finished training run on all splits; add context-wipe
# condition for the full-model seeds. Usage: bash scripts/run_evals.sh [N_TASKS]
set -u
N=${1:-512}
cd "$(dirname "$0")/.."

for run in full_s1 full_s2 full_s3 no_memory reset_attempts no_prev_action no_aux; do
  ck="runs/$run/ckpt_final.pt"
  [ -f "$ck" ] || { echo "[skip] $run (no final ckpt)"; continue; }
  out="results/$run"
  if [ ! -f "$out/eval_id.npz" ]; then
    python3 -m agl.eval.evaluate --ckpt "$ck" --out "$out" --n "$N" \
      --splits id,ood_geom,ood_dyn || echo "[fail] $run eval"
  fi
done

# eval-time context wipe on the full models (train-free memory ablation)
for run in full_s1 full_s2 full_s3; do
  ck="runs/$run/ckpt_final.pt"
  [ -f "$ck" ] || continue
  out="results/$run"
  if [ ! -f "$out/eval_id_wipe.npz" ]; then
    python3 -m agl.eval.evaluate --ckpt "$ck" --out "$out" --n "$N" \
      --splits id --wipe-context || echo "[fail] $run wipe eval"
  fi
done
echo "EVALS COMPLETE"
