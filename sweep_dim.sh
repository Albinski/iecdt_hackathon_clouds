#!/bin/bash
# Sensitivity of probe scores to embedding size, using the baseline autoencoder.
#
#   chmod +x sweep_dim.sh
#   nohup ./sweep_dim.sh > sweep_dim.log 2>&1 &
#   STEPS=5000 nohup ./sweep_dim.sh > sweep_dim.log 2>&1 &   # quicker first look
#
# Each run: train -> embed the full val split -> (at the end) one side-by-side evaluation.
set -euo pipefail
cd "$(dirname "$0")"

STEPS=${STEPS:-20000}
VAL=/gws/ssde/j25b/iecdt/modis_hackathon/val

run() {  # run <name> [--set key=value ...]
  local name=$1; shift
  echo "=== $name  steps=$STEPS  $(date) ==="
  uv run python -m iecdt_hackathon.train --out "runs/sweep/$name" \
      --set training.steps="$STEPS" "$@"
  uv run python -m iecdt_hackathon.embed --checkpoint "runs/sweep/$name/best.pt" \
      --data-dir "$VAL" --out embeddings/val --name "$name" \
      --num-workers 8 --overwrite
}

# Clean comparison: only the bottleneck changes (default width 32 -> 256 pooled features)
for D in 32 64 128 256; do
  run "d$D" --set model.embedding_dim=$D
done

# Seed repeat of the default, to measure run-to-run noise
run d256_seed1 --set model.embedding_dim=256 --set training.seed=1

# Above 256 the encoder must be widened too (pooled features = 8 * width)
run d512_w64 --set model.embedding_dim=512 --set model.width=64

# Score everything side by side
uv run python -m iecdt_hackathon.evaluate --out results/dim_sweep --n-jobs 8 \
  --embeddings $(for f in embeddings/val/d*.npz; do echo "$(basename "$f" .npz)=$f"; done)

echo "=== done $(date) ==="
