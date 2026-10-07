#!/bin/bash
# Embed a validation (or test) split with every checkpoint in a directory.
#
#   CKPT_DIR=~/ijepa-midsnap OUT=embeddings/val-mid sbatch -J embed-mid embed_many.sbatch
#   CKPT_DIR=runs/ijepa-s16 SPLIT=test OUT=embeddings/test sbatch -J embed-test embed_many.sbatch
#
# Written for scoring the *latest* checkpoints of runs that are still training:
# train_ijepa.py rewrites best_<pool>.pt whenever the in-loop probe improves, so
# snapshot them first (cp to a directory of their own) and point CKPT_DIR there.
# Reading a file the trainer is part-way through writing gives a truncated
# pickle, and the failure looks like a corrupt checkpoint rather than a race.
#
# Each checkpoint's own model_cfg decides the architecture and pooling, so a
# directory may mix runs and widths freely. The embedding is named after the
# checkpoint file, with a leading "<run>--" kept, so one output directory can
# hold several runs without collision.
#
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# An hour is generous: 10,000 tiles through ViT-S/16 takes about a minute on an
# A100 and the loader dominates. Kept short deliberately so the job fits inside
# a gap before a maintenance reservation, which a 12 h request would not.
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CKPT_DIR=${CKPT_DIR:?set CKPT_DIR to a directory of .pt checkpoints}
ROOT=${ROOT:-/gws/ssde/j25b/iecdt/modis_hackathon}
SPLIT=${SPLIT:-val}
OUT=${OUT:-embeddings/$SPLIT}
RUN="uv run python"

echo "Started $(date) on $(hostname)"
echo "checkpoints=$CKPT_DIR split=$SPLIT out=$OUT"

# Fail in seconds rather than after the queue wait if the GPU is not usable.
# nvidia-smi listing a healthy card while torch sees zero devices is a real
# failure mode here (a missing /dev/nvidia-uvm in the job's cgroup), and the
# CPU fallback would quietly take hours.
nvidia-smi -L || true
$RUN - <<'PY'
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "devices", torch.cuda.device_count(), flush=True)
if ok:
    torch.zeros(8, device="cuda").sum().item()
sys.exit(0 if ok else 1)
PY

shopt -s nullglob
checkpoints=("$CKPT_DIR"/*.pt)
if [ ${#checkpoints[@]} -eq 0 ]; then
    echo "No .pt files in $CKPT_DIR" >&2
    exit 1
fi
echo "${#checkpoints[@]} checkpoints to embed"

failed=()
for ckpt in "${checkpoints[@]}"; do
    name=$(basename "$ckpt" .pt)
    echo ""
    echo "=== $name ==="
    # Non-fatal: one checkpoint over the 4,096-dim cap, or a snapshot caught
    # mid-write, must not abandon the rest of the directory.
    if ! $RUN -m iecdt_hackathon.embed \
            --checkpoint "$ckpt" \
            --data-dir "$ROOT/$SPLIT" \
            --out "$OUT" --name "$name" --overwrite; then
        echo "  FAILED: $name"
        failed+=("$name")
    fi
done

echo ""
echo "Finished $(date)"
if [ ${#failed[@]} -gt 0 ]; then
    echo "${#failed[@]} of ${#checkpoints[@]} failed: ${failed[*]}"
fi
echo "Score them with:"
echo "  uv run python scripts/compare_fusion.py --dir $OUT"
