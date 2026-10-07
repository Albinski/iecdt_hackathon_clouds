#!/bin/bash
# Train one model, score it on the validation split, and (optionally) embed the
# test split for submission.
#
#   sbatch -J ijepa train_ijepa.sbatch ijepa
#   sbatch -J ijepa-cache train_ijepa.sbatch ijepa-cache --set data.cache=ram_uint8
#   MODULE=train EMBED_TEST=1 sbatch -J ae train_ijepa.sbatch ae --set model.width=64
#
# $1 is the run name: it becomes runs/<name>, the wandb run name and the
# embedding file's --name, so one string identifies the checkpoint, the
# embedding and the graphs. Everything after it is passed to the training
# script, so any config key can be swept from the command line.
#
# sweep_ijepa.sh submits a prepared set of these.
#
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
# 16 CPUs rather than 8: the loader, not the GPU, is the bottleneck. On a
# 4-GPU orchid node (32 CPU / 505 GB) this is twice the per-GPU share, so it
# costs some queue time and roughly doubles throughput.
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

RUN_NAME=${1:?usage: sbatch train_ijepa.sbatch NAME [--set key=value ...]}
shift
OVERRIDES=("$@")

ROOT=/gws/ssde/j25b/iecdt/modis_hackathon
RUN_DIR=runs/$RUN_NAME
MODULE=${MODULE:-train_ijepa}            # or 'train' for the autoencoder
CONFIG=${CONFIG:-configs/ijepa.yaml}
EMBED_TEST=${EMBED_TEST:-0}              # 1 to also embed test/ for submission
TEAM=${TEAM:-yourteam}

echo "Job $SLURM_JOB_ID ($RUN_NAME) on $(hostname), started $(date)"
echo "module=$MODULE config=$CONFIG overrides=${OVERRIDES[*]-none}"

# Environment is synced on the login node before submission; don't re-sync here.
#   uv sync --extra wandb        <- needed once, for the logging
RUN="uv run --frozen --no-sync"

# Orchid compute nodes do have outbound internet (verified on gpuhost003),
# so wandb runs appear live and need no post-hoc sync. Set
# WANDB_MODE=offline here if the network ever misbehaves.

# --- GPU preflight ----------------------------------------------------------
# A previous run died at torch._C._cuda_init() with "No CUDA GPUs are available"
# while nvidia-smi listed a healthy A100, and cost a full 12 h slot. That is
# cudaGetDeviceCount() == 0 against a working driver, most often a missing
# /dev/nvidia-uvm in the job's device cgroup: nvidia-smi needs only
# /dev/nvidiactl and /dev/nvidia0, but CUDA also needs UVM. `nvidia-smi -L` is
# the discriminator -- if it prints a UUID and torch still sees no device, the
# node is at fault, so --exclude it and resubmit.
#
# Do NOT work around this by letting the CPU fallback engage. Training on CPU
# for 12 h is worse than failing in ten seconds.
echo "CUDA_VISIBLE_DEVICES=[${CUDA_VISIBLE_DEVICES-unset}]"
nvidia-smi -L || true
$RUN python - <<'PY'
import sys
import torch

count = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  devices {count}")
if count:
    torch.zeros(8, device="cuda").sum().item()   # force a context and a kernel
    print("ok:", torch.cuda.get_device_name(0))
sys.exit(0 if count else 1)
PY
echo "preflight passed"

# --- Train ------------------------------------------------------------------
$RUN python -m "iecdt_hackathon.$MODULE" \
    --config "$CONFIG" \
    --out "$RUN_DIR" \
    --wandb \
    ${OVERRIDES[@]+"${OVERRIDES[@]}"}

# --- Score on the validation split ------------------------------------------
# I-JEPA writes one checkpoint per pooling, all from the same weights, so this
# settles which embedding to submit by measurement rather than by argument.
# The autoencoder writes a single best.pt.
if [ -f "$RUN_DIR/best_meanstd.pt" ]; then
    VARIANTS=(mean meanstd mean4 mean4std)
else
    VARIANTS=("")
fi

NPZ=()
for V in "${VARIANTS[@]}"; do
    if [ -z "$V" ]; then CKPT=$RUN_DIR/best.pt; NAME=$RUN_NAME
    else CKPT=$RUN_DIR/best_$V.pt; NAME=${RUN_NAME}_$V
    fi
    [ -f "$CKPT" ] || { echo "no $CKPT; skipping"; continue; }
    # Non-fatal: one refused embedding must not cost the whole run's scoring.
    if $RUN python -m iecdt_hackathon.embed \
        --checkpoint "$CKPT" \
        --data-dir "$ROOT/val" \
        --out embeddings/val \
        --name "$NAME" \
        --overwrite; then
        NPZ+=("embeddings/val/$NAME.npz")
    else
        echo "embed failed for $NAME; continuing with the rest"
    fi
done

$RUN python -m iecdt_hackathon.evaluate \
    --out "results/$RUN_NAME" \
    --embeddings "${NPZ[@]}"

# --- Embed the test split, only when asked ----------------------------------
# Off by default: exploratory runs do not need it, and the leaderboard allows
# four scored submissions a day.
if [ "$EMBED_TEST" = "1" ]; then
    if [ -f "$RUN_DIR/best_meanstd.pt" ]; then CKPT=$RUN_DIR/best_meanstd.pt
    else CKPT=$RUN_DIR/best.pt
    fi
    $RUN python -m iecdt_hackathon.embed \
        --checkpoint "$CKPT" \
        --data-dir "$ROOT/test" \
        --out embeddings/test \
        --name "$TEAM" \
        --overwrite
fi

echo "Finished $(date)"

# A no-op unless WANDB_MODE was forced to offline above: an online run is
# already in the browser and needs no push.
if [ -f "$RUN_DIR/wandb_sync.txt" ]; then
    echo
    echo "This run logged offline. Push it from a LOGIN node:"
    cat "$RUN_DIR/wandb_sync.txt"
fi
