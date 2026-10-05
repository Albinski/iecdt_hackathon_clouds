#!/bin/bash
#SBATCH --job-name=ijepa
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

ROOT=/gws/ssde/j25b/iecdt/modis_hackathon
RUN_DIR=runs/ijepa
TEAM=yourteam   # TODO: set your team name before the submission embed

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

# Environment is synced on the login node before submission; don't re-sync here.
#   uv sync --extra wandb        <- needed once, for the logging
RUN="uv run --frozen --no-sync"

# Orchid compute nodes have no outbound network, so the run is written to
# disk and pushed afterwards from a login node. `tracking.resolve_mode`
# already picks offline whenever SLURM_JOB_ID is set; this makes it explicit
# and survives anyone copying the command out of this file by hand.
export WANDB_MODE=offline

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
ls -l /dev/nvidia* || true
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

# --- Pretrain ---------------------------------------------------------------
$RUN python -m iecdt_hackathon.train_ijepa \
    --config configs/ijepa.yaml \
    --out "$RUN_DIR" \
    --wandb

# --- Score every pooling variant on the validation split --------------------
# All four come from the same weights and differ only in model_cfg["pool"], so
# this settles which embedding to submit by measurement rather than argument.
for V in mean meanstd mean4 mean4std; do
    $RUN python -m iecdt_hackathon.embed \
        --checkpoint "$RUN_DIR/best_$V.pt" \
        --data-dir "$ROOT/val" \
        --out embeddings/val \
        --name "ijepa_$V" \
        --overwrite
done
$RUN python -m iecdt_hackathon.evaluate \
    --out results/ijepa \
    --embeddings embeddings/val/ijepa_*.npz

# --- Embed the test split with the default pooling for submission -----------
# Swap best_meanstd.pt for whichever variant the evaluation above ranked first,
# then submit from a sci server with ./submit.sh
$RUN python -m iecdt_hackathon.embed \
    --checkpoint "$RUN_DIR/best_meanstd.pt" \
    --data-dir "$ROOT/test" \
    --out embeddings/test \
    --name "$TEAM" \
    --overwrite

echo "Finished $(date)"
echo
echo "Push the offline wandb run from a LOGIN node (not here):"
cat "$RUN_DIR/wandb_sync.txt" 2>/dev/null || echo "  uv run wandb sync $RUN_DIR/wandb/offline-run-*"
