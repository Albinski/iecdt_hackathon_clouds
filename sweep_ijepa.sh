#!/bin/bash
# Submit the I-JEPA sweep to orchid.
#
#   ./sweep_ijepa.sh --dry-run     # print the sbatch commands and stop
#   ./sweep_ijepa.sh               # submit them
#
# Six runs, each answering one question. Step budgets are set so every job
# finishes inside its 12 h slot rather than being cut off by `max_seconds`:
# the LR, weight-decay and EMA schedules are all defined over the *total* step
# count, so a truncated run never anneals and underperforms badly rather than
# slightly.
#
# Rate assumptions, from a measured 258 tiles/s on 8 workers (~2.0 steps/s at
# batch 128) and the ~8x headroom the GPU has over it:
#   uncached, 14 workers  ~3.5 steps/s
#   cached, ViT-S         ~10 steps/s   (GPU-bound; conservative)
#   cached, ViT-Ti        ~15 steps/s
#   cached, ViT-B          ~4 steps/s
set -euo pipefail
cd "$(dirname "$0")"

DRY=""
if [ "${1-}" = "--dry-run" ]; then DRY=1; shift; fi

# Any remaining arguments select a subset of runs by name, for resubmitting
# after a node fault:
#   EXCLUDE=gpuhost004 ./sweep_ijepa.sh ijepa-s16-cache ae-d512
ONLY=("$@")

# A node whose driver is up but whose CUDA is broken takes a job, fails the
# preflight and hands the slot straight back -- so without an exclusion the
# resubmission lands on it again. `nvidia-smi -L` printing a UUID while torch
# reports 0 devices is the signature.
EXCLUDE=${EXCLUDE:-}
EXCLUDE_ARG=()
[ -n "$EXCLUDE" ] && EXCLUDE_ARG=(--exclude="$EXCLUDE")

# A 12 h job will not start before a maintenance reservation that is closer
# than 12 h away -- SLURM parks it until the window *ends*. Check with
# `scontrol show reservation` and set WALLTIME to fit:
#   WALLTIME=11:00:00 MAX_SECONDS=32400 ./sweep_ijepa.sh
# MAX_SECONDS is the training budget inside that slot; the rest pays for the
# cache pre-pass, embedding four variants over 10k val tiles, and the probes.
WALLTIME=${WALLTIME:-09:00:00}
MAX_SECONDS=${MAX_SECONDS:-25200}

submit() {  # submit <name> [env=val ...] -- [--set key=value ...]
    local name=$1; shift
    if [ ${#ONLY[@]} -gt 0 ]; then
        local wanted="" n
        for n in ${ONLY[@]+"${ONLY[@]}"}; do [ "$n" = "$name" ] && wanted=1; done
        [ -n "$wanted" ] || { while [ "${1-}" != "--" ]; do shift; done; return 0; }
    fi
    local -a envs=()
    while [ "${1-}" != "--" ]; do envs+=("$1"); shift; done
    shift
    echo "=== $name"
    if [ -n "$DRY" ]; then
        echo "    ${envs[*]+${envs[*]} }sbatch ${EXCLUDE_ARG[*]-} --time=$WALLTIME -J $name train_ijepa.sbatch $name $* --set training.max_seconds=$MAX_SECONDS"
    else
        local id
        # ${a[@]+"${a[@]}"} so an empty array expands to nothing at all;
        # "${a[@]-}" would pass a single empty argument and env would reject it.
        id=$(env ${envs[@]+"${envs[@]}"} sbatch --parsable \
                 ${EXCLUDE_ARG[@]+"${EXCLUDE_ARG[@]}"} \
                 --time="$WALLTIME" -J "$name" \
                 train_ijepa.sbatch "$name" "$@" \
                 --set "training.max_seconds=$MAX_SECONDS")
        echo "    submitted $id"
    fi
}

# 1. The headline result: does latent-space prediction beat reconstruction at
#    all? Reference config, no cache, so it also measures the real step rate.
submit ijepa-s16 -- \
    --set training.steps=50000

# 2. Sample-limited or capacity-limited? The RAM cache removes the I/O
#    bottleneck entirely (~39 GB, ~5 min pre-pass) and buys roughly 8x the
#    epochs. If this beats run 1 by a lot, everything else should be cached.
submit ijepa-s16-cache -- \
    --set data.cache=ram_uint8 --set training.steps=150000

# 3. Is ViT-S already too big for 100k tiles? ViT-Ti is 4x smaller.
submit ijepa-ti16-cache -- \
    --set model.arch=vit_tiny --set model.pred_emb_dim=96 \
    --set data.cache=ram_uint8 --set training.steps=200000

# 4. The other direction. ViT-B is the paper's smallest well-tested encoder but
#    saw ~79x more samples there, so it is only worth a slot with the cache on.
#    Its mean_last4_std pooling would be 6,144 wide and is skipped automatically.
submit ijepa-b16-cache -- \
    --set model.arch=vit_base --set model.pred_emb_dim=384 \
    --set model.pred_num_heads=6 \
    --set data.cache=ram_uint8 --set training.steps=70000

# 5. The least-confident hyperparameter. The reference's 1e-3 peak is for a
#    global batch of 2048; 3.0e-4 is sqrt-scaled to batch 128, which is a rule
#    of thumb, not a measurement. 1.5e-4 is the documented fallback.
submit ijepa-s16-lr15 -- \
    --set data.cache=ram_uint8 --set training.steps=150000 \
    --set training.lr=1.5e-4 --set training.start_lr=3.0e-5

# 6. The fair comparator. run-log.md's autoencoder rows are all at 5,000 steps
#    and `baseline` at 20,000 was never run, so "I-JEPA beats the AE" would
#    otherwise be a comparison against an undertrained baseline. Best known AE
#    settings, at the config's own step count.
submit ae-d512 MODULE=train CONFIG=configs/default.yaml -- \
    --set model.embedding_dim=512 --set model.width=64 \
    --set training.steps=20000

echo
echo "Watch them with:  squeue -u \$(whoami)"
echo "Logs:             logs/<name>-<jobid>.out"
echo "wandb:            https://wandb.ai/moritz-hauschulz/iecdt-hackathon"
