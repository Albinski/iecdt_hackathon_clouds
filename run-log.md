# Run log

One row per experiment. Numbers are from
`uv run python -m iecdt_hackathon.evaluate` on the **validation** split, which
scores the four published tasks; the leaderboard scores ten, so a row here is an
indicator and not a prediction of rank.

**Seed noise is ±0.008 overall and ±0.017 on `task_6`** (`d256` vs `d256_seed1`).
Treat anything smaller than that as no change.

Conventions:

- **Run** is the `--out` directory under `runs/`, and also the wandb run name and
  the `--name` of its `.npz` under `embeddings/val/`. One string identifies the
  checkpoint, the embedding and the graphs.
- **Key settings** are the `--set` overrides from the config default
  (`configs/default.yaml` for the autoencoder, `configs/ijepa.yaml` for I-JEPA).
  Blank means the config as committed.
- **D** is the embedding width the probes actually see.
- Smoke and shape-check runs are not logged here; only runs whose numbers mean
  something.

## Results

| Run | Model | D | Steps | Key settings | task_4 | task_5 | task_6 | task_7 | Overall |
|---|---|---|---|---|---|---|---|---|---|
| `d32` | ConvAE | 32 | 5,000 | `model.embedding_dim=32` | 0.789 | 0.855 | 0.334 | 0.812 | 0.697 |
| `d64` | ConvAE | 64 | 5,000 | `model.embedding_dim=64` | 0.816 | 0.859 | 0.380 | 0.901 | 0.739 |
| `d128` | ConvAE | 128 | 5,000 | `model.embedding_dim=128` | 0.871 | 0.905 | 0.411 | 0.922 | 0.777 |
| `d256` | ConvAE | 256 | 5,000 | — (config default) | 0.888 | 0.913 | 0.435 | 0.931 | 0.792 |
| `d256_seed1` | ConvAE | 256 | 5,000 | `training.seed=1` | 0.885 | 0.905 | 0.418 | 0.928 | 0.784 |
| `d512_w64` | ConvAE | 512 | 5,000 | `model.embedding_dim=512` `model.width=64` | 0.926 | 0.928 | 0.415 | 0.944 | **0.803** |
| `handcrafted` | 55 per-tile statistics, no training | 55 | — | — | 0.914 | 0.923 | **0.516** | 0.886 | **0.810** |
| `ae-d512` | ConvAE | 512 | 20,000 | `model.embedding_dim=512` `model.width=64` | | | | | *queued* |
| `ijepa-s16` | I-JEPA ViT-S/16 | 768 | 50,000 | — (config default) | | | | | *queued* |
| `ijepa-s16-cache` | I-JEPA ViT-S/16 | 768 | 150,000 | `data.cache=ram_uint8` | | | | | *queued* |
| `ijepa-ti16-cache` | I-JEPA ViT-Ti/16 | 384 | 200,000 | `model.arch=vit_tiny` `model.pred_emb_dim=96` `data.cache=ram_uint8` | | | | | *queued* |
| `ijepa-b16-cache` | I-JEPA ViT-B/16 | 1536 | 70,000 | `model.arch=vit_base` `model.pred_emb_dim=384` `model.pred_num_heads=6` `data.cache=ram_uint8` | | | | | *queued* |
| `ijepa-s16-lr15` | I-JEPA ViT-S/16 | 768 | 150,000 | `data.cache=ram_uint8` `training.lr=1.5e-4` `training.start_lr=3.0e-5` | | | | | *queued* |
| `fused` | best I-JEPA ⊕ `handcrafted` | +55 | — | `concat_embeddings.py` | | | | | *after the above* |

## Notes

- **`d32`–`d512_w64`** — the dimension sweep (`sweep_dim.sh`), all at 5,000
  steps rather than the config's 20,000. Scores rise steadily with D and show no
  overfitting penalty even at 512, despite the probes being unregularised.
- **`d256_seed1`** — a seed repeat, run only to measure run-to-run noise. That is
  where the ±0.008 threshold above comes from.
- **`d512_w64`** — width and D moved together, so the two effects are not
  separable from this row alone.
- **`handcrafted`** — the bar to beat. Untrained, 55 dimensions, and it wins
  overall. Its margin is entirely on `task_6` (+0.10 over the best autoencoder)
  while it loses `task_7` (−0.06), which is what makes fusion worth measuring.
- **`task_6` plateaus at ≈0.42 for every D ≥ 128** while the regressions keep
  climbing. The limit is the objective, not the capacity — this is the
  hypothesis I-JEPA is meant to test, so report `task_6` separately rather than
  only the mean.
- **Each I-JEPA row is really four.** Pooling lives in `model_cfg`, so one
  training run writes `best_mean.pt`, `best_meanstd.pt`, `best_mean4.pt` and
  `best_mean4std.pt` from the same weights, and the job scores all of them as
  `<run>_mean`, `<run>_meanstd`, … The **D** column above is the `mean+std`
  default; the others are ×0.5, ×2 and ×4 of it. ViT-B's `mean4std` would be
  6,144 wide against the 4,096 submission cap, so it is skipped automatically.
  Report whichever variant wins, and say which it was.

## The first sweep

Submitted 2026-10-05 18:30 (`./sweep_ijepa.sh`), six jobs, one question each.

| Job | Run | Question it answers |
|---|---|---|
| 57375250 | `ijepa-s16` | Does latent prediction beat reconstruction at all? Also measures the true step rate. |
| 57375683 | `ijepa-s16-cache` | Sample-limited or capacity-limited? The RAM cache removes the I/O bottleneck. |
| 57375252 | `ijepa-ti16-cache` | Is ViT-S already too big for 100k tiles? |
| 57375253 | `ijepa-b16-cache` | The other direction — worth a slot only with the cache on. |
| 57375684 | `ijepa-s16-lr15` | The least-confident hyperparameter. 3.0e-4 is sqrt-scaled from the reference's batch-2048 peak, which is a rule of thumb, not a measurement. |
| 57375685 | `ae-d512` | The fair comparator. Every autoencoder row above is at 5,000 steps, so without this "I-JEPA beats the AE" compares against an undertrained baseline. |

**gpuhost004 took three of these and failed the GPU preflight on all three**
(jobs 57375251, 57375254, 57375255, each dead in 2–7 s). `nvidia-smi -L` printed
the A100's UUID while `torch.cuda.device_count()` returned 0 — the same fault
that cost a 12 h slot on gpuhost005 earlier. They were resubmitted with
`EXCLUDE=gpuhost004`, which is why three job IDs above are from the second
batch. Worth reporting to JASMIN support; the node will keep accepting jobs and
handing the slot straight back.

Note the orchid QOS allows **four concurrent jobs per user**, so a six-run
sweep runs in two waves.

Two scheduling notes, both of which shaped the numbers above:

- **A maintenance reservation (`Oct2026PatchDay`, 06 Oct 05:00–23:00) covers
  every orchid GPU host.** A 12 h job will not start within 12 h of a
  reservation — SLURM parks it until the window *ends* — so these run at
  `--time=09:00:00` with `training.max_seconds=25200`. Check
  `scontrol show reservation` before a sweep and pass `WALLTIME=`/`MAX_SECONDS=`
  to suit.
- **Step budgets are set to finish, not to fill the slot.** The LR, weight-decay
  and EMA schedules are all defined over the *total* step count, so a run cut
  off by `max_seconds` never anneals and underperforms badly rather than
  slightly.

**Measured once the sweep was running, replacing the estimates it was sized on:**

| | estimated | measured |
|---|---|---|
| ViT-S/16 uncached, 14 workers | 3.5 steps/s | **6.05 steps/s** (774 tiles/s) |
| cache pre-pass | ~258 tiles/s, ~5 min | **~490 tiles/s, ~3.5 min** |

So the loader was about twice as fast as the 8-worker benchmark suggested, and
the I/O bottleneck is roughly 3× rather than 8×. The cache is therefore a
smaller lever than projected — still worth the two runs testing it, but the
uncached configuration is not as starved as it looked. `ijepa-s16` will finish
50,000 steps in about 2.3 h, so the next sweep can afford materially more steps.

**One thing to watch:** `collapse/effective_rank` is around 5 on a batch of 128
at step 1,000. That is lower than a ViT on natural images would show, and it is
the metric to check first if the probe scores disappoint — though MODIS cloud
scenes genuinely have fewer degrees of freedom than photographs, and the
autoencoder reaches R² ≈ 0.9 on some tasks from 256 dimensions, so low rank is
not by itself a failure. `feature_std` 0.52 and `token_std` 0.29 are healthy.

## Adding a row

```bash
# 1. train (wandb run name = the --out basename)
sbatch train_ijepa.sbatch                        # or: --out runs/<name> --wandb

# 2. embed the validation split under the same name
uv run python -m iecdt_hackathon.embed \
  --checkpoint runs/<name>/best_meanstd.pt \
  --data-dir /gws/ssde/j25b/iecdt/modis_hackathon/val \
  --out embeddings/val --name <name> --overwrite

# 3. score it next to whatever you are comparing against
uv run python -m iecdt_hackathon.evaluate --out results/<name> \
  --embeddings embeddings/val/<name>.npz embeddings/val/d256.npz
```

`--overwrite` in step 2 is not optional: `embed.py` keeps an existing `.npz`
rather than recomputing it, so without it a retrained model is silently scored
on its predecessor's embeddings.

Then copy the four per-task numbers and the overall into the table. Runs log to
wandb live by default; only a run forced to `wandb_mode: offline` needs a push
afterwards, and it leaves the command in `runs/<name>/wandb_sync.txt`.
