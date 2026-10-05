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
| `baseline` | ConvAE | 256 | 20,000 | — (config default) | | | | | *not yet run* |
| `ijepa` | I-JEPA ViT-S/16, mean+std | 768 | 60,000 | — (config default) | | | | | *not yet run* |
| `ijepa_mean` | I-JEPA ViT-S/16, mean | 384 | 60,000 | same run, `pool=mean` | | | | | *not yet run* |
| `ijepa_mean4` | I-JEPA ViT-S/16, mean of last 4 | 1536 | 60,000 | same run, `pool=mean_last4` | | | | | *not yet run* |
| `ijepa_mean4std` | I-JEPA ViT-S/16, mean+std, last 4 | 3072 | 60,000 | same run, `pool=mean_last4_std` | | | | | *not yet run* |
| `fused` | `ijepa` ⊕ `handcrafted` | 823 | — | `concat_embeddings.py` | | | | | *not yet run* |

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
- **The four `ijepa_*` rows come from one training run.** Pooling lives in
  `model_cfg`, so training writes `best_mean.pt`, `best_meanstd.pt`,
  `best_mean4.pt` and `best_mean4std.pt` from the same weights, and
  `evaluate.py` ranks them together. `mean4std` at D = 3072 is well past
  anything the sweep probed, so it is the one variant that may regress.

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

Then copy the four per-task numbers and the overall into the table, and push the
offline wandb run from a **login** node:

```bash
uv run wandb sync runs/<name>/wandb/offline-run-*
```
