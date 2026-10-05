<p align="center">
  <img src="IE_Hackathon_2026_logo.png" alt="IE Hackathon 2026 logo" width="320">
</p>

<h1 align="center">Cloud embeddings from MODIS</h1>

<p align="center">
  Learning general-purpose representations of satellite cloud scenes for the<br>
  Intelligent Earth CDT Hackathon 2026
</p>

---

## About this repository

This is my working repository for the IECDT Hackathon 2026. It started from the organisers' starter code (see [Acknowledgements](#acknowledgements)) and is where I develop, run and keep track of my own embedding models.

The challenge in one sentence: build a model that turns a 256 × 256 km MODIS satellite tile into a compact vector, such that physical cloud properties can be read off that vector with a **linear** model.

## The challenge

**Input.** Aqua/MODIS Level-1B tiles, 256 × 256 pixels at roughly 1 km resolution, with six radiance channels:

| Channels | What they see |
|---|---|
| `Rad_1`, `Rad_3`, `Rad_4` | Visible red, blue, green (0.645, 0.469, 0.555 µm): cloud brightness and texture |
| `Rad_29`, `Rad_31`, `Rad_32` | Thermal infrared (8.55, 11.03, 12.02 µm): cloud-top temperature, and so height |

Each tile also carries `solar_zenith_angle` and a `land_mask`, which models may optionally use.

**Output.** One embedding vector per tile, of any width between 1 and 4,096.

**Scoring.** For each of ten downstream tasks, a linear probe (linear regression, or logistic regression for the classification task) is fitted on the embeddings with 5-fold cross-validation. Regression tasks are scored by R² and the classification task by macro-F1, and the leaderboard ranks submissions by their mean across tasks.

Only four of the ten tasks have published labels, and what each task physically measures is deliberately withheld:

| Task | Type | Notes |
|---|---|---|
| `task_4`, `task_5`, `task_7` | Regression | Standardised to zero mean, unit variance; left-skewed |
| `task_6` | 10-class classification | Highly imbalanced; the four rarest classes make up under 1.5% of tiles |
| `task_1`–`task_3`, `task_8`–`task_10` | Withheld | Scored on the leaderboard only |

Because most of the leaderboard is hidden, the aim is a general-purpose representation rather than one tuned to the four visible tasks.

## Data

Everything lives on JASMIN in the `iecdt` group workspace:

```
/gws/ssde/j25b/iecdt/modis_hackathon/
├── train/        100,000 tiles   2003–2010   (no labels)
├── val/           10,000 tiles   2015–2019   (labels for tasks 4–7)
├── test/          10,000 tiles   2020–2025   (embed and submit these)
├── labels/       val_labels.nc
└── stats/        modis_band_stats.json      per-band min/max for normalisation
```

The three splits cover separate time periods, so embeddings need to generalise across years.

## Workflow

### 1. Train

```bash
uv run python -m iecdt_hackathon.train --out runs/baseline
```

Any config value can be overridden from the command line:

```bash
uv run python -m iecdt_hackathon.train --out runs/d128 \
  --set model.embedding_dim=128 --set training.steps=5000
```

Training runs for a fixed number of steps on a cosine learning-rate schedule. To shorten a run, lower `training.steps` rather than stopping it partway. The validation loss printed during training is **reconstruction error**, which checks that training is working but is not the leaderboard metric.

### 2. Embed

```bash
uv run python -m iecdt_hackathon.embed \
  --checkpoint runs/baseline/best.pt \
  --data-dir /gws/ssde/j25b/iecdt/modis_hackathon/val \
  --out embeddings/val --name baseline
```

`--limit N` embeds only the first N tiles, which is useful for quick checks.

### 3. Evaluate locally

```bash
uv run python -m iecdt_hackathon.evaluate --embeddings embeddings/val/baseline.npz
```

Several embeddings can be compared side by side using `NAME=PATH` pairs:

```bash
uv run python -m iecdt_hackathon.evaluate \
  --embeddings base=embeddings/val/baseline.npz new=embeddings/val/new.npz
```

### 4. Submit

Embed the **test** split, then submit from a JASMIN **sci server** over SSH.

```bash
uv run python -m iecdt_hackathon.embed \
  --checkpoint runs/baseline/best.pt \
  --data-dir /gws/ssde/j25b/iecdt/modis_hackathon/test \
  --out embeddings/test --name myteam

# then, on a sci server:
./submit.sh embeddings/test/myteam.npz
```

Each team gets four scored submissions per day, and the leaderboard keeps each team's best result. A valid submission is a `.npz` containing `embeddings` (shape `(10000, D)`, finite float32) and `tile_index` (shape `(10000,)`, int64), covering every test tile exactly once.

## Experiments

| Script | What it does |
|---|---|
| `sweep_dim.sh` | Trains, embeds and evaluates the baseline autoencoder at embedding sizes 32–512, plus a repeat seed to measure run-to-run noise. Run with `STEPS=5000 nohup ./sweep_dim.sh > sweep_dim.log 2>&1 &` |

### Log

| Run | Change from baseline | Steps | task_4 R² | task_5 R² | task_6 F1 | task_7 R² | Notes |
|---|---|---|---|---|---|---|---|
| `baseline` | — | 20,000 | | | | | Reference |
| `dim_sweep` | `embedding_dim` 32–512 | 5,000 | | | | | In progress |

## Repository layout

```
iecdt_hackathon/
  data.py               ModisTileDataset and dataloader
  models.py             ConvAutoencoder baseline (any model must expose .encode)
  train.py              training loop, config-driven
  embed.py              model -> .npz embedding file for a split
  embeddings.py         submission format and validity checks
  evaluate.py           linear probes on validation tasks
  tasks.py              task definitions
  ranking.py            per-task scores -> leaderboard ranking
  tile_layout.py        flat or sharded tile paths
  print_leaderboard.py  current standings in the terminal
configs/default.yaml    baseline configuration
sweep_dim.sh            embedding-size sweep
submit.sh               submit test embeddings to the leaderboard
```

## Acknowledgements

The starter code, data pipeline and evaluation framework were written by **Tim Reichelt** (University of Oxford) for the IECDT Hackathon, and are published at [treigerm/iecdt_hackathon_climate](https://github.com/treigerm/iecdt_hackathon_climate). This repository builds on that work.

Compute and storage are provided by [JASMIN](https://jasmin.ac.uk), the UK's collaborative data analysis environment. MODIS data are from NASA's Aqua satellite. The hackathon is part of the Intelligent Earth CDT at the University of Oxford.

## Licence

My own additions are released under the MIT licence; see [LICENSE](LICENSE). The original starter code was published without a licence and remains the copyright of its author; see the note in [LICENSE](LICENSE).
