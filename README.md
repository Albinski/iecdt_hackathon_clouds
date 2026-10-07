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

| Run | Change from baseline | Steps | task_4 R² | task_5 R² | task_6 F1 | task_7 R² | Overall | Notes |
|---|---|---|---|---|---|---|---|---|
| `baseline` | — | 20,000 | 0.887 | 0.913 | 0.422 | 0.931 | 0.788 | |
| `d32` | `embedding_dim` 32 | 5,000 | 0.789 | 0.855 | 0.334 | 0.812 | 0.697 | |
| `d64` | `embedding_dim` 64 | 5,000 | 0.816 | 0.859 | 0.380 | 0.901 | 0.739 | |
| `d128` | `embedding_dim` 128 | 5,000 | 0.871 | 0.905 | 0.411 | 0.922 | 0.777 | |
| `d256` | default config (seed 0) | 5,000 | 0.888 | 0.913 | 0.435 | 0.931 | 0.792 | |
| `d256_seed1` | default config (seed 1) | 5,000 | 0.885 | 0.905 | 0.418 | 0.928 | 0.784 | Seed noise ≈ 0.008 overall, 0.017 on task 6 |
| `d512_w64` | `embedding_dim` 512, `width` 64 | 5,000 | 0.926 | 0.928 | 0.415 | 0.944 | 0.803 | Best so far; regression up, task 6 flat. Width and dim changed together |
| `handcrafted` | 55 per-tile statistics, no training | — | 0.914 | 0.923 | 0.516 | 0.886 | 0.810 | Best overall so far; beats AE on task 6 by +0.10, loses on task 7 |
| `physical` | 144 physical features (`hand` + 89 new), no training | — | 0.949 | 0.958 | 0.547 | 0.960 | 0.853 | Beats every autoencoder on every task; +0.03 on task 6 over `hand` |
| `ae512_phys` | `d512_w64` + `physical` stacked (656 dims) | 5,000 | 0.954 | 0.968 | 0.553 | 0.969 | 0.861 | Best on val so far; submitted as submission 1 |
| `contr_loss` | `d512_w64` + contrastive loss | 5,000 | 0.476 | 0.817 | 0.392 | 0.905 | 0.647 | Worse on all metrics :P |

**Dimension sweep takeaways:** Scores rise steadily with embedding size up to 256, with diminishing returns (+0.04 from 32 → 64, about +0.01 from 128 → 256). Widening the encoder to 64 at D = 512 improves the regression tasks well beyond seed noise but leaves task 6 unchanged. Task 6 plateaus at a macro-F1 of about 0.42 for all D ≥ 128, which suggests the reconstruction objective, not the embedding size, is what limits the classification task.

**Physical features takeaways:** Converting radiances to reflectance and brightness temperature, and adding a reflectance × temperature regime histogram, height relative to the local sea surface, cirrus/phase band differences, cloud-object statistics and multiscale texture, lifts every task: 144 untrained features (0.853) beat every autoencoder trained so far. Stacking them with the 512-dim autoencoder adds a further, smaller gain (0.861). Task 6 is still limited by its rare classes: in the combined model class 4 gets 1 of 20 tiles right, class 6 gets 11 of 41 and class 9 gets 17 of 43. Its balanced accuracy is lower than `physical` alone (0.57 vs 0.62), so stacking traded some rare-class recall for accuracy on the common classes.

### Submissions

| # | Date | Embedding | Dims | Test overall (10 tasks) | Rank | Notes |
|---|---|---|---|---|---|---|
| 1 | 2026-10-05 | `ae512_physical` (`d512_w64` + `physical`) | 656 | 0.780 | 1 of 3 | First submission |

#### Submission 1: per-task scores

| Task | Metric | Test | Val (5-fold CV) |
|---|---|---|---|
| task_1 | R² | 0.932 | hidden |
| task_2 | macro-F1 | 0.605 | hidden |
| task_3 | R² | 0.682 | hidden |
| task_4 | R² | 0.949 | 0.954 |
| task_5 | R² | 0.968 | 0.968 |
| task_6 | macro-F1 | 0.541 | 0.553 |
| task_7 | R² | 0.961 | 0.969 |
| task_8 | R² | 0.882 | hidden |
| task_9 | R² | 0.629 | hidden |
| task_10 | R² | 0.651 | hidden |

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

### Use of AI tools

Claude (Anthropic) was used to help with environment setup, experiment scripts and documentation. All code was reviewed and tested, and the experimental design and interpretation are my own.

## Licence

My own additions are released under the MIT licence; see [LICENSE](LICENSE). The original starter code was published without a licence and remains the copyright of its author; see the note in [LICENSE](LICENSE).
