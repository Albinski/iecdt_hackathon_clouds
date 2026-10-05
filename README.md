# IECDT Hackathon: MODIS tile embeddings

This repository supplies you with the basic functionality to train a convolutional
auto-encoder on MODIS satellite imagery, extract latent embeddings on a validation set, 
and evaluate linear probes on four validation tasks.

**Hackathon goal:** Develop your own embedding model which achieves the highest skill
on extracting cloud properties with a linear probe from the embeddings. This requires 
you to generate embeddings for the test MODIS tiles and submit them in the form of 
a `*.npz` file through a mechanism described below. **To be able to make a submission 
each team needs to send an email to tim.reichelt@physics.ox.ac.uk with the JASMIN
usernames of all team members and a team name.** 

The code provided in this repository is meant to merely provide a guidance for how 
to use the supplied data and is quite heavily Claude-generated. I'm expecting that 
most teams will want to build their own repository for training their models. The 
key bits of the repository are:
- The PyTorch dataset `ModisTileDataset` at `iecdt_hackathon/data.py` that describes 
  how to load the data
- The embedding script at `iecdt_hackathon/embed.py` that shows you how to format the 
  embedding file. If you develop your own model this script won't work anymore but it 
  shows you how to format the `*.npz` file for submission.
- Evaluation of validation embeddings on 4 validation tasks with `iecdt_hackathon/evaluate.py`.
- The `submit.sh` file that you'll use to submit your test embeddings for scoring.

## Quickstart

```bash
uv sync                                    # or: pip install -e .
uv run python -m iecdt_hackathon.train     # train the baseline autoencoder

ROOT=/gws/ssde/j25b/iecdt/modis_hackathon

# embed validation split, then evaluate on validation tasks
uv run python -m iecdt_hackathon.embed \
    --checkpoint runs/baseline/best.pt \
    --data-dir $ROOT/val --out embeddings/val 
uv run python -m iecdt_hackathon.evaluate --embeddings embeddings/val/*.npz

# embed test split and then submit for evaluation
uv run python -m iecdt_hackathon.embed \
    --checkpoint runs/baseline/best.pt \
    --data-dir $ROOT/test --out embeddings/test --name yourteam
./submit.sh embeddings/test/yourteam.npz
```

## The data

All the data for the hackathon is stored at `/gws/ssde/j25b/iecdt/modis_hackathon`:
```
train/       100,000 tiles (2003–2010)
val/          10,000 tiles (2015-2019)
test/         10,000 tiles (2020-2025)
labels/    val_labels.nc 
stats/     modis_band_stats.json
```

The `ModisTileDataset` at `iecdt_hackathon/data.py` provides you with a PyTorch 
`Dataset` class to load the data into a training loop. Each tile is a 256×256 crop 
of an Aqua/MODIS L1B granule, ~1 km per pixel:

| variable | meaning |
|---|---|
| `Rad_1`, `Rad_3`, `Rad_4` | visible red / blue / green (0.645, 0.469, 0.555 µm) |
| `Rad_29`, `Rad_31`, `Rad_32` | thermal infrared (8.55, 11.03, 12.02 µm) |
| `solar_zenith_angle` | degrees |
| `land_mask` | 0 = sea, 1 = land |

The six `Rad_{ix}` bands are what your model needs to encode to an embedding.
You are not required to use `solar_zenith_angle` or `land_mask` but it might provide 
helpful metadata.

`ModisTileDataset` min-max normalises each band to roughly [0, 1] using
`stats/modis_band_stats.json`.

## Generating embeddings

The `iecdt_hackathon/embed.py` provides an example with how to generate embeddings
from a checkpoint of the baseline model. For the test tiles this can be done with
```bash
uv run python -m iecdt_hackathon.embed \
    --checkpoint runs/baseline/best.pt \
    --data-dir $ROOT/test --out embeddings/test --name yourteam
```
Note that the script assumes a checkpoint from the baseline auto-encoder architecture 
in `iecdt_hackathon/train.py`. If you use your own model architecture you need to 
adapt the embedding script accordingly.

### What the `.npz` has to contain

The file is read with `np.load(path, allow_pickle=False)`, so write it with
`np.savez_compressed`. It must hold two arrays:

| array | shape | dtype | |
|---|---|---|---|
| `embeddings` | `(10000, D)` | float32 | one row per held-out tile (float64 is read too) |
| `tile_index` | `(10000,)` | int64 | which tile each row belongs to |

and satisfy all of the following. Each one is checked before any probe is
fitted:

- **Exactly the held-out split: all 10,000 tiles, nothing else.** For `test/`
  that is `tile_index` covering `0 … 9999` once each. A missing tile is refused
  (every submission is scored on the same rows, so never pass `--limit` to the
  file you send), and so is a tile that is not in the split — which is what
  embedding `val/` by mistake looks like.
- **`tile_index` is the tile's own index, not the row number.** It is the
  integer in the tile's filename (`4321.nc` → `4321`), which is what
  `ModisTileDataset` hands you as `batch["tile_index"]`; so **row order does not matter**.
- **No repeated index.** A duplicate is the one error that could put a single
  tile in both a probe's fitting folds and the fold it is scored on, so it is
  refused outright.
- **Every value finite.** One `NaN` or `inf` anywhere in `embeddings` and no
  probe can be fitted.
- **`1 ≤ D ≤ 4096`**, the same `D` for every row (so `embeddings` is strictly
  2-D). At the top of that range an "embedding" is a copy of the tile rather
  than a representation of it.
- **At most 256 MB on disk.** A 256-dimensional submission is about 10 MB and
  even a 4,096-dimensional one is under 170 MB, so this only catches mistakes.

### How to submit

You can submit embeddings with the following command:
```bash
./submit.sh embeddings/test/yourteam.npz
```

**You are identified by the account you submit from**, so the filename is ignored. 
As mentioned at the top tell me every JASMIN account your team will use so that I can
whitelist these accounts for submission. Submissions from non-whitelisted accounts 
will fail.

Your result is appended to `/gws/ssde/j25b/iecdt/modis_hackathon/submissions/feedback/<your team>.md` and should 
appear within a few minutes (email me if there are any issues), and the standings are in 
`/gws/ssde/j25b/iecdt/modis_hackathon/leaderboard/leaderboard.md`. You can write into the drop directory but not read it, so
nobody sees anyone else's embeddings.

**Your row is your best submission, not your latest**, a worse attempt is
reported to you and leaves the standings alone.

**Four scored submissions per team per day**, resetting at midnight UK time and
counted per team however many of your accounts submit. If an error occurs during the scoring of your 
submission it won't count towards your submission limit. If a submission is refused, 
the feedback file should tell you why. Some potential reasons include:

| | |
|---|---|
| not a `.npz`, or unreadable | check it arrived whole and is the file `embed.py` wrote |
| does not cover the held-out split | you passed `--limit`, or embedded `val` |
| too wide | over 4,096 dimensions; pool or project your output |
| not registered | that account is not on your team's list |
| out of submissions today | nothing was scored; resubmit after midnight UK time |

### Watching the leaderboard

To see the current leaderboard of submissions run:
```bash
uv run python -m iecdt_hackathon.print_leaderboard              # both tables
```

## Scoring locally

Running
```bash
uv run python -m iecdt_hackathon.evaluate --embeddings embeddings/val/*.npz
```
fits linear probes on your embeddings for 4 different tasks.


## Repository layout

```
iecdt_hackathon/
  data.py               ModisTileDataset, build_dataloader
  models.py             ConvAutoencoder
  train.py              training loop
  tasks.py              the downstream task registry
  embed.py              run your model over a split -> an embedding file
  embeddings.py         the .npz submission format, and its validity checks
  evaluate.py           linear probes over an embedding file
  ranking.py            per-task scores -> one global order
  tile_layout.py        flat/sharded tile paths
  print_leaderboard.py  pretty-print the standings in the console
configs/                training configuration
submit.sh               copy an embedding file into the drop directory
```
