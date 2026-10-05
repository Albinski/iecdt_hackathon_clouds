"""Explore what the probe tasks measure, and build a hand-crafted baseline.

For every tile in a split this computes simple, interpretable statistics
(per-band percentiles, texture, IR band differences, land fraction, sun angle)
and saves them as an embedding file that evaluate.py can score directly.

With labels (val split) it also prints:
  * the features most correlated with each regression task
  * per-class medians of key features for the classification task
and saves image galleries of example tiles from every task_6 class.

Usage (from the repo root):
    uv run python scripts/explore_tasks.py                    # val: features + diagnostics
    uv run python scripts/explore_tasks.py --split test --no-analysis   # test features only
    uv run python -m iecdt_hackathon.evaluate --embeddings embeddings/val/handcrafted.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy.stats import spearmanr

from iecdt_hackathon.data import DEFAULT_BANDS, discover_tile_indices, read_radiances
from iecdt_hackathon.embeddings import save_embeddings
from iecdt_hackathon.tile_layout import read_layout, tile_path

ROOT = "/gws/ssde/j25b/iecdt/modis_hackathon"
PCTS = (5, 25, 50, 75, 95)


def feature_names(bands=DEFAULT_BANDS):
    names = []
    for b in bands:
        names += [f"Rad_{b}_mean", f"Rad_{b}_std"]
        names += [f"Rad_{b}_p{p}" for p in PCTS]
        names += [f"Rad_{b}_grad"]  # mean absolute spatial gradient = texture
    names += ["d31_32_mean", "d31_32_std", "d29_31_mean", "d29_31_std"]
    names += ["land_frac", "cos_sza_mean", "nan_frac"]
    return names


def tile_features(rad, land, sza):
    """rad: (n_bands, y, x) physical radiances. Returns a 1-D float32 vector."""
    nan_frac = np.isnan(rad).mean()
    feats = []
    for x in rad:
        feats += [np.nanmean(x), np.nanstd(x)]
        feats += list(np.nanpercentile(x, PCTS))
        gy = np.abs(np.diff(x, axis=0))
        gx = np.abs(np.diff(x, axis=1))
        feats += [0.5 * (np.nanmean(gy) + np.nanmean(gx))]
    r = dict(zip(DEFAULT_BANDS, rad))
    d31_32 = r["31"] - r["32"]
    d29_31 = r["29"] - r["31"]
    feats += [np.nanmean(d31_32), np.nanstd(d31_32), np.nanmean(d29_31), np.nanstd(d29_31)]
    feats += [float(np.mean(land)), float(np.nanmean(np.cos(np.deg2rad(sza)))), nan_frac]
    return np.nan_to_num(np.asarray(feats, dtype=np.float32))


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tile_indices):
        self.data_dir = data_dir
        self.tile_indices = tile_indices
        self.shard_size = read_layout(data_dir)

    def __len__(self):
        return len(self.tile_indices)

    def __getitem__(self, i):
        ix = self.tile_indices[i]
        with xr.open_dataset(tile_path(self.data_dir, ix, "", self.shard_size)) as ds:
            rad = read_radiances(ds, DEFAULT_BANDS)
            land = ds["land_mask"].values
            sza = ds["solar_zenith_angle"].values
        return ix, tile_features(rad, land, sza)


def _collate(batch):
    return batch


def compute_features(data_dir, tile_indices, num_workers):
    loader = torch.utils.data.DataLoader(
        FeatureDataset(data_dir, tile_indices),
        batch_size=64, num_workers=num_workers, collate_fn=_collate,
    )
    out_ix, out_f = [], []
    for k, batch in enumerate(loader):
        for ix, f in batch:
            out_ix.append(ix)
            out_f.append(f)
        if k % 20 == 0:
            print(f"  {len(out_ix):>6}/{len(tile_indices)} tiles", flush=True)
    return np.array(out_ix, dtype=np.int64), np.stack(out_f)


def inspect_one_tile(data_dir):
    """Print what a tile file contains, in case it carries useful metadata."""
    ix = discover_tile_indices(data_dir)[0]
    path = tile_path(data_dir, ix)
    print(f"\nExample tile {path}")
    with xr.open_dataset(path) as ds:
        print(ds)
    siblings = sorted(p.name for p in path.parent.glob(f"{ix}_*.nc"))
    print(f"Sibling files for tile {ix}: {siblings or 'none'}")


def regression_report(names, X, y, task, top=8):
    ok = ~np.isnan(y)
    rhos = []
    for j, n in enumerate(names):
        rho = spearmanr(X[ok, j], y[ok]).correlation
        rhos.append((0.0 if np.isnan(rho) else rho, n))
    rhos.sort(key=lambda t: -abs(t[0]))
    print(f"\n{task}: strongest Spearman correlations ({ok.sum()} tiles)")
    for rho, n in rhos[:top]:
        print(f"  {n:<16} {rho:+.3f}")


def class_report(names, X, y, task):
    show = ["Rad_1_mean", "Rad_1_std", "Rad_1_grad", "Rad_31_mean",
            "Rad_31_p5", "d31_32_mean", "land_frac", "cos_sza_mean"]
    cols = [names.index(s) for s in show]
    ok = y >= 0
    print(f"\n{task}: per-class medians "
          f"(Rad_1 = visible brightness; Rad_31 = IR, lower = colder/higher cloud)")
    header = f"{'class':>5} {'n':>6} " + " ".join(f"{s:>12}" for s in show)
    print(header)
    for c in np.unique(y[ok]):
        m = y == c
        med = np.median(X[m][:, cols], axis=0)
        print(f"{c:>5} {m.sum():>6} " + " ".join(f"{v:>12.4g}" for v in med))


def gallery(data_dir, tile_ix, y, task, out_dir, per_class=6, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    shard = read_layout(data_dir)
    classes = [c for c in np.unique(y) if c >= 0]
    picks = {c: rng.choice(tile_ix[y == c], min(per_class, (y == c).sum()), replace=False)
             for c in classes}

    imgs = {}
    for c, ixs in picks.items():
        for ix in ixs:
            with xr.open_dataset(tile_path(data_dir, int(ix), "", shard)) as ds:
                imgs[int(ix)] = read_radiances(ds, DEFAULT_BANDS)
    allpix = np.stack(list(imgs.values()))  # (n, bands, y, x)
    lo = np.nanpercentile(allpix, 1, axis=(0, 2, 3))
    hi = np.nanpercentile(allpix, 99, axis=(0, 2, 3))

    def scale(x, b):
        return np.clip(np.nan_to_num((x - lo[b]) / (hi[b] - lo[b])), 0, 1)

    for kind in ("rgb", "ir"):
        fig, axes = plt.subplots(len(classes), per_class,
                                 figsize=(2 * per_class, 2 * len(classes)), squeeze=False)
        for r, c in enumerate(classes):
            for k in range(per_class):
                ax = axes[r, k]
                ax.set_xticks([]); ax.set_yticks([])
                if k >= len(picks[c]):
                    ax.axis("off"); continue
                rad = imgs[int(picks[c][k])]
                if kind == "rgb":  # bands 1, 4, 3 = red, green, blue
                    img = np.stack([scale(rad[0], 0), scale(rad[2], 2), scale(rad[1], 1)], -1)
                    ax.imshow(img ** 0.6)
                else:  # band 31, inverted so cold (high) cloud is white
                    ax.imshow(1 - scale(rad[4], 4), cmap="gray", vmin=0, vmax=1)
                if k == 0:
                    ax.set_ylabel(f"class {c}\n(n={(y == c).sum()})")
        fig.suptitle(f"{task} examples: {'true colour' if kind == 'rgb' else 'IR 11 µm, cold = white'}")
        fig.tight_layout()
        path = Path(out_dir) / f"{task}_gallery_{kind}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--root", default=ROOT)
    p.add_argument("--labels", default=f"{ROOT}/labels/val_labels.nc")
    p.add_argument("--out", default="embeddings", help="writes <out>/<split>/handcrafted.npz")
    p.add_argument("--results", default="results/explore")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-analysis", action="store_true", help="features only, skip label diagnostics")
    args = p.parse_args()

    data_dir = f"{args.root}/{args.split}"
    names = feature_names()
    analyse = args.split == "val" and not args.no_analysis

    labels = None
    if analyse:
        labels = xr.open_dataset(args.labels).load()
        tile_indices = [int(i) for i in labels["tile_index"].values]
        inspect_one_tile(data_dir)
    else:
        tile_indices = discover_tile_indices(data_dir)

    print(f"\nComputing {len(names)} features for {len(tile_indices)} {args.split} tiles")
    tile_ix, X = compute_features(data_dir, tile_indices, args.num_workers)
    order = np.argsort(tile_ix)
    tile_ix, X = tile_ix[order], X[order]

    out_path = Path(args.out) / args.split / "handcrafted.npz"
    save_embeddings(out_path, X, tile_ix, model="handcrafted", features=",".join(names))
    print(f"Wrote {out_path}  ({X.shape[0]} tiles x {X.shape[1]} features)")

    if not analyse:
        return

    Path(args.results).mkdir(parents=True, exist_ok=True)
    pos = {int(i): k for k, i in enumerate(labels["tile_index"].values)}
    rows = np.array([pos[int(i)] for i in tile_ix])
    for task in ("task_4", "task_5", "task_7"):
        regression_report(names, X, labels[task].values[rows].astype(float), task)
    y6 = labels["task_6"].values[rows].astype(int)
    class_report(names, X, y6, "task_6")
    gallery(data_dir, tile_ix, y6, "task_6", args.results)


if __name__ == "__main__":
    main()
