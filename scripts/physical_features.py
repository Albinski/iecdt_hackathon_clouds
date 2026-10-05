"""Physically motivated hand-crafted embeddings (v2).

Builds on the 55 statistics in explore_tasks.py and adds features aimed at what
the task_6 galleries show:

  * Radiance -> physics: visible reflectance (sun-angle corrected) and
    brightness temperature (inverse Planck) for the thermal bands.
  * Cloud-regime histogram: share of pixels in each reflectance x 11 um
    brightness-temperature bin (7 x 7), the quantity regime schemes cluster on.
  * Height relative to the local surface: warmest pixels approximate the sea
    surface, so (T_surface - T) separates high cloud from cold high-latitude
    scenes where the whole tile is cold.
  * Cirrus and phase: 11-12 um split window (thin cirrus) and 8.5-11 um
    difference (ice vs liquid).
  * Organisation: cloud fraction, number and size of cloud objects, for bright
    and for cold cloud separately (open/closed cells, scattered convection).
  * Multiscale texture: variability of block means at 4, 16 and 64 pixels.

Usage (from the repo root):
    uv run python scripts/physical_features.py --split val
    uv run python scripts/physical_features.py --split test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from explore_tasks import feature_names as base_names, tile_features as base_features  # noqa: E402

from iecdt_hackathon.data import DEFAULT_BANDS, discover_tile_indices, read_radiances
from iecdt_hackathon.embeddings import save_embeddings
from iecdt_hackathon.tile_layout import read_layout, tile_path

ROOT = "/gws/ssde/j25b/iecdt/modis_hackathon"

# MODIS band centres (um) for inverse Planck, and approximate band-1 solar irradiance
WAVELENGTH = {"29": 8.55, "31": 11.03, "32": 12.02}
E0_BAND1 = 1605.0  # W m-2 um-1; approximate, only sets the reflectance scale
C1 = 1.191042e8    # W um4 m-2 sr-1
C2 = 1.4387752e4   # um K

R_EDGES = [0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, np.inf]
T_EDGES = [-np.inf, 220, 235, 250, 262, 272, 282, np.inf]
DT_EDGES = [-np.inf, 2, 5, 10, 20, 35, 50, np.inf]
BLOCKS = (4, 16, 64)
R_PCTS = (5, 25, 50, 75, 95)


def brightness_temperature(rad, band):
    lam = WAVELENGTH[band]
    with np.errstate(invalid="ignore", divide="ignore"):
        rad = np.where(rad > 0, rad, np.nan)
        return C2 / (lam * np.log1p(C1 / (lam**5 * rad)))


def _safe_pct(x, q):
    x = x[np.isfinite(x)]
    return np.percentile(x, q) if x.size else np.full(np.shape(q), np.nan)


def _objects(mask):
    """Fraction, object count, mean object size, largest-object share."""
    frac = mask.mean()
    labels, n = ndimage.label(mask)
    if n == 0:
        return [frac, 0.0, 0.0, 0.0]
    sizes = np.bincount(labels.ravel())[1:]
    return [frac, np.log1p(n), np.log1p(sizes.mean()), sizes.max() / max(mask.sum(), 1)]


def _multiscale_std(x):
    x = np.nan_to_num(x, nan=np.nanmean(x) if np.isfinite(x).any() else 0.0)
    total = x.std() + 1e-6
    out = []
    for b in BLOCKS:
        h, w = (x.shape[0] // b) * b, (x.shape[1] // b) * b
        blocks = x[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))
        out.append(blocks.std() / total)
    return out


def physical_names():
    names = [f"R_p{p}" for p in R_PCTS] + ["R_mean", "R_std"]
    names += [f"hist_R{i}_T{j}" for i in range(len(R_EDGES) - 1) for j in range(len(T_EDGES) - 1)]
    names += ["T_surface", "T31_p1", "T31_p5", "T31_p50"]
    names += [f"dT_bin{k}" for k in range(len(DT_EDGES) - 1)]
    names += ["sw_mean", "sw_p50", "sw_p90", "sw_frac_gt1p5",
              "ph_mean", "ph_p10", "ph_p90", "ph_frac_gt0"]
    for m in ("bright", "cold"):
        names += [f"{m}_frac", f"{m}_log_nobj", f"{m}_log_meansize", f"{m}_largest"]
    names += [f"R_ms{b}" for b in BLOCKS] + [f"T31_ms{b}" for b in BLOCKS]
    return names


def physical_features(rad, sza):
    r = dict(zip(DEFAULT_BANDS, rad))
    mu = np.clip(np.cos(np.deg2rad(sza)), 0.1, 1.0)
    R = np.pi * r["1"] / (E0_BAND1 * mu)
    T29, T31, T32 = (brightness_temperature(r[b], b) for b in ("29", "31", "32"))

    f = list(_safe_pct(R, R_PCTS)) + [np.nanmean(R), np.nanstd(R)]

    ok = np.isfinite(R) & np.isfinite(T31)
    hist, _, _ = np.histogram2d(R[ok], T31[ok], bins=[R_EDGES, T_EDGES])
    f += list((hist / max(ok.sum(), 1)).ravel())

    t_surf = _safe_pct(T31, 99)
    f += [t_surf, *_safe_pct(T31, [1, 5, 50])]
    dT = t_surf - T31
    dh, _ = np.histogram(dT[np.isfinite(dT)], bins=DT_EDGES)
    f += list(dh / max(np.isfinite(dT).sum(), 1))

    sw = T31 - T32
    ph = T29 - T31
    f += [np.nanmean(sw), *_safe_pct(sw, [50, 90]), np.nanmean(sw > 1.5)]
    f += [np.nanmean(ph), *_safe_pct(ph, [10, 90]), np.nanmean(ph > 0)]

    f += _objects(np.nan_to_num(R) > 0.15)
    f += _objects(np.nan_to_num(dT) > 10)

    f += _multiscale_std(R) + _multiscale_std(T31)
    return np.asarray(f, dtype=np.float32)


class TileFeatures(torch.utils.data.Dataset):
    def __init__(self, data_dir, tile_indices):
        self.data_dir, self.tile_indices = data_dir, tile_indices
        self.shard = read_layout(data_dir)

    def __len__(self):
        return len(self.tile_indices)

    def __getitem__(self, i):
        ix = self.tile_indices[i]
        with xr.open_dataset(tile_path(self.data_dir, ix, "", self.shard)) as ds:
            rad = read_radiances(ds, DEFAULT_BANDS)
            land = ds["land_mask"].values
            sza = ds["solar_zenith_angle"].values.astype(np.float32)
        feats = np.concatenate([base_features(rad, land, sza), physical_features(rad, sza)])
        return ix, np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def _collate(batch):
    return batch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--root", default=ROOT)
    p.add_argument("--out", default="embeddings")
    p.add_argument("--name", default="physical")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="first N tiles only (quick checks)")
    args = p.parse_args()

    data_dir = f"{args.root}/{args.split}"
    tile_indices = discover_tile_indices(data_dir)[: args.limit]
    names = base_names() + physical_names()
    print(f"Computing {len(names)} features for {len(tile_indices)} {args.split} tiles")

    loader = torch.utils.data.DataLoader(
        TileFeatures(data_dir, tile_indices), batch_size=64,
        num_workers=args.num_workers, collate_fn=_collate,
    )
    ix_all, f_all = [], []
    for k, batch in enumerate(loader):
        for ix, f in batch:
            ix_all.append(ix)
            f_all.append(f)
        if k % 20 == 0:
            print(f"  {len(ix_all):>6}/{len(tile_indices)}", flush=True)

    tile_ix = np.array(ix_all, dtype=np.int64)
    X = np.stack(f_all)
    order = np.argsort(tile_ix)
    tile_ix, X = tile_ix[order], X[order]
    assert X.shape[1] == len(names), (X.shape, len(names))

    out = Path(args.out) / args.split / f"{args.name}.npz"
    save_embeddings(out, X, tile_ix, model="physical_v2",
                    split=data_dir, features=",".join(names))
    print(f"Wrote {out}  ({X.shape[0]} tiles x {X.shape[1]} features)")


if __name__ == "__main__":
    main()
