import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from .tile_layout import iter_tile_files, read_layout, tile_path

DEFAULT_BANDS = ("1", "3", "4", "29", "31", "32")


def load_band_stats(stats_path, bands):
    """Global per-band min/max, reindexed to `bands` order.

    Reindexing rather than assuming the file's order: the statistics file
    lists every MODIS band, and a model that selected six of them must get
    their ranges in the order it asked for.
    """
    with open(stats_path) as f:
        stats = json.load(f)
    missing = [b for b in bands if b not in stats["bands"]]
    if missing:
        raise ValueError(
            f"Bands {missing} not in {stats_path}; available: {stats['bands']}"
        )
    ixs = [stats["bands"].index(b) for b in bands]
    return (
        np.array([stats["min"][i] for i in ixs], dtype=np.float32),
        np.array([stats["max"][i] for i in ixs], dtype=np.float32),
    )


def discover_tile_indices(data_dir):
    """Sorted indices of every {idx}.nc in a flat or sharded tile directory."""
    ixs = set()
    for p in iter_tile_files(Path(data_dir), "*.nc"):
        if p.stem.isdigit():
            ixs.add(int(p.stem))
    return sorted(ixs)


def read_radiances(ds, bands):
    """(n_bands, y, x) float32 physical radiances from either tile schema."""
    if all(f"Rad_{b}" in ds for b in bands):
        return np.stack([ds[f"Rad_{b}"].values for b in bands]).astype(np.float32)
    if "Rad" in ds:
        return ds["Rad"].sel(band=list(bands)).values.astype(np.float32)
    raise KeyError(
        f"Tile has neither Rad_<band> variables nor a Rad(band, y, x) array; "
        f"found {sorted(ds.data_vars)}"
    )


class ModisTileDataset(torch.utils.data.Dataset):
    """MODIS tiles, optionally with the downstream label table attached.

    Args:
        data_dir: directory of {idx}.nc tiles, flat or sharded.
        stats_path: modis_band_stats.json providing the per-band min/max.
        bands: band names, in the channel order the model receives.
        include_land_mask: append the land mask as an extra channel.
        include_solar_zenith: append cos(solar zenith angle) as an extra channel.
        crop_size: random square crop, for training. None returns the full tile.
        labels_path: a labels NetCDF; when given, __getitem__ also returns the
            per-tile label vector, aligned by tile index.
        label_vars: which label variables to return (default: all scalar ones).
        tile_indices: restrict to these indices instead of discovering them.
    """

    def __init__(
        self,
        data_dir,
        stats_path,
        bands=DEFAULT_BANDS,
        include_land_mask=False,
        include_solar_zenith=False,
        crop_size=None,
        labels_path=None,
        label_vars=None,
        tile_indices=None,
        seed=None,
    ):
        self.data_dir = Path(data_dir)
        self.bands = tuple(bands)
        self.include_land_mask = include_land_mask
        self.include_solar_zenith = include_solar_zenith
        self.crop_size = crop_size
        self._shard_size = read_layout(self.data_dir)
        self._rng = np.random.default_rng(seed)

        self.band_min, self.band_max = load_band_stats(stats_path, self.bands)
        self._band_range = np.maximum(self.band_max - self.band_min, 1e-12)

        self.tile_indices = (
            sorted(int(i) for i in tile_indices)
            if tile_indices is not None
            else discover_tile_indices(self.data_dir)
        )
        if not self.tile_indices:
            raise ValueError(f"No {{idx}}.nc tiles found under {self.data_dir}")

        self.labels = None
        self.label_vars = None
        if labels_path is not None:
            self._attach_labels(labels_path, label_vars)

    def _attach_labels(self, labels_path, label_vars):
        """Load the label table and drop tiles it does not cover."""
        with xr.open_dataset(labels_path) as ds:
            ds = ds.load()
        index = ds["tile_index"].values.astype(np.int64)
        if label_vars is None:
            label_vars = [
                v for v in ds.data_vars if ds[v].dims == ("tile",) and v != "tile_index"
            ]
        self.label_vars = list(label_vars)

        position = {int(ix): i for i, ix in enumerate(index)}
        keep = [ix for ix in self.tile_indices if ix in position]
        if not keep:
            raise ValueError(
                f"{labels_path} covers none of the {len(self.tile_indices)} tiles "
                f"in {self.data_dir}"
            )
        self.tile_indices = keep
        rows = np.array([position[ix] for ix in keep])
        self.labels = {v: ds[v].values[rows] for v in self.label_vars}

    @property
    def n_channels(self):
        return len(self.bands) + self.include_land_mask + self.include_solar_zenith

    def __len__(self):
        return len(self.tile_indices)

    def normalize(self, rad):
        """Physical radiance -> roughly [0, 1], per band."""
        return (rad - self.band_min[:, None, None]) / self._band_range[:, None, None]

    def denormalize(self, x):
        return x * self._band_range[:, None, None] + self.band_min[:, None, None]

    def __getitem__(self, i):
        tile_ix = self.tile_indices[i]
        path = tile_path(self.data_dir, tile_ix, "", self._shard_size)
        with xr.open_dataset(path) as ds:
            rad = read_radiances(ds, self.bands)
            channels = [self.normalize(rad)]
            if self.include_solar_zenith:
                sza = ds["solar_zenith_angle"].values.astype(np.float32)
                channels.append(np.cos(np.deg2rad(sza))[None])
            if self.include_land_mask:
                channels.append(ds["land_mask"].values.astype(np.float32)[None])

        x = np.concatenate(channels, axis=0)
        # A handful of pixels in the thermal bands are NaN in the source data.
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.crop_size is not None:
            x = self._random_crop(x, self.crop_size)

        out = {
            "image": torch.from_numpy(np.ascontiguousarray(x)),
            "tile_index": tile_ix,
        }
        if self.labels is not None:
            out["labels"] = {
                v: np.float32(self.labels[v][i])
                if self.labels[v].dtype.kind == "f"
                else np.int64(self.labels[v][i])
                for v in self.label_vars
            }
        return out

    def _random_crop(self, x, size):
        _, h, w = x.shape
        if size > h or size > w:
            raise ValueError(f"crop_size {size} exceeds tile shape {(h, w)}")
        y0 = int(self._rng.integers(0, h - size + 1))
        x0 = int(self._rng.integers(0, w - size + 1))
        return x[:, y0 : y0 + size, x0 : x0 + size]


def collate(batch):
    """Default collate plus label dicts and an int64 tile-index tensor."""
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "tile_index": torch.tensor([b["tile_index"] for b in batch], dtype=torch.int64),
    }
    if "labels" in batch[0]:
        out["labels"] = {
            k: torch.tensor(np.array([b["labels"][k] for b in batch]))
            for k in batch[0]["labels"]
        }
    return out


def build_dataloader(
    dataset,
    batch_size,
    shuffle,
    num_workers=8,
    drop_last=False,
    seed=0,
    collate_fn=None,
):
    """DataLoader with the settings that keep GWS-backed reads fast.

    `collate_fn` defaults to `collate`. I-JEPA passes a collator that also
    samples its patch masks, so these settings stay in one place.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate if collate_fn is None else collate_fn,
        generator=generator,
    )
