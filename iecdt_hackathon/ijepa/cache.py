"""An in-RAM uint8 copy of a split, for when the loader is the bottleneck.

Reading the training split is the limiting cost here, not the GPU. Measured on
this data path: `ModisTileDataset.__getitem__` alone manages 29 tiles/s, and
`build_dataloader` with 8 workers reaches 258 tiles/s -- about 2 steps/s at
batch 128 -- against roughly 16 steps/s of ViT-S/16 I-JEPA compute. Nearly all
of that is zlib/HDF5 decompression of 100,000 ~1 MB NetCDF files. Holding the
split in memory removes it entirely, for one sequential pass of about five
minutes and 39 GB of RAM.

uint8 rather than float16 because 100000 x 6 x 256 x 256 is 39.3 GB as bytes and
78.6 GB as halves; the former leaves room for torch and a CUDA context inside a
120 GB allocation, the latter does not.

Quantisation needs care, though. Min-max normalisation leaves each band using
only part of [0, 1] -- on a sample tile the visible bands sit around 0.07-0.13
and the thermal bands are crushed into a ~0.13-wide window near 0.5 -- so a
naive `x * 255` would spend most of the available range on values that never
occur. Each band gets its own affine map from a sampled percentile range
instead, which puts the residual error near 0.2% of the band's observed spread,
comfortably below MODIS L1B noise.
"""

import time

import numpy as np
import torch

from ..data import build_dataloader


def quantisation_range(dataset, n_sample=2000, percentiles=(0.1, 99.9), seed=0):
    """Per-band (lo, hi) covering `percentiles` of a random tile sample.

    Percentiles rather than min/max so that a single extreme pixel -- or the
    zeros `nan_to_num` leaves behind in the thermal bands -- cannot stretch the
    range and quantise everything else into a handful of levels.
    """
    rng = np.random.default_rng(seed)
    n_sample = min(n_sample, len(dataset))
    rows = rng.choice(len(dataset), size=n_sample, replace=False)
    sample = np.stack([dataset[int(i)]["image"].numpy() for i in rows])
    lo, hi = np.percentile(sample, percentiles, axis=(0, 2, 3))
    # Cast before any arithmetic. `np.percentile` returns float64, and letting
    # one float64 survive into the returned range silently promotes every
    # cached image to double -- which torch then refuses against bf16 weights
    # with "Input type (double) and bias type (c10::BFloat16) should be the
    # same", several minutes into a job.
    lo = lo.astype(np.float32)
    hi = np.maximum(hi.astype(np.float32), lo + np.float32(1e-6))
    return lo, hi


def build_ram_cache(dataset, num_workers=8, n_sample=2000, log_every=20_000):
    """Read `dataset` once into a shared uint8 tensor.

    Returns `(cache, lo, hi)` with `cache` in `dataset.tile_indices` order.
    Shared memory is claimed before the training workers fork, so they read it
    without copying.
    """
    if dataset.crop_size is not None:
        raise ValueError(
            "Refusing to cache a randomly cropped dataset: the crop would be "
            "frozen into the cache and every epoch would see the same one. "
            "Set data.crop_size to null."
        )
    lo, hi = quantisation_range(dataset, n_sample=n_sample)

    # The dataloader workers must inherit this tensor rather than receive a
    # copy of it. `fork` gives them copy-on-write access for free, since
    # nothing ever writes to the cache after it is built; `spawn` would pickle
    # the whole 39 GB to every worker instead.
    start_method = torch.multiprocessing.get_start_method(allow_none=True)
    if start_method not in (None, "fork"):
        raise RuntimeError(
            f"data.cache needs the 'fork' start method so workers share the "
            f"cache; this process uses {start_method!r}. Either set "
            f"data.cache=none or torch.multiprocessing.set_start_method('fork')."
        )

    probe = dataset[0]["image"]
    # Deliberately *not* share_memory_(): that allocates a second 39 GB block
    # in /dev/shm and copies into it, so peak RSS doubles and both halves count
    # against the job's --mem cgroup. Measured at 96.7 GB against a 120 GB
    # limit on the first attempt. Fork COW gives the same sharing for one copy.
    cache = torch.empty((len(dataset), *probe.shape), dtype=torch.uint8)
    scale = 255.0 / (hi - lo)

    loader = build_dataloader(dataset, 64, shuffle=False, num_workers=num_workers)
    started, filled = time.time(), 0
    for batch in loader:
        x = batch["image"].numpy()
        quantised = np.clip(
            (x - lo[None, :, None, None]) * scale[None, :, None, None], 0, 255
        )
        n = len(x)
        cache[filled : filled + n] = torch.from_numpy(quantised.astype(np.uint8))
        filled += n
        if filled % log_every < n:
            rate = filled / (time.time() - started)
            print(
                f"  cached {filled:,}/{len(dataset):,} tiles  {rate:.0f} tiles/s",
                flush=True,
            )
    print(
        f"  cache built: {cache.numel() / 1e9:.1f} GB in "
        f"{(time.time() - started) / 60:.1f} min",
        flush=True,
    )
    return cache, lo, hi


class CachedTileDataset(torch.utils.data.Dataset):
    """Serves `build_ram_cache`'s output with `ModisTileDataset`'s item shape.

    Deliberately returns the same `{"image", "tile_index"}` dict, so the mask
    collator and the training loop cannot tell the difference.
    """

    def __init__(self, cache, tile_indices, lo, hi):
        self.cache = cache
        self.tile_indices = list(tile_indices)
        # float32 explicitly, not inherited from whatever numpy handed over:
        # a single float64 here promotes every image to double and the failure
        # surfaces in the first conv, minutes into a run.
        self.lo = torch.as_tensor(lo, dtype=torch.float32).view(-1, 1, 1)
        self.span = (
            torch.as_tensor(hi - lo, dtype=torch.float32).view(-1, 1, 1) / 255.0
        )

    def __len__(self):
        return len(self.tile_indices)

    def __getitem__(self, i):
        x = self.cache[i].to(torch.float32) * self.span + self.lo
        return {"image": x, "tile_index": self.tile_indices[i]}
