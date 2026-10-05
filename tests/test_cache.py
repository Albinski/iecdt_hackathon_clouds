"""Tests for the in-RAM uint8 tile cache.

This path had no coverage, and that is exactly how a dtype promotion in
`quantisation_range` reached the cluster: every smoke run used
`data.cache: none`, so the first thing to exercise it was a real job, which
died in its first conv several minutes in with "Input type (double) and bias
type (c10::BFloat16) should be the same".
"""

import numpy as np
import pytest
import torch

from iecdt_hackathon.ijepa.cache import (
    CachedTileDataset,
    build_ram_cache,
    quantisation_range,
)
from iecdt_hackathon.ijepa.masking import MultiBlockMaskCollator
from iecdt_hackathon.ijepa.model import IJepa

TILE = 64      # a 4x4 patch grid; 2x2 cannot satisfy any min_keep
PATCH = 16


class FakeTiles(torch.utils.data.Dataset):
    """Stands in for ModisTileDataset, with its band-range asymmetry.

    The visible bands occupy a wide slice of [0, 1] while the thermal bands sit
    in a narrow window near 0.5, which is what makes per-band quantisation
    worth doing at all.
    """

    crop_size = None
    bands = ("1", "3", "4", "29", "31", "32")
    include_land_mask = False
    include_solar_zenith = False

    def __init__(self, n=24, channels=6, size=TILE, seed=0):
        self.n = n
        self.channels = channels
        self.size = size
        self.tile_indices = list(range(100, 100 + n))
        rng = np.random.default_rng(seed)
        lo = np.array([0.02, 0.07, 0.03, 0.45, 0.47, 0.49][:channels], np.float32)
        span = np.array([0.75, 0.70, 0.85, 0.13, 0.14, 0.12][:channels], np.float32)
        self.data = (
            lo[None, :, None, None]
            + rng.random((n, channels, size, size)).astype(np.float32)
            * span[None, :, None, None]
        )

    @property
    def n_channels(self):
        return self.channels

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "image": torch.from_numpy(self.data[i]),
            "tile_index": self.tile_indices[i],
        }


# --- the dtype regression -------------------------------------------------


def test_quantisation_range_is_float32():
    """A float64 surviving here promotes every cached image to double."""
    lo, hi = quantisation_range(FakeTiles(), n_sample=8)
    assert lo.dtype == np.float32
    assert hi.dtype == np.float32
    assert (hi - lo).dtype == np.float32


def test_quantisation_range_is_ordered_per_band():
    lo, hi = quantisation_range(FakeTiles(), n_sample=16)
    assert (hi > lo).all()


def test_cached_dataset_yields_float32_images():
    """The dtype the model actually sees. bf16 weights reject a double input."""
    ds = FakeTiles()
    cache, lo, hi = build_ram_cache(ds, num_workers=0, n_sample=8)
    cached = CachedTileDataset(cache, ds.tile_indices, lo, hi)
    item = cached[0]
    assert item["image"].dtype == torch.float32
    assert cached.lo.dtype == torch.float32
    assert cached.span.dtype == torch.float32


def test_cached_dataset_survives_a_float64_range():
    """Belt and braces: even handed float64 bounds, the item is float32."""
    ds = FakeTiles()
    cache, lo, hi = build_ram_cache(ds, num_workers=0, n_sample=8)
    cached = CachedTileDataset(
        cache, ds.tile_indices, lo.astype(np.float64), hi.astype(np.float64)
    )
    assert cached[0]["image"].dtype == torch.float32


# --- fidelity and shape ---------------------------------------------------


def test_cache_matches_the_source_within_quantisation_error():
    ds = FakeTiles()
    cache, lo, hi = build_ram_cache(ds, num_workers=0, n_sample=24)
    cached = CachedTileDataset(cache, ds.tile_indices, lo, hi)

    worst = 0.0
    for i in range(len(ds)):
        original = ds[i]["image"]
        error = (cached[i]["image"] - original).abs()
        # One quantisation step per band, with a little slack for the
        # percentile clip at the extremes.
        step = torch.as_tensor(hi - lo).view(-1, 1, 1) / 255.0
        assert (error <= step * 1.5 + 1e-6).all(), f"tile {i} exceeds one step"
        worst = max(worst, float((error / step.clamp(min=1e-9)).max()))
    assert worst < 1.5


def test_cache_preserves_shape_and_tile_index():
    ds = FakeTiles()
    cache, lo, hi = build_ram_cache(ds, num_workers=0, n_sample=8)
    cached = CachedTileDataset(cache, ds.tile_indices, lo, hi)
    assert cache.dtype == torch.uint8
    assert tuple(cache.shape) == (len(ds), ds.n_channels, TILE, TILE)
    assert len(cached) == len(ds)
    assert cached[3]["tile_index"] == ds.tile_indices[3]
    assert tuple(cached[3]["image"].shape) == (ds.n_channels, TILE, TILE)


def test_cache_uses_the_full_uint8_range():
    """Per-band scaling is the point: a naive x*255 would waste most of it."""
    ds = FakeTiles()
    cache, _, _ = build_ram_cache(ds, num_workers=0, n_sample=24)
    for band in range(ds.n_channels):
        spread = int(cache[:, band].max()) - int(cache[:, band].min())
        assert spread > 200, f"band {band} only spans {spread} of 255 levels"


def test_refuses_to_cache_a_cropped_dataset():
    """A random crop would be frozen into the cache, one per tile forever."""
    ds = FakeTiles()
    ds.crop_size = 16
    with pytest.raises(ValueError, match="randomly cropped"):
        build_ram_cache(ds, num_workers=0, n_sample=8)


# --- the thing the failed jobs would have caught -------------------------


def test_a_cached_batch_runs_through_the_model():
    """End to end: cache -> collator -> IJepa.forward, as a job does it.

    The four cached jobs in the first sweep died exactly here, between the
    dataloader and the first conv.
    """
    ds = FakeTiles(n=16, size=TILE)
    cache, lo, hi = build_ram_cache(ds, num_workers=0, n_sample=16)
    cached = CachedTileDataset(cache, ds.tile_indices, lo, hi)

    collator = MultiBlockMaskCollator(
        input_size=(TILE, TILE), patch_size=PATCH,
        min_keep_enc=2, min_keep_pred=0,
    )
    batch, masks_ctx, masks_tgt = collator([cached[i] for i in range(4)])
    assert batch["image"].dtype == torch.float32

    model = IJepa(arch="vit_tiny", patch_size=PATCH)
    loss = model(batch["image"], masks_ctx, masks_tgt)
    assert torch.isfinite(loss)
    assert model.encode(batch["image"]).dtype == torch.float32
