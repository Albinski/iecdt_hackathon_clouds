"""The only augmentation I-JEPA gets here.

The paper's own config turns almost everything off (`use_color_distortion:
false`, `use_gaussian_blur: false`, `use_horizontal_flip: false`) and keeps only
a random resized crop -- not doing invariance by augmentation is one of its
selling points. For six-band radiance tiles the remaining choices are narrower
still:

- **Resizing is out.** It changes km/pixel. Cloud-top height and optical
  thickness have absolute spatial scales, and the probes always see tiles at
  ~1 km/pixel, so scale invariance is capacity spent on something actively
  unwanted.
- **Anything photometric is out.** The radiance magnitude *is* the label:
  band 31 brightness temperature is cloud-top height. A per-channel gain or a
  brightness jitter destroys the signal being scored.
- **Channel dropout is out.** The probes always get all six bands.
- **The dihedral group is in.** The split spans 2003-2010, every latitude and
  every season, so each illumination geometry a flip or rotation produces does
  occur somewhere in the real data.
"""

import torch


def d4(x, k):
    """Apply element `k` of the dihedral group of order 8 to a (C, H, W) tile.

    `k % 4` quarter turns, then a horizontal flip when `k >= 4`.

    Every channel is transformed together. That is a correctness requirement
    rather than a convenience: with `include_solar_zenith` on, flipping the
    radiances but not the illumination gradient would produce a tile whose
    geometry contradicts its own brightness.
    """
    if not 0 <= k < 8:
        raise ValueError(f"k must be in 0..7, got {k}")
    if k % 4:
        x = torch.rot90(x, k % 4, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if k >= 4 else x
