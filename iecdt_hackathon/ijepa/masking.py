"""I-JEPA's multi-block masking, as a dataloader collate function.

Each batch gets one large *context* block and several smaller *target* blocks
per image, with the context restricted to the complement of that image's
targets. Following `src/masks/multiblock.py` in the reference implementation,
the block **sizes** are drawn once per batch while the block **positions** are
drawn independently per image, and both mask sets are then truncated to the
batch-minimum keep count so they stack into rectangular tensors.

Measured on the 16x16 grid this repo's 256x256 tiles produce, with the paper's
scales: target blocks come out at 35-54 patches, and the context block clamps to
15x15 = 225 before the target complements cut it to 75-157. The batch-minimum
truncation then lands the context at 57-95 patches, so roughly a third of the
available context is discarded. That is faithful to the reference and acts as a
mild extra augmentation, but `n_ctx` and `n_tgt` are worth logging so the cost
is visible rather than assumed.
"""

import math
from multiprocessing import Value

import torch
from torch.utils.data import default_collate

from ..data import collate
from .transforms import d4


class MultiBlockMaskCollator:
    """Collates a batch and the context/target patch masks that go with it.

    Returns `(batch, masks_ctx, masks_tgt)` where `batch` is whatever
    `data.collate` produces -- so `tile_index` and any attached `labels`
    survive -- and the two mask lists hold `nenc` and `npred` int64 tensors of
    shape `(B, K)` respectively.

    Four deliberate departures from the reference, all of them small:

    1. It wraps this repo's `data.collate` rather than `default_collate`, to
       keep `tile_index` (the join key every submission is scored by).
    2. `min_keep_enc` and `min_keep_pred` are separate. The reference's single
       `min_keep=10` is tuned for a 14-patch grid; on a 16-patch grid with
       target blocks of ~44 patches it never binds.
    3. The aspect ratio is drawn from its own uniform. The reference reuses one
       sample for both the scale and the aspect ratio, correlating them for no
       stated reason.
    4. The dihedral augmentation is applied here, per sample, before collation.

    The last point is also why the random state is a `multiprocessing.Value`:
    `collate_fn` runs *inside* the dataloader workers, and with a plain integer
    counter every forked worker would walk the identical seed sequence, so all
    of them would draw the same block sizes and the same flips.
    """

    def __init__(
        self,
        input_size=(256, 256),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep_enc=64,
        min_keep_pred=16,
        allow_overlap=False,
        d4=True,
        seed=0,
    ):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if input_size[0] % patch_size or input_size[1] % patch_size:
            raise ValueError(
                f"input_size {input_size} is not divisible by patch_size {patch_size}"
            )
        self.grid_h = input_size[0] // patch_size
        self.grid_w = input_size[1] // patch_size
        self.n_patches = self.grid_h * self.grid_w
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep_enc = min_keep_enc
        self.min_keep_pred = min_keep_pred
        self.allow_overlap = allow_overlap
        self.d4 = d4
        self._counter = Value("i", seed)

    def step(self):
        """The next shared seed. Lock-guarded, so workers never collide."""
        with self._counter.get_lock():
            self._counter.value += 1
            return self._counter.value

    def _sample_block_size(self, generator, scale, aspect_ratio):
        """A (h, w) block size in patches, clamped to fit inside the grid."""
        low, high = scale
        area = self.n_patches * (low + torch.rand(1, generator=generator).item() * (high - low))
        low, high = aspect_ratio
        ratio = low + torch.rand(1, generator=generator).item() * (high - low)
        h = max(1, int(round(math.sqrt(area * ratio))))
        w = max(1, int(round(math.sqrt(area / ratio))))
        # Strictly inside the grid, so there is always at least one valid
        # position to sample a top-left corner from.
        h = min(h, self.grid_h - 1)
        w = min(w, self.grid_w - 1)
        return h, w

    def _sample_block_mask(self, block, generator, min_keep, acceptable=None):
        """Flat indices of one block, plus its complement as a 0/1 grid.

        `acceptable` is a list of complement grids the block must fall inside.
        When a block cannot satisfy all of them -- the usual case once four
        target blocks have been carved out -- the constraint is relaxed one
        region at a time rather than failing, which is what the reference does.
        """
        h, w = block
        tries, timeout = 0, 20
        # Hard ceiling on total attempts. Once `tries` exceeds the number of
        # acceptable regions the block is unconstrained, so if it still cannot
        # satisfy `min_keep` then no block of this size ever will -- the
        # configuration is impossible and this loop would spin forever. A hung
        # job looks exactly like a slow one and would quietly consume its whole
        # allocation, so fail loudly instead.
        attempts, max_attempts = 0, 20 * (len(acceptable or ()) + 2)
        while True:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    f"Cannot sample a {h}x{w} block keeping more than "
                    f"{min_keep} of {self.n_patches} patches on a "
                    f"{self.grid_h}x{self.grid_w} grid. Lower min_keep_enc / "
                    f"min_keep_pred, or use a larger tile or smaller patch."
                )
            top = int(torch.randint(0, self.grid_h - h, (1,), generator=generator))
            left = int(torch.randint(0, self.grid_w - w, (1,), generator=generator))
            mask = torch.zeros((self.grid_h, self.grid_w), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1
            if acceptable is not None:
                for region in acceptable[: max(len(acceptable) - tries, 0)]:
                    mask *= region
            kept = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
            if len(kept) > min_keep:
                break
            timeout -= 1
            if timeout == 0:
                tries += 1
                timeout = 20

        complement = torch.ones((self.grid_h, self.grid_w), dtype=torch.int32)
        complement[top : top + h, left : left + w] = 0
        return kept, complement

    def __call__(self, samples):
        generator = torch.Generator().manual_seed(self.step())

        if self.d4:
            samples = [
                dict(
                    sample,
                    image=d4(
                        sample["image"],
                        int(torch.randint(0, 8, (1,), generator=generator)),
                    ),
                )
                for sample in samples
            ]
        batch = collate(samples)

        tgt_block = self._sample_block_size(
            generator, self.pred_mask_scale, self.aspect_ratio
        )
        ctx_block = self._sample_block_size(generator, self.enc_mask_scale, (1.0, 1.0))

        per_image_tgt, per_image_ctx = [], []
        keep_tgt = keep_ctx = self.n_patches
        for _ in samples:
            targets, complements = [], []
            for _ in range(self.npred):
                kept, complement = self._sample_block_mask(
                    tgt_block, generator, self.min_keep_pred
                )
                targets.append(kept)
                complements.append(complement)
                keep_tgt = min(keep_tgt, len(kept))
            per_image_tgt.append(targets)

            acceptable = None if self.allow_overlap else complements
            contexts = []
            for _ in range(self.nenc):
                kept, _ = self._sample_block_mask(
                    ctx_block, generator, self.min_keep_enc, acceptable
                )
                contexts.append(kept)
                keep_ctx = min(keep_ctx, len(kept))
            per_image_ctx.append(contexts)

        masks_tgt = default_collate(
            [[m[:keep_tgt] for m in masks] for masks in per_image_tgt]
        )
        masks_ctx = default_collate(
            [[m[:keep_ctx] for m in masks] for masks in per_image_ctx]
        )
        return batch, masks_ctx, masks_tgt
