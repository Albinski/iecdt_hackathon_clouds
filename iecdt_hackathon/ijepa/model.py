"""I-JEPA (Assran et al., arXiv:2301.08243) as a single `build_model` entry.

Why one module holding all three networks -- context encoder, EMA target
encoder and predictor -- rather than three: `embed.py` rebuilds a checkpoint
with `build_model(ckpt["model_name"], **ckpt["model_cfg"])` and then a *strict*
`load_state_dict`. Keeping everything in one module whose `model_cfg` round-trips
through the checkpoint means the reconstructed object has exactly the keys that
were saved, so the whole evaluation path -- `embed.py`, `evaluate.py`,
`submit.sh` -- works without a single change. The cost is a ~190 MB checkpoint
carrying two encoders and a predictor, which is cheap next to the alternative.

The things that would break that strict load, all avoided here: a persistent
positional-embedding buffer (see `vision_transformer`, where the table is a
module-level dict), a step counter or momentum kept as module state, naming the
channel argument anything other than `in_channels` (`train.py` injects that
name), and `torch.compile` (which prefixes every key with `_orig_mod.`, and buys
nothing when the run is I/O-bound anyway).
"""

import copy
import importlib

import torch
import torch.nn.functional as F
from torch import nn

from .vision_transformer import (
    ARCHS,
    VisionTransformer,
    VisionTransformerPredictor,
    apply_masks,
    repeat_interleave_batch,
)

#: pool name -> (how many of the last blocks to use, whether to add token std).
#:
#: `std` across the token axis measures how heterogeneous a tile is internally,
#: which is exactly what cloud fraction and scene type turn on and exactly what
#: a mean pool throws away. It costs one op.
POOLS = {
    "mean": (1, False),
    "mean_std": (1, True),
    "mean_last4": (4, False),
    "mean_last4_std": (4, True),
}


def _resolve(path):
    """Import `"package.module:attribute"`. Keeps `model_cfg` plain data.

    A hook taking a live callable could not be pickled into a checkpoint, and
    `model_cfg` has to survive the round-trip through `torch.save`.
    """
    module, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:attribute', got {path!r}")
    return getattr(importlib.import_module(module), attr)


class IJepa(nn.Module):
    """Predict the representations of masked target blocks from a context block.

    Args:
        in_channels: channels the dataset yields (6 bands, plus any extras).
        arch: encoder preset, one of `ARCHS`.
        patch_size: 16 over a 256 tile gives a 16x16 = 256 token grid, the same
            geometry as the paper's 224/14.
        pred_depth, pred_emb_dim, pred_num_heads: the predictor, kept narrow.
        pool: how `encode` turns tokens into one vector; see `POOLS`.
        encode_from: which encoder `encode` reads. See the note there.
        input_shift, input_scale: optional per-band affine applied before the
            patch embedding, held in non-persistent buffers so they never
            affect a strict load.
        extra_features: optional `"module:attribute"` hook, concatenated onto
            `encode`'s output. The callable must carry an `n_features` int.
            The lower-risk way to fuse features is `concat_embeddings.py`,
            which joins finished `.npz` files; this exists so computing them
            on the GPU at embed time is not foreclosed.
    """

    def __init__(
        self,
        in_channels=6,
        arch="vit_small",
        patch_size=16,
        pred_depth=6,
        pred_emb_dim=192,
        pred_num_heads=3,
        pool="mean_std",
        encode_from="target",
        input_shift=None,
        input_scale=None,
        extra_features=None,
    ):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"Unknown arch {arch!r}; choose from {sorted(ARCHS)}")
        if pool not in POOLS:
            raise ValueError(f"Unknown pool {pool!r}; choose from {sorted(POOLS)}")
        if encode_from not in ("target", "context"):
            raise ValueError(f"encode_from must be 'target' or 'context', got {encode_from!r}")

        spec = ARCHS[arch]
        self.arch = arch
        self.patch_size = patch_size
        self.pool = pool
        self.encode_from = encode_from
        self.embed_dim = spec["embed_dim"]

        self.context_encoder = VisionTransformer(
            in_channels=in_channels, patch_size=patch_size, **spec
        )
        self.target_encoder = copy.deepcopy(self.context_encoder).requires_grad_(False)
        self.predictor = VisionTransformerPredictor(
            embed_dim=self.embed_dim,
            pred_emb_dim=pred_emb_dim,
            depth=pred_depth,
            num_heads=pred_num_heads,
        )

        shift = torch.zeros(in_channels) if input_shift is None else torch.tensor(
            input_shift, dtype=torch.float32
        )
        scale = torch.ones(in_channels) if input_scale is None else torch.tensor(
            input_scale, dtype=torch.float32
        )
        self.register_buffer("input_shift", shift.view(1, -1, 1, 1), persistent=False)
        self.register_buffer("input_scale", scale.view(1, -1, 1, 1), persistent=False)

        self.extra_features = _resolve(extra_features) if extra_features else None

    @property
    def embedding_dim(self):
        """The D in `encode`'s (B, D). Read by `train.save`."""
        n_layers, with_std = POOLS[self.pool]
        dim = self.embed_dim * n_layers * (2 if with_std else 1)
        return dim + (self.extra_features.n_features if self.extra_features else 0)

    def _prepare(self, x):
        return (x - self.input_shift) * self.input_scale

    def forward(self, x, masks_ctx, masks_tgt):
        """The I-JEPA objective. -> a scalar loss."""
        x = self._prepare(x)
        grid = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)

        with torch.no_grad():
            h = self.target_encoder(x)
            # Non-affine LayerNorm over the feature axis, applied after the
            # encoder's own learned final norm and before the target blocks are
            # gathered. This is the piece that stops the trivial solution: left
            # out, the objective can be driven to zero by shrinking the target
            # scale rather than by predicting anything.
            h = F.layer_norm(h, (h.shape[-1],))
            h = apply_masks(h, masks_tgt)
            h = repeat_interleave_batch(h, x.shape[0], repeat=len(masks_ctx))

        z = self.context_encoder(x, masks_ctx)
        z = self.predictor(z, masks_ctx, masks_tgt, grid)

        # smooth_l1, not MSE, and no explicit mean over blocks: `apply_masks`
        # stacked every (context, target) pair onto the batch axis, so
        # reduction="mean" already averages over all of them uniformly.
        return F.smooth_l1_loss(z, h)

    @torch.no_grad()
    def update_target(self, momentum):
        """EMA the context encoder into the target encoder.

        Call this *outside* any autocast block. The parameters are fp32
        masters, and with momentum at 0.996 the update is three orders of
        magnitude smaller than the value it is added to -- in bfloat16, with
        eight mantissa bits, most of it rounds away silently.

        Only parameters are averaged. Neither encoder has buffers to carry:
        there is no normalisation with running statistics anywhere, and the
        positional table lives outside the module.
        """
        for src, dst in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            dst.mul_(momentum).add_(src.detach(), alpha=1.0 - momentum)

    def encode(self, x):
        """(B, C, H, W) -> (B, embedding_dim). The evaluation contract.

        Reads the **target** encoder by default. The usual argument is that an
        EMA is lower variance, but the decisive one here is specific to I-JEPA:
        the context encoder is only ever fed the context block -- about 72 of
        256 patches -- so running it on a whole tile at embed time is out of
        distribution. The target encoder is fed the full 256-token tile on
        every single training step, so full-tile inference is exactly its
        training condition.

        The non-affine LayerNorm before pooling is not decoration.
        `evaluate.py` fits its probes behind a `StandardScaler`, which divides
        by each dimension's standard deviation and only guards against an
        exactly-zero one. A ViT dimension that is merely *nearly* constant gets
        amplified into numerical noise, and the unregularised `LinearRegression`
        behind it will happily give that noise a large coefficient -- which is
        how a probe R^2 goes negative. Normalising bounds per-token magnitude
        and keeps the embedding in the same space the objective operated in.
        """
        x = self._prepare(x)
        encoder = (
            self.target_encoder if self.encode_from == "target" else self.context_encoder
        )
        n_layers, with_std = POOLS[self.pool]
        tokens = [encoder(x)] if n_layers == 1 else encoder.forward_last_k(x, n_layers)
        tokens = [F.layer_norm(t, (t.shape[-1],)) for t in tokens]

        parts = [t.mean(dim=1) for t in tokens]
        if with_std:
            parts += [t.std(dim=1) for t in tokens]
        z = torch.cat(parts, dim=-1)
        if self.extra_features is not None:
            z = torch.cat([z, self.extra_features(x)], dim=-1)
        return z

    @torch.no_grad()
    def collapse_stats(self, x):
        """Diagnostics that tell learning apart from collapse.

        The loss cannot do this on its own: a target encoder emitting one
        constant vector drives `smooth_l1` to zero, so a loss falling smoothly
        toward nothing is the *symptom* rather than the reassurance. All three
        numbers below read the target features before any normalisation, which
        is the only place collapse shows -- a LayerNorm forces unit variance
        and will always look healthy.

        - `feature_std`: spread across the batch. Heading for 0 is collapse.
        - `token_std`: spread *within* a tile. This is the mode specific to
          this setup, where every token in a tile converges on one vector: the
          loss stays plausible while mean pooling returns a near-constant and
          every probe dies. Watch it hardest.
        - `effective_rank`: RankMe over the pooled embeddings. The single most
          predictive number; a slide into single digits is collapse.
        """
        h = self.target_encoder(self._prepare(x)).float()
        z = self.encode(x).float()
        z = z - z.mean(dim=0)
        singular = torch.linalg.svdvals(z)
        return {
            "feature_std": h.std(dim=0).mean().item(),
            "token_std": h.std(dim=1).mean().item(),
            "effective_rank": (singular.sum() ** 2 / singular.pow(2).sum()).item(),
        }
