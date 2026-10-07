"""A self-contained ViT for I-JEPA, plus the mask-gathering helpers it needs.

Deliberately not `timm`: it is not in the lockfile, and the pieces I-JEPA wants
are unusual enough -- no CLS token, patch selection inside `forward`, a separate
narrow predictor over (context, target) pairs -- that vendoring is clearer than
bending a general-purpose implementation around them.

One substantive departure from facebookresearch/ijepa, in the positional
embeddings. The reference keeps a fixed sin-cos table as a buffer and bicubically
resamples it when the token count changes, which normalises position to the
image's field of view. For MODIS tiles that is the wrong convention: a 256x256
tile covers four times the ground area of a 128x128 crop at the same ~1 km/pixel,
so the spacing between neighbouring patches must *not* change with crop size.
Here the coordinate is simply the patch index, the table is generated on demand
for whatever grid arrives, and it is memoised in a module-level dict rather than
a buffer -- so it never enters `state_dict` and the strict `load_state_dict` in
`embed.py` is immune to a change of resolution.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

#: Encoder presets, following the paper's naming.
ARCHS = {
    "vit_tiny": dict(embed_dim=192, depth=12, num_heads=3),
    "vit_small": dict(embed_dim=384, depth=12, num_heads=6),
    "vit_base": dict(embed_dim=768, depth=12, num_heads=12),
}

_POS_CACHE = {}


def sincos_2d(embed_dim, h, w, device=None, dtype=torch.float32):
    """(1, h*w, embed_dim) fixed 2-D sin-cos positional table, row-major.

    Scale-preserving: the coordinate *is* the patch index, so the table for a
    16x16 grid restricted to its first k rows is bit-identical to the table for
    a kx16 grid. That is what lets one checkpoint embed tiles at a resolution it
    was not trained on without reinterpreting what "adjacent" means.
    """
    if embed_dim % 4:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")
    omega = 1.0 / 10_000 ** (
        torch.arange(embed_dim // 4, dtype=torch.float64) / (embed_dim // 4)
    )
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float64),
        torch.arange(w, dtype=torch.float64),
        indexing="ij",
    )
    parts = []
    for grid in (grid_y, grid_x):
        angles = grid.reshape(-1, 1) * omega.reshape(1, -1)
        parts += [torch.sin(angles), torch.cos(angles)]
    pos = torch.cat(parts, dim=1).reshape(1, h * w, embed_dim)
    return pos.to(device=device, dtype=dtype)


def pos_embed_2d(embed_dim, h, w, device, dtype):
    """`sincos_2d`, memoised. The table is read-only, so sharing it is safe."""
    key = (embed_dim, h, w, str(device), dtype)
    if key not in _POS_CACHE:
        _POS_CACHE[key] = sincos_2d(embed_dim, h, w, device, dtype)
    return _POS_CACHE[key]


def apply_masks(x, masks):
    """Keep only the patches each mask indexes, stacking the masks on the batch.

    x: (B, N, D); masks: a list of (B, K) int64 patch indices.
    -> (len(masks) * B, K, D)

    Stacking on the batch axis rather than a new axis is what makes the loss
    average uniformly over every (context, target) pair without an explicit
    per-block mean.
    """
    return torch.cat(
        [x.gather(1, m.unsqueeze(-1).expand(-1, -1, x.shape[-1])) for m in masks],
        dim=0,
    )


def repeat_interleave_batch(x, batch_size, repeat):
    """Tile each batch-sized block of `x` `repeat` times, blocks kept in order.

    `apply_masks` has already laid the target blocks out along the batch axis;
    with more than one context mask the predictor emits every (context, target)
    pair, so the targets have to be expanded in the matching order. With a
    single context mask -- the default, and the paper's -- this is the identity.
    """
    n_blocks = x.shape[0] // batch_size
    return torch.cat(
        [
            x[i * batch_size : (i + 1) * batch_size].repeat(
                repeat, *([1] * (x.ndim - 1))
            )
            for i in range(n_blocks)
        ],
        dim=0,
    )


def _init_weights(module, std):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)


def _rescale_blocks(blocks):
    """Shrink each block's output projections by 1/sqrt(2 * depth_so_far).

    Keeps the residual stream's variance from growing with depth, which matters
    more here than usual: there is no learning-rate warmup long enough to
    rescue a 12-layer stack that starts out unstable.
    """
    for i, block in enumerate(blocks, start=1):
        block.attn.proj.weight.data.div_(math.sqrt(2.0 * i))
        block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * i))


class PatchEmbed(nn.Module):
    """(B, C, H, W) -> (B, H/p * W/p, D), with the patch grid it produced."""

    def __init__(self, patch_size=16, in_channels=6, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        p = self.patch_size
        h, w = x.shape[-2:]
        if h % p or w % p:
            raise ValueError(f"input {(h, w)} is not divisible by patch_size {p}")
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2), h // p, w // p


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, d // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    """Pre-norm transformer block. LayerNorm only, and no dropout anywhere.

    That is a compatibility property, not just a simplification: `embed.py`
    calls `.eval()`, and with no normalisation statistics and no stochastic
    layers the train and eval forward passes are identical, so an embedding can
    never silently differ from what training optimised.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """The I-JEPA encoder: a ViT with no CLS token that can take patch masks.

    `forward(x, masks)` gathers the indexed patches *after* the positional
    embedding and before the first block, so the context encoder only ever pays
    for the patches it keeps -- which is the whole point of the architecture.
    """

    def __init__(
        self,
        in_channels=6,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        init_std=0.02,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        _init_weights(self, init_std)
        _rescale_blocks(self.blocks)

    def tokens(self, x):
        t, grid_h, grid_w = self.patch_embed(x)
        return t + pos_embed_2d(self.embed_dim, grid_h, grid_w, t.device, t.dtype)

    def forward(self, x, masks=None):
        """(B, C, H, W) -> (B, N, D), or (len(masks) * B, K, D) when masked."""
        t = self.tokens(x)
        if masks is not None:
            t = apply_masks(t, masks)
        for block in self.blocks:
            t = block(t)
        return self.norm(t)

    def forward_last_k(self, x, k):
        """Raw outputs of the last `k` blocks, oldest first. Never masked.

        The learned final `self.norm` is skipped: it is trained for the last
        block's scale only, and applying it to earlier blocks would be
        meaningless. Callers put all `k` on a common scale with a non-affine
        LayerNorm instead -- and since every probe in `evaluate.py` is fitted
        behind a `StandardScaler`, a per-dimension affine either way makes no
        difference to the score.
        """
        t = self.tokens(x)
        first = len(self.blocks) - k
        out = []
        for i, block in enumerate(self.blocks):
            t = block(t)
            if i >= first:
                out.append(t)
        return out


class VisionTransformerPredictor(nn.Module):
    """Predicts target-block representations from the context representation.

    Deliberately much narrower and shallower than the encoder (the paper's
    ViT-H config pairs a 1280-wide encoder with a 384-wide predictor). The
    asymmetry is the mechanism: a predictor strong enough to infer the targets
    from a thin context would let the encoder off the hook, and the semantics
    would end up here instead of in the representation being evaluated.
    """

    def __init__(
        self,
        embed_dim=384,
        pred_emb_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        init_std=0.02,
    ):
        super().__init__()
        self.pred_emb_dim = pred_emb_dim
        self.predictor_embed = nn.Linear(embed_dim, pred_emb_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_emb_dim))
        self.blocks = nn.ModuleList(
            [Block(pred_emb_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(pred_emb_dim)
        self.proj = nn.Linear(pred_emb_dim, embed_dim)
        _init_weights(self, init_std)
        _rescale_blocks(self.blocks)
        nn.init.trunc_normal_(self.mask_token, std=init_std)

    def forward(self, x, masks_ctx, masks_tgt, grid):
        """(nenc*B, K_ctx, D) -> (nenc*npred*B, K_tgt, D).

        The predictor sees the context tokens at their own positions and one
        learned mask token per target patch, carrying only that patch's
        position. So the only thing distinguishing two target predictions is
        where they are -- it cannot cheat by looking at the target content.
        """
        batch_size = x.shape[0] // len(masks_ctx)
        x = self.predictor_embed(x)
        pos = pos_embed_2d(
            self.pred_emb_dim, grid[0], grid[1], x.device, x.dtype
        ).expand(batch_size, -1, -1)

        x = x + apply_masks(pos, masks_ctx)
        n_ctx = x.shape[1]

        tgt_pos = repeat_interleave_batch(
            apply_masks(pos, masks_tgt), batch_size, repeat=len(masks_ctx)
        )
        tgt = self.mask_token.expand(tgt_pos.shape[0], tgt_pos.shape[1], -1) + tgt_pos

        x = torch.cat([x.repeat(len(masks_tgt), 1, 1), tgt], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.proj(self.norm(x)[:, n_ctx:])
