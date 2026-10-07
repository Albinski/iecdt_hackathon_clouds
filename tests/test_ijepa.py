"""Tests for the I-JEPA additions.

Most of these guard the contract in `embed.py` rather than the mathematics:
a strict `load_state_dict` on a `build_model`-reconstructed module is easy to
break by accident -- a buffer here, a renamed argument there -- and the failure
surfaces hours later at embedding time rather than in training.

Small grids and `vit_tiny` throughout, so the file runs in seconds on CPU.
"""

import copy

import pytest
import torch

from iecdt_hackathon.embed import load_checkpoint_model
from iecdt_hackathon.ijepa.masking import MultiBlockMaskCollator
from iecdt_hackathon.ijepa.model import POOLS, IJepa
from iecdt_hackathon.ijepa.transforms import d4
from iecdt_hackathon.ijepa.vision_transformer import (
    apply_masks,
    repeat_interleave_batch,
    sincos_2d,
)
from iecdt_hackathon.models import build_model
from iecdt_hackathon.train import save

TILE = 128
PATCH = 16
GRID = TILE // PATCH          # 8 -> 64 tokens
SMALL = dict(arch="vit_tiny", patch_size=PATCH)
# A 64-token grid puts target blocks at ~0.15-0.2 * 64 = 9-13 patches, so the
# production min_keep values would never be satisfiable here.
SMALL_MASK = dict(min_keep_enc=16, min_keep_pred=4)


def samples(n, channels=6, size=TILE):
    return [
        {"image": torch.rand(channels, size, size), "tile_index": i} for i in range(n)
    ]


def collator(**kwargs):
    return MultiBlockMaskCollator(
        input_size=(TILE, TILE), patch_size=PATCH, **{**SMALL_MASK, **kwargs}
    )


# --- positional embeddings -------------------------------------------------


def test_sincos_is_scale_preserving():
    """A sub-grid of the table must equal the table for that sub-grid.

    This is the property that lets one checkpoint embed tiles at a resolution
    it was not trained on without silently redefining "adjacent patch". The
    reference implementation's bicubic resample does *not* have it, which is
    correct for natural images and wrong for fixed-km/pixel satellite tiles.
    """
    big = sincos_2d(192, 16, 16)
    small = sincos_2d(192, 8, 16)
    assert torch.equal(big[:, : 8 * 16], small)


def test_sincos_rejects_indivisible_dim():
    with pytest.raises(ValueError, match="divisible by 4"):
        sincos_2d(190, 8, 8)


# --- mask helpers ----------------------------------------------------------


def test_apply_masks_stacks_on_the_batch_axis():
    x = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    masks = [torch.tensor([[0, 1], [2, 3]]), torch.tensor([[4, 0], [1, 2]])]
    out = apply_masks(x, masks)
    assert out.shape == (4, 2, 3)
    assert torch.equal(out[0], x[0, [0, 1]])
    assert torch.equal(out[3], x[1, [1, 2]])


def test_repeat_interleave_batch_is_identity_for_one_context():
    x = torch.randn(6, 4, 3)
    assert torch.equal(repeat_interleave_batch(x, 6, repeat=1), x)


def test_repeat_interleave_batch_keeps_blocks_contiguous():
    x = torch.arange(4 * 1 * 1, dtype=torch.float32).reshape(4, 1, 1)
    out = repeat_interleave_batch(x, 2, repeat=2).flatten().tolist()
    assert out == [0, 1, 0, 1, 2, 3, 2, 3]


# --- the collator ----------------------------------------------------------


def test_collator_shapes_and_passthrough():
    batch, masks_ctx, masks_tgt = collator(nenc=1, npred=4)(samples(4))
    assert batch["image"].shape == (4, 6, TILE, TILE)
    assert batch["tile_index"].tolist() == [0, 1, 2, 3]
    assert len(masks_ctx) == 1 and len(masks_tgt) == 4
    for m in masks_ctx + masks_tgt:
        assert m.shape[0] == 4
        assert m.dtype == torch.int64        # torch.gather requires int64
        assert int(m.max()) < GRID * GRID


def test_context_and_targets_are_disjoint_per_image():
    _, masks_ctx, masks_tgt = collator(nenc=1, npred=4, allow_overlap=False)(
        samples(6)
    )
    for i in range(6):
        ctx = set(masks_ctx[0][i].tolist())
        targets = set().union(*(set(m[i].tolist()) for m in masks_tgt))
        assert not (ctx & targets)


def test_collator_respects_min_keep():
    _, masks_ctx, masks_tgt = collator(
        nenc=1, npred=4, min_keep_enc=12, min_keep_pred=4
    )(samples(4))
    assert masks_ctx[0].shape[1] > 12
    assert masks_tgt[0].shape[1] > 4


def test_successive_batches_get_different_masks():
    """Block sizes are redrawn per batch, so two calls must not coincide."""
    c = collator()
    first = c(samples(4))[2][0]
    second = c(samples(4))[2][0]
    assert not (first.shape == second.shape and torch.equal(first, second))


def test_mask_rng_is_shared_across_forked_workers():
    """The counter must be process-shared, not a per-worker copy.

    `collate_fn` runs inside the dataloader workers. With a plain integer each
    forked worker would walk the identical seed sequence and every worker would
    draw the same blocks and the same flips -- silently, with no error and no
    visible symptom beyond a worse model. The parent seeing the counter advance
    is the direct evidence that the shared `Value` is doing its job.
    """

    class Tiles(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, i):
            return {"image": torch.rand(6, TILE, TILE), "tile_index": i}

    c = collator(seed=0)
    loader = torch.utils.data.DataLoader(
        Tiles(), batch_size=2, num_workers=2, collate_fn=c
    )
    batches = sum(1 for _ in loader)
    assert batches == 4
    assert c._counter.value == 4


def test_collator_rejects_indivisible_tile():
    with pytest.raises(ValueError, match="divisible"):
        MultiBlockMaskCollator(input_size=(250, 250), patch_size=16)


# --- d4 --------------------------------------------------------------------


def test_d4_is_a_group_of_order_eight():
    x = torch.rand(6, 8, 8)
    seen = {d4(x, k).numpy().tobytes() for k in range(8)}
    assert len(seen) == 8
    assert torch.equal(d4(x, 0), x)


def test_d4_rejects_out_of_range():
    with pytest.raises(ValueError, match="0..7"):
        d4(torch.rand(6, 8, 8), 8)


# --- the model -------------------------------------------------------------


def test_forward_returns_a_finite_scalar_loss():
    model = IJepa(**SMALL)
    batch, masks_ctx, masks_tgt = collator()(samples(4))
    loss = model(batch["image"], masks_ctx, masks_tgt)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_forward_with_several_context_masks():
    """nenc > 1 exercises repeat_interleave_batch, which is otherwise dead."""
    model = IJepa(**SMALL)
    batch, masks_ctx, masks_tgt = collator(nenc=2, npred=3)(samples(4))
    assert torch.isfinite(model(batch["image"], masks_ctx, masks_tgt))


@pytest.mark.parametrize("pool", sorted(POOLS))
def test_encode_dimensions(pool):
    model = IJepa(pool=pool, **SMALL)
    z = model.encode(torch.rand(2, 6, TILE, TILE))
    assert z.shape == (2, model.embedding_dim)
    assert torch.isfinite(z).all()


def test_encode_is_resolution_agnostic():
    model = IJepa(**SMALL)
    for size in (TILE, 256):
        assert model.encode(torch.rand(1, 6, size, size)).shape == (
            1,
            model.embedding_dim,
        )


def test_target_encoder_is_frozen_but_tracks_the_context_encoder():
    model = IJepa(**SMALL)
    assert all(not p.requires_grad for p in model.target_encoder.parameters())

    batch, masks_ctx, masks_tgt = collator()(samples(4))
    model(batch["image"], masks_ctx, masks_tgt).backward()
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert any(p.grad is not None for p in model.context_encoder.parameters())

    before = copy.deepcopy(model.target_encoder.state_dict())
    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(0.1)
    model.update_target(0.996)
    moved = [
        not torch.equal(before[k], v)
        for k, v in model.target_encoder.state_dict().items()
    ]
    assert any(moved)


def test_collapse_stats_are_finite_and_nontrivial():
    model = IJepa(**SMALL)
    stats = model.collapse_stats(torch.rand(8, 6, TILE, TILE))
    assert set(stats) == {"feature_std", "token_std", "effective_rank"}
    assert all(v == v and v != float("inf") for v in stats.values())
    assert stats["effective_rank"] > 1.0


def test_rejects_unknown_arch_pool_and_encoder():
    for kwargs, match in [
        (dict(arch="vit_enormous"), "Unknown arch"),
        (dict(pool="median"), "Unknown pool"),
        (dict(encode_from="predictor"), "encode_from"),
    ]:
        with pytest.raises(ValueError, match=match):
            IJepa(**{**SMALL, **kwargs})


# --- the embed.py contract -------------------------------------------------


def test_state_dict_carries_no_resolution_dependent_entries():
    """No positional table and no input affine, so a strict load cannot break."""
    keys = IJepa(**SMALL).state_dict()
    assert not [k for k in keys if "pos_embed" in k]
    assert not [k for k in keys if k in ("input_shift", "input_scale")]


def test_build_model_registers_ijepa():
    assert isinstance(build_model("ijepa", in_channels=6, **SMALL), IJepa)


def test_checkpoint_round_trips_through_embed_py(tmp_path):
    """The whole point: `embed.py` must need no changes.

    Saves through `train.save` and reloads through `embed.load_checkpoint_model`,
    which does a *strict* `load_state_dict` on a `build_model`-reconstructed
    module.
    """
    model_cfg = dict(in_channels=6, pool="mean_std", **SMALL)
    model = build_model("ijepa", **model_cfg)

    class FakeDataset:
        bands = ("1", "3", "4", "29", "31", "32")
        include_land_mask = False
        include_solar_zenith = False

    path = tmp_path / "best.pt"
    save(path, model, "ijepa", model_cfg, {}, 10, 0.5, FakeDataset())

    loaded, settings, ckpt = load_checkpoint_model(path, torch.device("cpu"))
    # vit_tiny is 192 wide, so mean+std is 384. Production is vit_small -> 768.
    assert ckpt["embedding_dim"] == 384
    assert settings["bands"] == FakeDataset.bands
    z = loaded.encode(torch.rand(2, 6, TILE, TILE))
    assert z.shape == (2, 384)


def test_a_checkpoint_loads_at_a_resolution_it_was_not_saved_at(tmp_path):
    model_cfg = dict(in_channels=6, **SMALL)
    model = build_model("ijepa", **model_cfg)
    model.encode(torch.rand(1, 6, TILE, TILE))     # populate the 8x8 table

    path = tmp_path / "small.pt"
    torch.save(
        {"state_dict": model.state_dict(), "model_name": "ijepa",
         "model_cfg": model_cfg, "bands": ["1"], "embedding_dim": 384},
        path,
    )
    rebuilt = build_model("ijepa", **model_cfg)
    rebuilt.load_state_dict(torch.load(path, weights_only=False)["state_dict"])
    assert rebuilt.encode(torch.rand(1, 6, 256, 256)).shape[0] == 1


def test_pool_variants_share_one_set_of_weights(tmp_path):
    """Each variant checkpoint must load into its own `pool` and nothing else.

    This is what lets one training run emit four embeddings for `evaluate.py`
    to rank, instead of guessing the pooling in advance.
    """
    base = dict(in_channels=6, **SMALL)
    model = build_model("ijepa", **base, pool="mean")
    state = model.state_dict()
    for pool, dim in [("mean", 192), ("mean_std", 384), ("mean_last4", 768),
                      ("mean_last4_std", 1536)]:
        variant = build_model("ijepa", **base, pool=pool)
        variant.load_state_dict(state)             # strict
        assert variant.embedding_dim == dim
        assert variant.encode(torch.rand(2, 6, TILE, TILE)).shape == (2, dim)
