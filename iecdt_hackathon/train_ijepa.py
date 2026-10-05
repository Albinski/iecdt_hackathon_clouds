"""Pretrain an I-JEPA encoder on MODIS tiles.

    python -m iecdt_hackathon.train_ijepa --config configs/ijepa.yaml

Same conventions as `train.py`: a YAML config, `--set section.key=value`
overrides, step-based, single GPU. The checkpoints it writes are consumed by the
unmodified `embed.py`.

Two things differ from the autoencoder loop, both because there is no
reconstruction error to lean on.

**Model selection is a linear probe, not the training loss.** The I-JEPA loss is
minimised by collapse -- a target encoder emitting one constant vector drives it
to zero -- and the target encoder drifts as the EMA momentum anneals, so values
are not even comparable across steps. Instead, every `probe_every` steps the run
embeds a fixed subset of labelled validation tiles and fits the same probes
`evaluate.py` uses. The held-out I-JEPA loss is still logged, as provenance.

**The probe scores only the three regression tasks.** `task_6` is the task
I-JEPA is meant to improve, but it is useless as a *selection signal* at this
scale: its validation class counts are
`{0: 30, 1: 1792, 2: 1393, 3: 121, 4: 20, 5: 336, 6: 41, 7: 927, 8: 5297, 9: 43}`,
so in a 4,000-tile subset the rarest classes have two or three rows and macro-F1
over five folds is mostly fold-assignment luck -- this repo's own seed repeat
puts its noise at 0.017, twice the overall figure. It is also the slow half of
the probe, since `LogisticRegression(C=np.inf, max_iter=5000)` carries no
penalty. Dropping it keeps the in-loop probe near 20 s, cheap enough to run
often. `task_6` is then measured properly at the end, by the full four-task
`evaluate.py` over the whole validation split.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml

from .data import build_dataloader
from .evaluate import overall_score, run_probes_local
from .ijepa.cache import CachedTileDataset, build_ram_cache
from .ijepa.masking import MultiBlockMaskCollator
from .ijepa.schedules import linear_schedule, param_groups, warmup_cosine
from .models import build_model
from .tasks import REGRESSION_TASKS, TASKS
from .train import build_datasets, load_config, save

#: The three regression tasks. See the module docstring for why `task_6` is out.
PROBE_TASKS = {k: TASKS[k] for k in REGRESSION_TASKS}

#: One checkpoint per pooling, all from the same trained weights. Because
#: `pool` lives in `model_cfg` and that round-trips through the checkpoint, each
#: file is independently loadable by the unmodified `embed.py` -- so which
#: pooling wins is settled by `evaluate.py` rather than by argument.
POOL_VARIANTS = {
    "mean": "mean",
    "meanstd": "mean_std",
    "mean4": "mean_last4",
    "mean4std": "mean_last4_std",
}


def build_collator(cfg, tile_size, seed):
    m = cfg.get("mask", {})
    return MultiBlockMaskCollator(
        input_size=tile_size,
        patch_size=cfg["model"].get("patch_size", 16),
        enc_mask_scale=tuple(m.get("enc_mask_scale", (0.85, 1.0))),
        pred_mask_scale=tuple(m.get("pred_mask_scale", (0.15, 0.2))),
        aspect_ratio=tuple(m.get("aspect_ratio", (0.75, 1.5))),
        nenc=m.get("nenc", 1),
        npred=m.get("npred", 4),
        min_keep_enc=m.get("min_keep_enc", 64),
        min_keep_pred=m.get("min_keep_pred", 16),
        allow_overlap=m.get("allow_overlap", False),
        d4=cfg.get("augment", {}).get("d4", True),
        seed=seed,
    )


def maybe_cache(train_ds, cfg):
    """Wrap the training dataset in an in-RAM uint8 copy, if asked for."""
    mode = cfg["data"].get("cache", "none")
    if mode in (None, "none"):
        return train_ds
    if mode != "ram_uint8":
        raise ValueError(f"Unknown data.cache {mode!r}; use 'none' or 'ram_uint8'")
    gb = train_ds[0]["image"].numel() * len(train_ds) / 1e9
    print(f"building {gb:.1f} GB uint8 RAM cache of {len(train_ds):,} tiles", flush=True)
    cache, lo, hi = build_ram_cache(
        train_ds, num_workers=cfg["training"].get("num_workers", 8)
    )
    return CachedTileDataset(cache, train_ds.tile_indices, lo, hi)


@torch.no_grad()
def embed_split(model, loader, device, amp_dtype):
    """(embeddings, tile_indices) over a loader, in its order."""
    model.eval()
    chunks, indices = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            z = model.encode(x)
        chunks.append(z.float().cpu().numpy())
        indices.append(batch["tile_index"].numpy())
    model.train()
    return np.concatenate(chunks), np.concatenate(indices)


def probe(model, loader, labels, device, amp_dtype, n_jobs=4):
    """Mean out-of-fold R^2 over the regression tasks. Higher is better."""
    z, tile_index = embed_split(model, loader, device, amp_dtype)
    results, _ = run_probes_local(
        z, tile_index, labels, n_jobs=n_jobs, tasks=PROBE_TASKS
    )
    return overall_score(results, tasks=PROBE_TASKS)


@torch.no_grad()
def held_out_loss(model, loader, collator, device, amp_dtype, max_batches, seed):
    """The I-JEPA loss on validation tiles, under a reset mask seed.

    Provenance only -- never a selection signal. The seed reset makes the masks
    roughly repeatable across evaluations; with more than one worker the order
    in which they claim the shared counter is not pinned, so treat small
    movements as noise.
    """
    collator._counter.value = seed
    model.eval()
    total, count = 0.0, 0
    for i, (batch, masks_ctx, masks_tgt) in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        masks_ctx = [m.to(device) for m in masks_ctx]
        masks_tgt = [m.to(device) for m in masks_tgt]
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            loss = model(x, masks_ctx, masks_tgt)
        total += loss.item() * x.shape[0]
        count += x.shape[0]
    model.train()
    return total / max(1, count)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/ijepa.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config entry, e.g. --set training.steps=1000")
    p.add_argument("--out", default=None, help="Run directory (default: from config)")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    t = cfg["training"]
    run_dir = Path(args.out or t.get("run_dir", "runs/ijepa"))
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = t.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(
        t.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    amp_dtype = getattr(torch, t.get("amp_dtype", "bfloat16"))

    base_ds, val_ds = build_datasets(cfg)
    tile_size = tuple(base_ds[0]["image"].shape[-2:])
    print(
        f"train: {len(base_ds)} tiles | val: {len(val_ds)} tiles | "
        f"{base_ds.n_channels} channels | {tile_size[0]}x{tile_size[1]}",
        flush=True,
    )

    # `nan_to_num(nan=0.0)` in the dataset maps a handful of thermal pixels to a
    # value that, post min-max, reads as "colder than ever observed". Harmless
    # for a reconstruction loss; for I-JEPA it is a constant the predictor could
    # memorise instead of learning physics, so the scale is worth knowing.
    zero_frac = float((base_ds[0]["image"] == 0).float().mean())
    print(f"sentinel zeros in tile 0: {zero_frac:.5%}", flush=True)

    train_ds = maybe_cache(base_ds, cfg)
    collator = build_collator(cfg, tile_size, seed)
    train_loader = build_dataloader(
        train_ds, t["batch_size"], shuffle=True,
        num_workers=t.get("num_workers", 8), drop_last=True, seed=seed,
        collate_fn=collator,
    )
    probe_loader = build_dataloader(
        val_ds, t["batch_size"], shuffle=False,
        num_workers=max(2, t.get("num_workers", 8) // 4),
    )
    val_collator = build_collator(cfg, tile_size, seed)
    val_mask_loader = build_dataloader(
        val_ds, t["batch_size"], shuffle=False, num_workers=2,
        collate_fn=val_collator,
    )
    labels = xr.open_dataset(cfg["data"]["labels_path"]).load()

    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name", "ijepa")
    model_cfg["in_channels"] = base_ds.n_channels
    model = build_model(model_name, **model_cfg).to(device)
    trainable = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(
        f"model: {model_name}/{model_cfg.get('arch')}, "
        f"{trainable / 1e6:.2f}M trainable, embedding_dim {model.embedding_dim}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        param_groups(model.context_encoder, model.predictor), lr=t["lr"]
    )

    run = None
    if args.wandb:
        # Never let a logging import kill a 12 h job: wandb is an optional extra
        # and may simply not be installed.
        try:
            import wandb

            run = wandb.init(
                project=t.get("wandb_project", "iecdt-hackathon"),
                config=cfg, dir=str(run_dir),
            )
        except Exception as exc:
            print(f"wandb unavailable ({exc}); continuing without it", flush=True)

    total_steps = t["steps"]
    warmup = t.get("warmup_steps") or int(total_steps * t.get("warmup_frac", 0.1))
    ema_start, ema_end = t.get("ema", (0.996, 1.0))
    max_seconds = t.get("max_seconds")
    log_every = t.get("log_every", 50)
    probe_every = t.get("probe_every", 2000)
    collapse_every = t.get("collapse_every", 500)

    best_score = -float("inf")
    step, started, running = 0, time.time(), []
    stop = False

    model.train()
    while step < total_steps and not stop:
        for batch, masks_ctx, masks_tgt in train_loader:
            if step >= total_steps:
                break
            if max_seconds and time.time() - started > max_seconds:
                print(
                    f"\nreached max_seconds ({max_seconds}s) at step {step}; "
                    f"stopping so the final probe and checkpoints still run",
                    flush=True,
                )
                stop = True
                break

            lr = warmup_cosine(
                step, total_steps, warmup,
                t.get("start_lr", t["lr"] / 5), t["lr"], t.get("final_lr", 1e-6),
            )
            wd = linear_schedule(
                step, total_steps,
                t.get("weight_decay", 0.04), t.get("final_weight_decay", 0.4),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
                if group.get("wd_scheduled"):
                    group["weight_decay"] = wd

            x = batch["image"].to(device, non_blocking=True)
            masks_ctx = [m.to(device, non_blocking=True) for m in masks_ctx]
            masks_tgt = [m.to(device, non_blocking=True) for m in masks_tgt]

            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = model(x, masks_ctx, masks_tgt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if t.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(
                    [q for q in model.parameters() if q.requires_grad], t["grad_clip"]
                )
            optimizer.step()

            # Outside the autocast block on purpose: see IJepa.update_target.
            momentum = linear_schedule(step, total_steps, ema_start, ema_end)
            model.update_target(momentum)

            running.append(loss.item())
            step += 1

            if step % log_every == 0:
                mean_loss = float(np.mean(running))
                running = []
                rate = step / (time.time() - started)
                print(
                    f"step {step:>7}/{total_steps}  loss {mean_loss:.6f}  "
                    f"lr {lr:.2e}  wd {wd:.3f}  m {momentum:.4f}  "
                    f"ctx {masks_ctx[0].shape[1]:>3} tgt {masks_tgt[0].shape[1]:>3}  "
                    f"{rate:.2f} it/s",
                    flush=True,
                )
                if run:
                    run.log(
                        {"train/loss": mean_loss, "train/lr": lr, "train/wd": wd,
                         "train/momentum": momentum}, step=step,
                    )

            if step % collapse_every == 0:
                stats = model.collapse_stats(x)
                print(
                    f"step {step:>7}  feature_std {stats['feature_std']:.4f}  "
                    f"token_std {stats['token_std']:.4f}  "
                    f"eff_rank {stats['effective_rank']:.1f}",
                    flush=True,
                )
                if run:
                    run.log({f"collapse/{k}": v for k, v in stats.items()}, step=step)

            if step % probe_every == 0 or step == total_steps:
                jepa_loss = held_out_loss(
                    model, val_mask_loader, val_collator, device, amp_dtype,
                    t.get("val_batches", 20), seed,
                )
                score = probe(model, probe_loader, labels, device, amp_dtype)
                print(
                    f"step {step:>7}  val jepa {jepa_loss:.6f}  "
                    f"probe mean R2 {score:.4f}"
                    + ("  <- best" if score > best_score else ""),
                    flush=True,
                )
                if run:
                    run.log({"val/jepa_loss": jepa_loss, "val/probe_r2": score},
                            step=step)

                # `val_loss` in the checkpoint is the honest held-out objective;
                # selection uses the probe score, tracked separately.
                save(run_dir / "last.pt", model, model_name, model_cfg, cfg, step,
                     jepa_loss, base_ds)
                if score > best_score:
                    best_score = score
                    # `save` reads `embedding_dim` off the live model, so the
                    # pool is swapped in for the write as well as recorded in
                    # model_cfg -- otherwise every variant would claim the
                    # default pooling's width in its metadata.
                    live_pool = model.pool
                    for tag, pool in POOL_VARIANTS.items():
                        model.pool = pool
                        save(run_dir / f"best_{tag}.pt", model, model_name,
                             model_cfg | {"pool": pool}, cfg, step, jepa_loss, base_ds)
                    model.pool = live_pool

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"\nDone at step {step}. Best probe mean R2 {best_score:.4f}.")
    print(
        f"Checkpoints in {run_dir}: "
        + ", ".join(f"best_{k}.pt" for k in POOL_VARIANTS)
    )
    print(
        "\nScore every pooling side by side with:\n"
        "  for v in " + " ".join(POOL_VARIANTS) + "; do \\\n"
        f"    python -m iecdt_hackathon.embed --checkpoint {run_dir}/best_$v.pt \\\n"
        "      --data-dir $ROOT/val --out embeddings/val --name ijepa_$v --overwrite; done\n"
        "  python -m iecdt_hackathon.evaluate --embeddings embeddings/val/ijepa_*.npz"
    )
    if run:
        run.finish()


if __name__ == "__main__":
    main()
