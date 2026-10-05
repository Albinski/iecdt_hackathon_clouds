"""Train a tile autoencoder.

    python -m iecdt_hackathon.train --config configs/default.yaml

Everything is driven by a YAML config; any key can be overridden on the command
line with `--set section.key=value`. The checkpoint written here is what
`iecdt_hackathon.evaluate` consumes, and it carries enough of the config to
rebuild the model without the YAML.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import ModisTileDataset, build_dataloader
from .models import build_model


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for override in overrides:
        key, _, value = override.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def cosine_lr(step, total_steps, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_datasets(cfg):
    d = cfg["data"]
    common = dict(
        stats_path=d["stats_path"],
        bands=tuple(d.get("bands", ("1", "3", "4", "29", "31", "32"))),
        include_land_mask=d.get("include_land_mask", False),
        include_solar_zenith=d.get("include_solar_zenith", False),
    )
    train = ModisTileDataset(d["train_dir"], crop_size=d.get("crop_size"),
                             seed=cfg["training"].get("seed", 0), **common)
    val = ModisTileDataset(d["val_dir"], crop_size=None, **common)
    if d.get("n_val_tiles"):
        val.tile_indices = val.tile_indices[: d["n_val_tiles"]]
    return train, val


@torch.no_grad()
def evaluate_reconstruction(model, loader, device, amp_dtype, max_batches=None):
    model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            loss = torch.nn.functional.mse_loss(model(x), x)
        total += loss.item() * x.shape[0]
        count += x.shape[0]
    model.train()
    return total / max(1, count)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config entry, e.g. --set training.steps=1000")
    p.add_argument("--out", default=None, help="Run directory (default: from config)")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    t = cfg["training"]
    run_dir = Path(args.out or t.get("run_dir", "runs/default"))
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(t.get("seed", 0))
    np.random.seed(t.get("seed", 0))
    device = torch.device(t.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = getattr(torch, t.get("amp_dtype", "bfloat16"))

    train_ds, val_ds = build_datasets(cfg)
    print(f"train: {len(train_ds)} tiles | val: {len(val_ds)} tiles "
          f"| {train_ds.n_channels} channels", flush=True)

    train_loader = build_dataloader(train_ds, t["batch_size"], shuffle=True,
                                    num_workers=t.get("num_workers", 8),
                                    drop_last=True, seed=t.get("seed", 0))
    val_loader = build_dataloader(val_ds, t["batch_size"], shuffle=False,
                                  num_workers=max(2, t.get("num_workers", 8) // 2))

    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name", "conv_autoencoder")
    model_cfg["in_channels"] = train_ds.n_channels
    model = build_model(model_name, **model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {model_name}, {n_params / 1e6:.2f}M trainable parameters", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"],
                                  weight_decay=t.get("weight_decay", 0.01))

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=t.get("wandb_project", "iecdt-hackathon"),
                         config=cfg, dir=str(run_dir))

    total_steps = t["steps"]
    warmup = t.get("warmup_steps", min(500, total_steps // 20))
    log_every = t.get("log_every", 50)
    val_every = t.get("val_every", 1000)
    best_val = float("inf")
    step = 0
    started = time.time()
    running = []

    model.train()
    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break
            lr = cosine_lr(step, total_steps, t["lr"], warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x = batch["image"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = torch.nn.functional.mse_loss(model(x), x)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if t.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            optimizer.step()

            running.append(loss.item())
            step += 1

            if step % log_every == 0:
                mean_loss = float(np.mean(running))
                running = []
                rate = step / (time.time() - started)
                print(f"step {step:>7}/{total_steps}  loss {mean_loss:.6f}  "
                      f"lr {lr:.2e}  {rate:.1f} it/s", flush=True)
                if run:
                    run.log({"train/loss": mean_loss, "train/lr": lr}, step=step)

            if step % val_every == 0 or step == total_steps:
                val_loss = evaluate_reconstruction(
                    model, val_loader, device, amp_dtype,
                    max_batches=t.get("val_batches", 20))
                print(f"step {step:>7}  val loss {val_loss:.6f}", flush=True)
                if run:
                    run.log({"val/loss": val_loss}, step=step)
                save(run_dir / "last.pt", model, model_name, model_cfg, cfg, step,
                     val_loss, train_ds)
                if val_loss < best_val:
                    best_val = val_loss
                    save(run_dir / "best.pt", model, model_name, model_cfg, cfg,
                         step, val_loss, train_ds)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"\nDone. Best val loss {best_val:.6f}. Checkpoints in {run_dir}")
    print(f"Evaluate with:\n  python -m iecdt_hackathon.evaluate "
          f"--checkpoint {run_dir / 'best.pt'}")
    if run:
        run.finish()


def save(path, model, model_name, model_cfg, cfg, step, val_loss, dataset):
    """Checkpoint carrying everything evaluate.py needs to rebuild the model."""
    torch.save({
        "state_dict": model.state_dict(),
        "model_name": model_name,
        "model_cfg": model_cfg,
        "config": cfg,
        "step": step,
        "val_loss": val_loss,
        "bands": list(dataset.bands),
        "include_land_mask": dataset.include_land_mask,
        "include_solar_zenith": dataset.include_solar_zenith,
        "embedding_dim": getattr(model, "embedding_dim", None),
    }, path)


if __name__ == "__main__":
    main()
