"""Run a trained model over a split's tiles and write the embeddings.

    python -m iecdt_hackathon.embed --checkpoint runs/baseline/best.pt \
        --data-dir /gws/ssde/j25b/iecdt/modis_hackathon/val \
        --out embeddings/val
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import ModisTileDataset, build_dataloader
from .embeddings import save_embeddings
from .models import build_model


def load_checkpoint_model(checkpoint_path, device):
    """Rebuild a trained model plus the dataset settings it was trained with."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = dict(ckpt.get("model_cfg", {}))
    model = build_model(ckpt.get("model_name", "conv_autoencoder"), **model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    settings = {
        "bands": tuple(ckpt["bands"]),
        "include_land_mask": ckpt.get("include_land_mask", False),
        "include_solar_zenith": ckpt.get("include_solar_zenith", False),
    }
    return model, settings, ckpt


@torch.no_grad()
def extract_embeddings(
    model, dataset, device, batch_size=64, num_workers=8, amp_dtype=torch.bfloat16
):
    """(embeddings, tile_indices) in dataset order."""
    loader = build_dataloader(
        dataset, batch_size, shuffle=False, num_workers=num_workers
    )
    chunks, indices = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            z = model.encode(x)
        chunks.append(z.float().cpu().numpy())
        indices.append(batch["tile_index"].numpy())
    return np.concatenate(chunks), np.concatenate(indices)


def write_embeddings(model, dataset, device, out_path, args, **meta):
    """Extract one encoder's embeddings and write them, unless already written.

    An existing file is kept rather than recomputed: extraction is the
    expensive half of the pipeline and probing the same vectors again is the
    common case. `--overwrite` is the way to redo it after retraining.
    """
    out_path = Path(out_path)
    if out_path.exists() and not args.overwrite:
        print(f"  {out_path} exists; keeping it (--overwrite to redo)", flush=True)
        return out_path
    z, ix = extract_embeddings(
        model, dataset, device, args.batch_size, args.num_workers
    )
    written = save_embeddings(out_path, z, ix, split=str(args.data_dir), **meta)
    print(
        f"  wrote {out_path}  ({written['n_tiles']:,} tiles x "
        f"{written['embedding_dim']} dims, tiles {written['tiles_digest']})",
        flush=True,
    )
    return out_path


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = "/gws/ssde/j25b/iecdt/modis_hackathon"
    p.add_argument("--checkpoint", required=True, help="The model to embed with")
    p.add_argument(
        "--data-dir",
        default=f"{root}/val",
        help="Directory of tiles to embed (a whole split)",
    )
    p.add_argument("--stats-path", default=f"{root}/stats/modis_band_stats.json")
    p.add_argument(
        "--out", default="embeddings", help="Directory the .npz files are written to"
    )
    p.add_argument(
        "--name",
        default=None,
        help="Name for this submission's file (default: the checkpoint's stem)",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Embed only the first N tiles (for a quick check). A "
        "submission must cover the whole split, so never pass "
        "this for the file you send in.",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)

    model, settings, ckpt = load_checkpoint_model(args.checkpoint, device)
    dataset = ModisTileDataset(args.data_dir, stats_path=args.stats_path, **settings)
    if args.limit:
        dataset.tile_indices = dataset.tile_indices[: args.limit]
    print(
        f"{args.data_dir}: {len(dataset)} tiles, {dataset.n_channels} channels "
        f"| device {device.type}",
        flush=True,
    )

    entry_name = args.name or Path(args.checkpoint).stem
    print(f"\n=== {entry_name} ===", flush=True)
    write_embeddings(
        model,
        dataset,
        device,
        out / f"{entry_name}.npz",
        args,
        checkpoint=str(args.checkpoint),
        model_name=ckpt.get("model_name"),
        train_step=ckpt.get("step"),
    )

    print(
        f"\nNext: score them locally with\n"
        f"  python -m iecdt_hackathon.evaluate --embeddings {out}/*.npz",
        flush=True,
    )


if __name__ == "__main__":
    main()
