"""Join several embedding files into one wider submission.

    python -m iecdt_hackathon.concat_embeddings \
        --inputs embeddings/val/ijepa_meanstd.npz embeddings/val/handcrafted.npz \
        --out embeddings/val/fused.npz

Why this exists: in this repo's experiment log the 55 handcrafted per-tile
statistics and the learned encoder fail in different places -- the statistics win
`task_6` by about +0.10 and lose `task_7` by about -0.06. Errors that
complementary are worth concatenating rather than choosing between, and a linear
probe over the union can use whichever half carries the signal for each task.
55 + 768 = 823 dimensions, comfortably inside the 4,096 cap.

Rows are matched by `tile_index`, never by position -- that is the documented
join key for everything downstream, so the inputs may be in any order. The
result is written through `save_embeddings`, so every submission check that
applies to a model's own output applies here too.
"""

import argparse
from pathlib import Path

import numpy as np

from .embeddings import load_all, require_same_tiles, save_embeddings



def require_same_split(submissions, allow_unknown=False):
    """Refuse to join files from different splits.

    This is the one error this script cannot otherwise catch, and it is silent.
    `val` and `test` carry the **same** tile indices -- both are 0-9999, so both
    have the same `tiles_digest` -- which means joining a test embedding to a
    val feature file passes `require_same_tiles`, passes every validity check,
    and writes a file that looks perfect. Every row then pairs one tile's
    learned embedding with a different tile's handcrafted features, and the
    probes score catastrophically negative rather than merely badly.

    `embed.py` records the split it read in each file's metadata, so a mismatch
    is detectable whenever both sides carry it. A file with no split recorded
    cannot be checked, so it is refused by default rather than waved through:
    the whole point is to fail here instead of on the leaderboard.
    """
    splits = {}
    for name, sub in submissions.items():
        split = sub.meta.get("split")
        splits[name] = Path(split).name if split else None

    unknown = [n for n, s in splits.items() if s is None]
    if unknown and not allow_unknown:
        raise SystemExit(
            f"These inputs do not record which split they came from: "
            f"{', '.join(unknown)}.\n"
            f"val and test share tile indices 0-9999, so a cross-split join "
            f"cannot be detected from the tile indices alone and would score "
            f"catastrophically. Rebuild them with a tool that records the "
            f"split, or pass --allow-unknown-split if you are certain."
        )

    distinct = {s for s in splits.values() if s is not None}
    if len(distinct) > 1:
        lines = "\n".join(f"  {n:<24} {s or 'unknown'}" for n, s in splits.items())
        raise SystemExit(
            f"Refusing to join embeddings from different splits:\n{lines}\n"
            f"Every input must come from the same split. Embed the split you "
            f"are submitting with every model, then join those."
        )
    return distinct.pop() if distinct else None


def concat(submissions):
    """-> (z, tile_index) with every input's columns, in the order given.

    Each input is reindexed to one common tile order first. `require_same_tiles`
    has already established that they cover the same set, so this is a
    permutation rather than a join that can drop rows.
    """
    order = None
    columns = []
    for sub in submissions.values():
        rank = np.argsort(sub.tile_index, kind="stable")
        if order is None:
            order = sub.tile_index[rank]
        columns.append(sub.z[rank])
    return np.concatenate(columns, axis=1), order


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--inputs", nargs="+", required=True, metavar="[NAME=]PATH",
                   help="Two or more embedding files to concatenate")
    p.add_argument("--out", required=True, help="Where to write the joined file")
    p.add_argument(
        "--allow-unknown-split", action="store_true",
        help="Join inputs that do not record their split. Only safe if you "
             "built them yourself and know they match; see require_same_split.",
    )
    args = p.parse_args()

    if len(args.inputs) < 2:
        raise SystemExit("--inputs needs at least two files to concatenate")

    submissions = load_all(args.inputs)
    require_same_tiles(submissions)
    split = require_same_split(submissions, allow_unknown=args.allow_unknown_split)
    if split:
        print(f"  all inputs from the {split!r} split", flush=True)
    for name, sub in submissions.items():
        print(f"  {name:<24} {len(sub):>7,} tiles x {sub.dim:>5} dims", flush=True)

    z, tile_index = concat(submissions)
    written = save_embeddings(
        Path(args.out), z, tile_index,
        sources={n: str(s.path) for n, s in submissions.items()},
        source_dims={n: s.dim for n, s in submissions.items()},
    )
    print(
        f"\nwrote {args.out}  ({written['n_tiles']:,} tiles x "
        f"{written['embedding_dim']} dims, tiles {written['tiles_digest']})",
        flush=True,
    )
    print(
        "\nScore it against its parts with:\n"
        "  python -m iecdt_hackathon.evaluate --embeddings "
        + " ".join(f"{n}={s.path}" for n, s in submissions.items())
        + f" fused={args.out}"
    )


if __name__ == "__main__":
    main()
