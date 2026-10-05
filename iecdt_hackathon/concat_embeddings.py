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
    args = p.parse_args()

    if len(args.inputs) < 2:
        raise SystemExit("--inputs needs at least two files to concatenate")

    submissions = load_all(args.inputs)
    require_same_tiles(submissions)
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
