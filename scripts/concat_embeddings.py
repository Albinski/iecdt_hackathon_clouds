"""Concatenate several embedding files side by side, aligned by tile index.

    uv run python scripts/concat_embeddings.py embeddings/val/combo.npz \
        embeddings/val/d512_w64.npz embeddings/val/handcrafted.npz
"""

import sys

import numpy as np

from iecdt_hackathon.embeddings import load_embeddings, save_embeddings

out, *inputs = sys.argv[1:]
embs = [load_embeddings(p) for p in inputs]
tiles = np.sort(embs[0].tile_index)
for e in embs:
    if not np.array_equal(np.sort(e.tile_index), tiles):
        sys.exit(f"{e.path} covers different tiles from {embs[0].path}")

blocks = []
for e in embs:
    order = np.argsort(e.tile_index)
    blocks.append(e.z[order])
z = np.concatenate(blocks, axis=1)
save_embeddings(out, z, tiles, model="concat", parts=",".join(map(str, inputs)))
print(f"Wrote {out}  ({z.shape[0]} tiles x {z.shape[1]} dims = "
      + " + ".join(str(e.dim) for e in embs) + ")")
