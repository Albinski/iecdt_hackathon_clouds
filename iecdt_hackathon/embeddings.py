"""The submission format: one embedding per tile, in a `.npz` file.

A submission is a **file of embeddings**, not a model: `embed.py` runs your
encoder over a split and writes the vectors here, and everything downstream
reads only this file -- which is why `evaluate.py` imports neither torch nor the
dataset.

    embeddings   (n_tiles, embedding_dim) float32
    tile_index   (n_tiles,) int64, the tile each row came from
    meta         a JSON blob: the split, the checkpoint, when, and the digest

**`tile_index` is the contract.** Every join downstream is by tile index and
never by row order, so a shuffled file scores identically. A *missing* tile is
not tolerated: `tiles_digest` fingerprints the set, and the organisers' scorer
refuses submissions whose digests disagree, since probes fitted on different
tiles are not comparable.

`meta` is provenance, not input: nothing reads it to decide how to score.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

#: Bumped if the layout of the file ever changes incompatibly.
FORMAT_VERSION = 1

#: The widest embedding a submission may carry.
#:
#: A cap is needed because nothing else in the protocol bounds `D`, and the two
#: ends of the range fail differently. A tile is 6 x 256 x 256 = 393,216
#: numbers, so an "embedding" of that order is a copy of its input rather than a
#: representation of it -- it would score well while answering none of the
#: question the benchmark asks. Well before that it becomes a denial of service
#: against the scorer, since every probe is fitted on every dimension.
#:
#: 4,096 is sixteen times the reference encoder's 256 and still fits in seconds,
#: so it constrains nobody doing the intended thing. It is checked on write as
#: well as read, so a participant meets it when they build the file rather than
#: when they submit it.
MAX_EMBEDDING_DIM = 4096


def tiles_digest(tile_indices):
    """Short fingerprint of *which* tiles a file covers, order-independent."""
    joined = ",".join(str(int(ix)) for ix in sorted(tile_indices)).encode()
    return hashlib.sha256(joined).hexdigest()[:16]


@dataclass
class Embeddings:
    """One submission's embeddings for one split."""

    name: str
    z: np.ndarray
    tile_index: np.ndarray
    meta: dict = field(default_factory=dict)
    path: Path = None

    @property
    def dim(self):
        return int(self.z.shape[1])

    @property
    def digest(self):
        return tiles_digest(self.tile_index)

    def __len__(self):
        return len(self.tile_index)


def save_embeddings(path, z, tile_indices, **meta):
    """Write one submission file, creating its directory if need be."""
    z = np.asarray(z, dtype=np.float32)
    tile_indices = np.asarray(tile_indices, dtype=np.int64)
    _validate(z, tile_indices, path)
    meta = {
        "format_version": FORMAT_VERSION,
        "n_tiles": int(len(tile_indices)),
        "embedding_dim": int(z.shape[1]),
        "tiles_digest": tiles_digest(tile_indices),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    } | {k: v for k, v in meta.items() if v is not None}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=z, tile_index=tile_indices,
                        meta=np.array(json.dumps(meta)))
    return meta


def load_embeddings(path, name=None):
    """Read one submission file, checking it before anything is fitted on it.

    The checks are cheap and the failures they catch are not: a file with a
    non-finite row makes every probe on it fail with a message about sklearn
    internals, and one with a repeated tile index silently double-weights a
    tile and can place the same tile in a probe's fitting folds and in the fold
    it is scored on.
    """
    path = Path(path)
    try:
        f = np.load(path, allow_pickle=False)
    except Exception as exc:
        # numpy's own message here is about pickles or zip headers and sends a
        # participant looking in the wrong place entirely. The likely causes are
        # a file that is not an .npz at all, or one truncated in transit.
        raise ValueError(
            f"{path}: could not be read as an .npz file ({exc}). Write it with "
            f"`iecdt_hackathon.embed`, and check it arrived whole.") from None
    with f:
        missing = [k for k in ("embeddings", "tile_index") if k not in f]
        if missing:
            raise ValueError(f"{path}: not an embedding file -- no {missing} "
                             f"array (found {sorted(f.files)})")
        z = np.asarray(f["embeddings"], dtype=np.float64)
        tile_index = np.asarray(f["tile_index"]).astype(np.int64)
        meta = json.loads(str(f["meta"])) if "meta" in f else {}
    _validate(z, tile_index, path)
    return Embeddings(name=name or path.stem, z=z, tile_index=tile_index,
                      meta=meta, path=path)


def _validate(z, tile_index, path):
    if z.ndim != 2:
        raise ValueError(f"{path}: embeddings must be (n_tiles, dim), got "
                         f"shape {z.shape}. `model.encode(x)` returns (B, D).")
    if z.shape[1] == 0:
        # Every other check passes a (n, 0) array: it is finite, it is 2-D, and
        # it has a row per tile. The probes are where it fails, one obscure
        # sklearn message per task, so it is refused here instead.
        raise ValueError(f"{path}: the embeddings are zero-dimensional "
                         f"({z.shape}). `model.encode(x)` must return at least "
                         f"one feature per tile.")
    if z.shape[1] > MAX_EMBEDDING_DIM:
        raise ValueError(f"{path}: embedding dimension {z.shape[1]} exceeds the "
                         f"limit of {MAX_EMBEDDING_DIM}. Pool or project your "
                         f"encoder's output down before submitting.")
    if len(z) != len(tile_index):
        raise ValueError(f"{path}: {len(z)} embeddings for "
                         f"{len(tile_index)} tile indices")
    if len(z) == 0:
        raise ValueError(f"{path}: no embeddings")
    if not np.isfinite(z).all():
        bad = int((~np.isfinite(z).all(axis=1)).sum())
        raise ValueError(f"{path}: {bad} of {len(z)} embeddings contain NaN or "
                         f"inf. A probe cannot be fitted on them; find them "
                         f"before submitting.")
    if len(np.unique(tile_index)) != len(tile_index):
        dupes = len(tile_index) - len(np.unique(tile_index))
        raise ValueError(f"{path}: {dupes} repeated tile indices. Each tile "
                         f"must appear exactly once.")


def parse_spec(text):
    """`NAME=PATH`, or `PATH` to name the submission after the file."""
    name, sep, path = text.partition("=")
    if not sep:
        path, name = name, Path(name).stem
    if not Path(path).exists():
        raise SystemExit(f"--embeddings {text}: {path} does not exist")
    return name, Path(path)


def load_all(specs):
    """Load several `[NAME=]PATH` submissions, keyed by name and in order."""
    out = {}
    for spec in specs:
        name, path = parse_spec(spec)
        if name in out:
            raise SystemExit(f"Two submissions are called '{name}'; pass "
                             f"NAME=PATH to tell them apart.")
        out[name] = load_embeddings(path, name=name)
    return out


def require_same_tiles(submissions):
    """Refuse a field of submissions that do not cover the same tiles.

    Ranking them against each other assumes they were fitted and scored on the
    same rows; a submission that dropped tiles would otherwise be ranked on an
    easier or simply different subset, and nothing later in the pipeline would
    notice.
    """
    digests = {name: e.digest for name, e in submissions.items()}
    if len(set(digests.values())) > 1:
        lines = "\n".join(f"  {n:<24} {len(submissions[n]):>7,} tiles  {d}"
                          for n, d in digests.items())
        raise SystemExit(
            "Submissions cover different sets of tiles, so they cannot be "
            f"ranked against each other:\n{lines}\n"
            "Every submission must embed the whole split it was given.")
    return next(iter(digests.values()))
