import json
from pathlib import Path
from typing import Iterator, Optional, Union

MARKER_NAME = ".tile_layout.json"
DEFAULT_SHARD_SIZE = 10_000

# Sentinel: distinguishes "not passed" from an explicit None (= flat).
_UNSET = object()


def read_layout(data_dir: Union[str, Path]) -> Optional[int]:
    """Return the shard size if `data_dir` uses the sharded layout, else None.

    Call once per process/dataset init and pass the result to `tile_path`;
    avoid calling per tile.
    """
    marker = Path(data_dir) / MARKER_NAME
    if not marker.exists():
        return None
    with open(marker) as f:
        layout = json.load(f)
    if layout.get("layout", "flat") != "sharded":
        return None
    return int(layout["shard_size"])


def write_marker(
    data_dir: Union[str, Path], shard_size: int = DEFAULT_SHARD_SIZE
) -> None:
    """Declare `data_dir` as sharded. Idempotent; errors on a conflicting marker."""
    existing = read_layout(data_dir)
    if existing is not None:
        if existing != shard_size:
            raise ValueError(
                f"{data_dir} already declares shard_size={existing}, "
                f"refusing to overwrite with shard_size={shard_size}"
            )
        return
    marker = Path(data_dir) / MARKER_NAME
    if marker.exists():
        raise ValueError(f"{marker} exists but does not declare a sharded layout")
    with open(marker, "w") as f:
        json.dump({"version": 1, "layout": "sharded", "shard_size": shard_size}, f)


def shard_dir(data_dir: Union[str, Path], tile_ix: int, shard_size: int) -> Path:
    return Path(data_dir) / f"{tile_ix // shard_size:05d}"


def tile_path(
    data_dir: Union[str, Path],
    tile_ix: int,
    suffix: str = "",
    shard_size=_UNSET,
) -> Path:
    """Path of f"{tile_ix}{suffix}.nc" under `data_dir` in the declared layout.

    suffix: "" | "_era5" | "_clouds" | "_destriped" | "_ccfs" | "_sgff".
    shard_size: cached result of `read_layout(data_dir)` (None = flat). If not
    passed, `read_layout` is consulted — fine for one-off calls, but loops
    should cache.
    """
    if shard_size is _UNSET:
        shard_size = read_layout(data_dir)
    fname = f"{tile_ix}{suffix}.nc"
    if shard_size is None:
        return Path(data_dir) / fname
    return shard_dir(data_dir, tile_ix, shard_size) / fname


def iter_tile_files(data_dir: Union[str, Path], pattern: str) -> Iterator[Path]:
    """Yield files matching `pattern` from `data_dir` and its shard subdirs.

    Layout-agnostic: works on flat, sharded, and mixed directories. Shard
    subdirs are recognized as all-digit directory names.
    """
    data_dir = Path(data_dir)
    for p in data_dir.glob(pattern):
        if p.name != MARKER_NAME:
            yield p
    for sub in sorted(data_dir.iterdir()):
        if sub.is_dir() and sub.name.isdigit():
            yield from sub.glob(pattern)
