"""Score every embedding in a directory on its own and fused with `physical`.

    uv run python scripts/compare_fusion.py
    uv run python scripts/compare_fusion.py --dir embeddings/val --fuse-with physical

One row per model, two columns of score: the model alone, and the model
concatenated with the handcrafted physical features on `tile_index`. The point
of pairing them is that the two numbers answer different questions. Alone tells
you how good a representation the encoder learned; fused tells you whether it
knows anything the 144 physical features do not -- which is the only thing that
matters for a submission, since `physical` is free.

`delta` is the second minus the first. A large positive delta on a weak model
just means `physical` is carrying it; what to look for is a *fused* score above
what `physical` scores by itself.

The last column is a reweighting, not a measurement. The validation split
publishes 3 regression tasks and 1 classification task, so its mean is 75%
regression; the leaderboard scores 8 and 2, so it is 80%. For encoders whose
strength and weakness split along that line -- I-JEPA wins the regressions and
loses `task_6` -- the 4-task mean is a biased proxy, and this column says which
way. It assumes the unpublished tasks behave like the published ones of the
same kind, which is an assumption and not a fact.

Fused files are written to the embeddings directory so that whichever row wins
can go straight to `embed.py`'s test-split counterpart and `submit.sh` without
being rebuilt.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from iecdt_hackathon.concat_embeddings import concat
from iecdt_hackathon.embeddings import (
    MAX_EMBEDDING_DIM,
    load_embeddings,
    save_embeddings,
)
from iecdt_hackathon.evaluate import (
    overall_score,
    ranking_metric,
    run_probes_local,
)
from iecdt_hackathon.tasks import TASKS

ROOT = "/gws/ssde/j25b/iecdt/modis_hackathon"

#: How the leaderboard's ten tasks split, from the submission report.
LEADERBOARD_SPLIT = {"regression": 8, "classification": 2}


def leaderboard_weighted(results):
    """The 4-task scores reweighted to the leaderboard's 8:2 task mix."""
    by_kind = {"regression": [], "classification": []}
    for name, task in TASKS.items():
        value = ranking_metric(task, results[name]) if name in results else None
        if value is not None and np.isfinite(value):
            by_kind[task.kind].append(value)
    total = sum(LEADERBOARD_SPLIT.values())
    weighted, used = 0.0, 0
    for kind, n in LEADERBOARD_SPLIT.items():
        if by_kind[kind]:
            weighted += (n / total) * float(np.mean(by_kind[kind]))
            used += n
    # Renormalise if a kind had no usable task, so the figure stays on [0, 1].
    return weighted * total / used if used else float("nan")


def score(name, z, tile_index, labels, n_jobs, seed):
    print(f"\n=== {name} ===  {len(tile_index):,} tiles x {z.shape[1]} dims", flush=True)
    results, _ = run_probes_local(
        z, tile_index, labels, n_jobs=n_jobs, seed=seed
    )
    row = {
        "dim": int(z.shape[1]),
        "tasks": {
            k: ranking_metric(t, results[k]) for k, t in TASKS.items() if k in results
        },
        "overall": overall_score(results),
        "leaderboard_weighted": leaderboard_weighted(results),
    }
    print(f"  OVERALL {row['overall']:.4f}"
          f"   (leaderboard-weighted {row['leaderboard_weighted']:.4f})", flush=True)
    return row


def discover(directory, fuse_with, suffix):
    """Base embedding files: not the fusion partner, not already fused."""
    return sorted(
        p for p in Path(directory).glob("*.npz")
        if p.stem != fuse_with and not p.stem.endswith(suffix)
    )


def _row(label, dim, tasks, alone, fused, delta, weighted):
    """One markdown row. Every cell is already a string or None."""
    cells = [f"`{label}`", dim] + tasks + [alone, fused, delta, weighted]
    return "| " + " | ".join("\u2014" if c is None else c for c in cells) + " |"


def _fmt(value):
    return None if value is None or not np.isfinite(value) else f"{value:.4f}"


def table(rows, fuse_with, reference):
    """The paired comparison, best fused score first.

    Per-task columns show the fused scores where a fusion exists, because those
    are the numbers a submission would actually be made of. The `alone` column
    keeps the unfused overall so the pair stays visible.
    """
    task_names = list(TASKS)
    header = _row("Model", "D", task_names, "alone", f"+{fuse_with}",
                  "delta", "lb-weighted").replace("`Model`", "Model")
    lines = [header, "|" + "---|" * (len(task_names) + 6)]

    def sort_key(name):
        fused = rows[name]["fused"]
        return -(fused or rows[name]["alone"])["overall"]

    for name in sorted(rows, key=sort_key):
        alone, fused = rows[name]["alone"], rows[name]["fused"]
        shown = fused or alone
        lines.append(_row(
            name,
            f"{alone['dim']}\u2192{fused['dim']}" if fused else str(alone["dim"]),
            [_fmt(shown["tasks"].get(k)) for k in task_names],
            _fmt(alone["overall"]),
            _fmt(fused["overall"]) if fused else None,
            f"{fused['overall'] - alone['overall']:+.4f}" if fused else None,
            _fmt(shown["leaderboard_weighted"]),
        ))

    lines.append(_row(
        f"{fuse_with} (reference)", str(reference["dim"]),
        [_fmt(reference["tasks"].get(k)) for k in task_names],
        _fmt(reference["overall"]), None, None,
        _fmt(reference["leaderboard_weighted"]),
    ))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dir", default="embeddings/val")
    p.add_argument("--fuse-with", default="physical",
                   help="stem of the embedding every model is fused with")
    p.add_argument("--suffix", default="_phys",
                   help="suffix for written fused files, also used to skip them")
    p.add_argument("--labels", default=f"{ROOT}/labels/val_labels.nc")
    p.add_argument("--out", default="results/comparison")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", nargs="*", help="limit to these stems")
    args = p.parse_args()

    directory = Path(args.dir)
    partner_path = directory / f"{args.fuse_with}.npz"
    if not partner_path.exists():
        raise SystemExit(
            f"{partner_path} not found. Build it with\n"
            f"  uv run python scripts/physical_features.py "
            f"--split {directory.name}"
        )
    partner = load_embeddings(partner_path, name=args.fuse_with)
    labels = xr.open_dataset(args.labels).load()

    bases = discover(directory, args.fuse_with, args.suffix)
    if args.only:
        bases = [p for p in bases if p.stem in args.only]
    print(f"{len(bases)} models to compare, fused with "
          f"{args.fuse_with} ({partner.dim} dims)", flush=True)

    reference = score(args.fuse_with, partner.z, partner.tile_index,
                      labels, args.n_jobs, args.seed)

    rows = {}
    for path in bases:
        emb = load_embeddings(path, name=path.stem)
        alone = score(path.stem, emb.z, emb.tile_index, labels,
                      args.n_jobs, args.seed)

        fused_dim = emb.dim + partner.dim
        if fused_dim > MAX_EMBEDDING_DIM:
            print(f"  skipping fusion: {emb.dim} + {partner.dim} = {fused_dim} "
                  f"exceeds the {MAX_EMBEDDING_DIM}-dim submission cap",
                  flush=True)
            rows[path.stem] = {"alone": alone, "fused": None}
            continue

        z, tile_index = concat({path.stem: emb, args.fuse_with: partner})
        out_path = directory / f"{path.stem}{args.suffix}.npz"
        save_embeddings(out_path, z, tile_index,
                        split=emb.meta.get("split") or directory.name,
                        sources={path.stem: str(path),
                                 args.fuse_with: str(partner_path)})
        fused = score(f"{path.stem}{args.suffix}", z, tile_index, labels,
                      args.n_jobs, args.seed)
        fused["path"] = str(out_path)
        rows[path.stem] = {"alone": alone, "fused": fused}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(
        {"fuse_with": args.fuse_with, "reference": reference, "models": rows},
        indent=2))

    print("\n" + table(rows, args.fuse_with, reference))
    print(f"\nWrote {out / 'comparison.json'}")
    print(f"\nPer-task columns are the *fused* scores where a fusion exists, "
          f"the model's own otherwise. `delta` is +{args.fuse_with} minus alone.")


if __name__ == "__main__":
    main()
