import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.parallel import Parallel, delayed

from .embeddings import load_all
from .tasks import TASKS

#: Folds the labelled tiles are cut into. Both protocols use the same K, so a
#: local number and a leaderboard number are the same arithmetic.
N_FOLDS = 5

MIN_TILES = 50

NOT_PUBLISHED = "no labels published for this split"

TOO_FEW = "too few labelled tiles"

SINGLE_CLASS = "only one class among the labelled tiles"


def _skipped(reason, n_usable=0):
    return {"skipped": reason, "n_usable": int(n_usable)}


def join_labels(tile_indices, labels_ds, variable):
    """Label values aligned to `tile_indices`, by tile index rather than order."""
    index = labels_ds["tile_index"].values.astype(np.int64)
    position = {int(ix): i for i, ix in enumerate(index)}
    rows = np.array([position.get(int(ix), -1) for ix in tile_indices])
    values = np.asarray(labels_ds[variable].values)
    out = np.full(len(tile_indices), np.nan, dtype=np.float64)
    found = rows >= 0
    out[found] = values[rows[found]]
    return out, found


def usable(task, y):
    """Mask of rows this task can be fitted or scored on."""
    ok = np.isfinite(y)
    for value in task.drop_values:
        ok &= y != value
    return ok


PROBE_FOLD_SALT = "iecdt-probe-folds-v1"

LOCAL_FOLD_SALT = "iecdt-local-folds-v1"


def _tile_hash(tile_index, salt=PROBE_FOLD_SALT):
    digest = hashlib.sha256(f"{salt}:{int(tile_index)}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def probe_folds(tile_indices, n_folds=N_FOLDS, salt=PROBE_FOLD_SALT):
    """Assign each tile to one of `n_folds` cross-validation folds.

    Returns an integer array over `tile_indices`. A probe is fitted on the tiles
    outside a fold and scored on the tiles in it, once per fold, so every tile
    is scored exactly once.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    return np.array(
        [_tile_hash(ix, salt) % n_folds for ix in tile_indices], dtype=np.int64
    )


def local_folds(tile_indices, n_folds=N_FOLDS):
    """The participants' folds: the same rule under a different salt."""
    return probe_folds(tile_indices, n_folds=n_folds, salt=LOCAL_FOLD_SALT)


def folds_digest(tile_indices, folds):
    """Short fingerprint of exactly which tile went into which fold."""
    pairs = sorted((int(ix), int(k)) for ix, k in zip(tile_indices, folds))
    joined = ",".join(f"{ix}:{k}" for ix, k in pairs).encode()
    return hashlib.sha256(joined).hexdigest()[:16]


def ranking_metric(task, metrics):
    if "skipped" in metrics:
        return None
    return metrics.get("r2" if task.kind == "regression" else "macro_f1")


def make_probe(task, seed=0):
    if task.kind == "regression":
        return make_pipeline(StandardScaler(), LinearRegression())
    return make_pipeline(
        StandardScaler(),
        # lbfgs is multinomial by default for multi-class targets. No penalty,
        # so max_iter is generous: an unregularised fit on near-separable data
        # converges slowly, and the budget freed by dropping the penalty search
        # is more than enough to pay for it.
        LogisticRegression(
            C=np.inf, max_iter=5000, class_weight="balanced", random_state=seed
        ),
    )


def score_predictions(task, y_true, pred):
    """Metrics for one task from true values and probe predictions."""
    if task.kind == "regression":
        return {
            "r2": float(r2_score(y_true, pred)),
            "mae": float(mean_absolute_error(y_true, pred)),
            "target_std": float(np.std(y_true)),
        }

    y_true = y_true.astype(int)
    pred = np.asarray(pred).astype(int)
    present = sorted(set(y_true.tolist()))
    with warnings.catch_warnings():
        # A probe may predict a class absent from the scored rows; balanced
        # accuracy handles that correctly, it just says so loudly.
        warnings.filterwarnings("ignore", message="y_pred contains classes not in")
        balanced = balanced_accuracy_score(y_true, pred)
        macro_f1 = f1_score(
            y_true, pred, average="macro", labels=present, zero_division=0
        )
    return {
        "balanced_accuracy": float(balanced),
        "macro_f1": float(macro_f1),
        "accuracy": float((pred == y_true).mean()),
        "n_classes_present": len(present),
        "confusion": _confusion(y_true, pred, task.n_classes).tolist(),
    }


def _predict_one_fold(task, z, y, fit, score, seed):
    """Fit on the tiles outside one fold, predict the tiles in it.

    Predictions only: the metric is computed once over every fold's
    predictions together, not fold by fold.
    """
    probe = make_probe(task, seed=seed)
    probe.fit(z[fit], y[fit])
    return probe.predict(z[score])


def fit_probe(task, z, y, folds, n_jobs=-1, seed=0):
    """Cross-validate one task's probe and score the pooled predictions.

    `folds` is the fold id of each row of `z`, already restricted to the rows
    this task can use. Each fold is predicted by a probe that never saw it, the
    out-of-fold predictions are **pooled**, and the metric is computed once over
    all of them -- so every usable tile contributes one prediction to one
    number.

    Pooled rather than averaged over the folds. For macro-F1 the difference
    matters: a fold holds a fifth of the tiles and a class under 1% of its task
    may be nearly absent from one, so that fold's macro-F1 is not the same
    quantity as the next's. Pooling puts every rare-class row into one metric.

    The folds are parallelised rather than the fit inside one: each fit is a
    single unregularised model, so there is nothing to search and the folds are
    the only parallelism left.
    """
    if task.kind == "classification":
        y = y.astype(int)
    jobs = []
    for k in sorted({int(k) for k in folds}):
        score = folds == k
        fit = ~score
        if not score.any() or not fit.any():
            continue
        if task.kind == "classification" and len(set(y[fit].tolist())) < 2:
            # LogisticRegression cannot be fitted on one class, and a fold whose
            # fitting rows are all one class says nothing about the embedding.
            # Its rows get no prediction, so they stay out of the pool.
            continue
        jobs.append((fit, score))
    if len(jobs) < 2:
        return None
    predictions = Parallel(n_jobs=n_jobs)(
        delayed(_predict_one_fold)(task, z, y, fit, score, seed)
        for fit, score in jobs
    )

    pooled = np.empty(
        len(y), dtype=np.int64 if task.kind == "classification" else np.float64
    )
    scored = np.zeros(len(y), dtype=bool)
    for (_, score), pred in zip(jobs, predictions):
        pooled[score] = pred
        scored |= score
    metrics = score_predictions(task, y[scored], pooled[scored])
    return metrics | {
        "n_folds": len(jobs),
        # Each pooled tile was predicted exactly once, so this is the task's
        # row count -- short of the usable rows only if a fold was unfittable.
        "n_usable": int(scored.sum()),
        "n_fit_mean": float(np.mean([int(fit.sum()) for fit, _ in jobs])),
    }


def _confusion(y_true, y_pred, n_classes):
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            m[t, p] += 1
    return m


def _report(name, task, metrics):
    if "skipped" in metrics:
        # The reason matters: a task with no published labels for this split and
        # a task with too few usable tiles are different situations, and only
        # one of them is something you can do anything about.
        counts = f" ({metrics['n_usable']} usable tiles)" if metrics["n_usable"] else ""
        print(f"  {name:<10} skipped: {metrics['skipped']}{counts}", flush=True)
        return
    # The ranking metric alone: that is the number which places a submission on
    # the leaderboard. MAE, balanced accuracy, accuracy and the confusion matrix
    # are still computed and still in `results.json`.
    headline = (
        f"R2 {metrics['r2']:+.4f}"
        if task.kind == "regression"
        else f"macro-F1 {metrics['macro_f1']:.4f}"
    )
    print(f"  {name:<10} {headline}  [{metrics['n_folds']} folds]", flush=True)


def run_probes(z, tile_indices, labels, folds, n_jobs=-1, seed=0, tasks=None):
    """Cross-validate every task's probe over one labelled split."""
    results = {}
    for name, task in (tasks or TASKS).items():
        if task.variable not in labels:
            results[name] = _skipped(NOT_PUBLISHED)
            _report(name, task, results[name])
            continue
        y, found = join_labels(tile_indices, labels, task.variable)
        mask = usable(task, y) & found
        n_usable = int(mask.sum())

        if n_usable < MIN_TILES:
            results[name] = _skipped(TOO_FEW, n_usable)
        elif task.kind == "classification" and len(set(y[mask].astype(int))) < 2:
            results[name] = _skipped(SINGLE_CLASS, n_usable)
        else:
            metrics = fit_probe(
                task, z[mask], y[mask], folds[mask], n_jobs=n_jobs, seed=seed
            )
            results[name] = (
                metrics if metrics is not None else _skipped(TOO_FEW, n_usable)
            )
        _report(name, task, results[name])
    return results


def run_probes_local(
    z, tile_indices, labels, n_folds=N_FOLDS, n_jobs=-1, seed=0, tasks=None
):
    """`run_probes` over a participant's own labelled split.

    Returns `(results, folds)`. The folds are fixed, so this is repeatable
    without a seed to carry around; `folds_digest` records which folding
    produced a results file.
    """
    folds = local_folds(tile_indices, n_folds=n_folds)
    return run_probes(
        z, tile_indices, labels, folds, n_jobs=n_jobs, seed=seed, tasks=tasks
    ), folds


def overall_score(results, tasks=None):
    values = [
        ranking_metric(task, results[name])
        for name, task in (tasks or TASKS).items()
        if name in results
    ]
    scored = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(scored)) if scored else float("nan")


def add_probe_arguments(p):
    """Probe settings shared by this script and the organisers' scorer."""
    p.add_argument(
        "--folds",
        type=int,
        default=N_FOLDS,
        help="Cross-validation folds the labelled tiles are cut "
        "into (default %(default)s)",
    )
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plots", action="store_true")
    return p


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    root = "/gws/ssde/j25b/iecdt/modis_hackathon"
    p.add_argument(
        "--embeddings",
        nargs="+",
        required=True,
        metavar="[NAME=]PATH",
        help="Embedding files written by `embed.py`. Several are "
        "scored side by side; `NAME=PATH` names a row.",
    )
    p.add_argument("--labels", default=f"{root}/labels/val_labels.nc")
    p.add_argument("--out", default="results")
    add_probe_arguments(p)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    labels = xr.open_dataset(args.labels).load()

    submissions = load_all(args.embeddings)
    all_results = {}
    for name, emb in submissions.items():
        print(f"\n=== {name} ===", flush=True)
        print(f"  {len(emb):,} tiles x {emb.dim} dims  ({emb.path})", flush=True)
        results, folds = run_probes_local(
            emb.z,
            emb.tile_index,
            labels,
            n_folds=args.folds,
            n_jobs=args.n_jobs,
            seed=args.seed,
        )
        all_results[name] = {
            "embeddings": str(emb.path),
            "embedding_dim": emb.dim,
            "n_tiles": len(emb),
            "tiles_digest": emb.digest,
            "folds_digest": folds_digest(emb.tile_index, folds),
            "tasks": results,
            "overall_score": overall_score(results),
        }
        print(f"  OVERALL {all_results[name]['overall_score']:.4f}", flush=True)

    n_tiles = max(len(e) for e in submissions.values())
    print(
        f"\n{args.folds}-fold cross-validation over {n_tiles:,} labelled tiles: "
        f"every tile is predicted by a probe that never saw it, and each task's "
        f"score is one metric over all of those predictions pooled. The folds "
        f"are fixed, so rerunning this gives the same numbers.",
        flush=True,
    )
    record = {
        "protocol": "cross_validation",
        "labels": str(args.labels),
        "n_tiles": n_tiles,
        "n_folds": args.folds,
        "results": all_results,
    }
    write_outputs(out, record, no_plots=args.no_plots)


def write_outputs(out, record, no_plots=False):
    (out / "results.json").write_text(json.dumps(record, indent=2))
    print(f"\nWrote {out / 'results.json'}")
    if not no_plots:
        try:
            write_plots(out, record)
        except Exception as exc:  # plotting must never fail a scoring run
            print(f"Plotting skipped: {exc}")


def write_plots(out, record, names=None):
    """Per-task score bar chart comparing every scored entry."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = names or {k: k for k in TASKS}
    values = {
        entry: {n: ranking_metric(TASKS[n], res["tasks"].get(n, {})) for n in names}
        for entry, res in record["results"].items()
    }
    shown = [n for n in names if any(v[n] is not None for v in values.values())]
    if not shown:
        return
    entries = list(record["results"])
    width = 0.8 / len(entries)
    fig, ax = plt.subplots(figsize=(1.1 * len(shown) + 3, 4))
    x = np.arange(len(shown))
    for i, entry in enumerate(entries):
        vals = [values[entry][n] or 0.0 for n in shown]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=entry)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([names[n] for n in shown], rotation=30, ha="right")
    ax.set_ylabel("R$^2$ / macro-F1")
    # Not ylim(0, 1): R^2 is unclipped, so a probe worse than the target's own
    # mean goes below the axis and should be visible doing it.
    ax.set_ylim(
        min(-0.05, *(v for e in entries for v in (values[e][n] or 0.0 for n in shown)))
        - 0.02,
        1.0,
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "task_scores.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out / 'task_scores.png'}")


if __name__ == "__main__":
    main()
