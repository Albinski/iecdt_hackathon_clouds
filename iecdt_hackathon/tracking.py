"""Weights & Biases logging, with the things that break it here handled.

A bare `wandb.init()` loses data or kills the job in three ways on this setup:

1. **`wandb` is an optional extra** (`pyproject.toml`), so it may simply not be
   installed. A logging import must never take down a 12 h training run, which
   is what `import wandb` at the top of `train.py` would do.
2. **Orchid compute nodes have no outbound internet.** An online run blocks on
   the handshake and then fails or silently drops metrics. Runs are written to
   disk with `WANDB_MODE=offline` instead and pushed afterwards from a login
   node with `wandb sync`. `mode="auto"` picks offline whenever `SLURM_JOB_ID`
   is set, which is exactly the case where the network is missing.
3. **Unnamed runs are unusable.** The default is a random name like
   `fiery-sponge-7`, which cannot be matched against a results table keyed by
   run directory. Runs are named after their `--out` directory and tagged with
   the objective and architecture.

Nothing here raises: every failure degrades to "no logging" with a printed
reason, because a missing graph is cheaper than a lost run.
"""

import os
from pathlib import Path

#: Written next to the run so `wandb sync` can be reconstructed later.
SYNC_HINT = "wandb_sync.txt"


def resolve_mode(mode):
    """`auto` -> offline inside a SLURM job, online outside it.

    An explicit `WANDB_MODE` in the environment wins over the config, so a
    one-off `WANDB_MODE=disabled` on the command line works as expected.
    """
    env = os.environ.get("WANDB_MODE")
    if env:
        return env
    if mode in (None, "auto"):
        return "offline" if os.environ.get("SLURM_JOB_ID") else "online"
    return mode


def init(cfg, run_dir, enabled=True, name=None, tags=()):
    """Start a run, or return None. Never raises.

    `run_dir` holds the offline run data, so a synced run and its checkpoints
    stay together.
    """
    if not enabled:
        return None

    training = cfg.get("training", {})
    mode = resolve_mode(training.get("wandb_mode"))
    if mode == "disabled":
        print("wandb: disabled", flush=True)
        return None

    try:
        import wandb
    except ImportError:
        print(
            "wandb: not installed, continuing without it "
            "(uv sync --extra wandb on the login node)",
            flush=True,
        )
        return None

    run_dir = Path(run_dir)
    model = cfg.get("model", {})
    tags = list(tags) or [t for t in (model.get("name"), model.get("arch")) if t]

    try:
        run = wandb.init(
            project=training.get("wandb_project", "iecdt-hackathon"),
            name=name or run_dir.name,
            tags=tags,
            config=cfg,
            dir=str(run_dir),
            mode=mode,
        )
    except Exception as exc:
        # Most often an unauthenticated online run: `wandb login` has not been
        # run and no WANDB_API_KEY is set.
        print(f"wandb: init failed ({exc}); continuing without it", flush=True)
        return None

    print(f"wandb: {mode} run '{run.name}' -> {run.url or run_dir}", flush=True)
    if mode == "offline":
        _write_sync_hint(run_dir)
    return run


def _write_sync_hint(run_dir):
    """Leave the exact `wandb sync` command beside the run.

    An offline run is worthless until it is pushed, and the glob is easy to get
    wrong weeks later when the job has long since finished.
    """
    hint = (
        f"# Push this offline run from a login node (not a compute node):\n"
        f"cd {Path.cwd()}\n"
        f"uv run wandb sync {run_dir}/wandb/offline-run-*\n"
    )
    try:
        (run_dir / SYNC_HINT).write_text(hint)
    except OSError:
        pass


def finish(run):
    """Close a run, and say how to push it if it was offline."""
    if run is None:
        return
    offline = getattr(run, "offline", False)
    run_dir = Path(run.dir).parent.parent if getattr(run, "dir", None) else None
    try:
        run.finish()
    except Exception as exc:
        print(f"wandb: finish failed ({exc})", flush=True)
        return
    if offline and run_dir:
        print(
            f"\nwandb ran offline. Push it from a login node:\n"
            f"  uv run wandb sync {run_dir}/offline-run-*",
            flush=True,
        )
