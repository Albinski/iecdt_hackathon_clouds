"""Tests for the Weights & Biases wiring.

The contract is that logging can never affect training. Every one of these
failure modes is reachable in normal use -- the package is an optional extra,
compute nodes have no network, and `wandb login` is a manual step -- so each
has to degrade to "no logging" rather than raise.
"""

import os

import pytest

from iecdt_hackathon import tracking

CFG = {
    "model": {"name": "ijepa", "arch": "vit_small"},
    "training": {"wandb_project": "iecdt-hackathon", "wandb_mode": "auto"},
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)


def test_auto_is_online_outside_slurm():
    assert tracking.resolve_mode("auto") == "online"


def test_auto_is_offline_inside_slurm(monkeypatch):
    """Orchid nodes have no outbound network, so a SLURM job must go to disk."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert tracking.resolve_mode("auto") == "offline"
    assert tracking.resolve_mode(None) == "offline"


def test_explicit_mode_beats_the_slurm_heuristic(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert tracking.resolve_mode("online") == "online"


def test_env_beats_the_config(monkeypatch):
    """So a one-off `WANDB_MODE=disabled` on the command line works."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert tracking.resolve_mode("auto") == "disabled"


def test_disabled_flag_short_circuits_before_importing_wandb(monkeypatch):
    """`--wandb` absent must not even import the optional dependency."""
    monkeypatch.setitem(os.environ, "WANDB_MODE", "online")
    assert tracking.init(CFG, "/nonexistent", enabled=False) is None


def test_disabled_mode_returns_none(tmp_path):
    cfg = {**CFG, "training": {**CFG["training"], "wandb_mode": "disabled"}}
    assert tracking.init(cfg, tmp_path, enabled=True) is None


def test_init_survives_a_missing_wandb(monkeypatch, tmp_path, capsys):
    """The package is an optional extra; its absence must not stop a run."""
    import builtins

    real_import = builtins.__import__

    def no_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("No module named 'wandb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_wandb)
    assert tracking.init(CFG, tmp_path, enabled=True) is None
    assert "not installed" in capsys.readouterr().out


def test_init_survives_a_failing_wandb(monkeypatch, tmp_path, capsys):
    """The usual cause is an online run with no API key configured."""
    import wandb

    def boom(**kwargs):
        raise ValueError("No API key configured")

    monkeypatch.setattr(wandb, "init", boom)
    assert tracking.init(CFG, tmp_path, enabled=True) is None
    assert "init failed" in capsys.readouterr().out


def test_finish_accepts_none():
    tracking.finish(None)      # must be a no-op, not an AttributeError


def test_offline_run_writes_a_sync_hint(tmp_path):
    """An offline run is worthless until pushed, so leave the command behind."""
    cfg = {**CFG, "training": {**CFG["training"], "wandb_mode": "offline"}}
    run = tracking.init(cfg, tmp_path, enabled=True)
    assert run is not None
    assert run.name == tmp_path.name
    assert set(run.tags) == {"ijepa", "vit_small"}
    tracking.finish(run)

    hint = (tmp_path / tracking.SYNC_HINT).read_text()
    assert "wandb sync" in hint
    assert sorted((tmp_path / "wandb").glob("offline-run-*"))
