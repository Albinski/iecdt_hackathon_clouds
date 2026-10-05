"""The three schedules I-JEPA runs on, and the AdamW parameter groups.

All three are defined over the *total* step count, which is why a run killed at
walltime underperforms badly rather than slightly: the learning rate never
anneals, the weight decay never reaches its final value, and the EMA momentum
never reaches 1.0, so the target encoder is still moving when training stops.
`train_ijepa.py` has a `max_seconds` guard for exactly this reason.

Two of the three run in the direction people expect them not to: the weight
decay **rises** 0.04 -> 0.4, and the EMA momentum **rises** 0.996 -> 1.0.
"""

import math


def warmup_cosine(step, total_steps, warmup_steps, start_lr, ref_lr, final_lr):
    """Linear warmup `start_lr` -> `ref_lr`, then cosine down to `final_lr`.

    `ref_lr` is the peak. Note that the reference implementation's 1e-3 is the
    peak at a *global* batch of 2048 (128 per GPU across 16 GPUs); at a single
    GPU's batch of 128, square-root scaling puts the equivalent peak near
    2.5e-4. Copying 1e-3 across is a reliable way to collapse the run.
    """
    if step < warmup_steps:
        return start_lr + (ref_lr - start_lr) * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return final_lr + (ref_lr - final_lr) * cosine


def linear_schedule(step, total_steps, start, end):
    """`start` -> `end`, linearly, clamped at `end`. Weight decay and EMA."""
    return start + (end - start) * min(1.0, step / max(1, total_steps))


def param_groups(*modules):
    """Two AdamW groups: decay the weight matrices, leave everything else.

    Mirrors `init_opt` in the reference: biases and every 1-D tensor are
    excluded. That covers the LayerNorm scales and shifts and the predictor's
    `mask_token`, all of which a decay ramp climbing to 0.4 would actively
    damage -- `mask_token` in particular is the predictor's only handle on
    "something belongs here", and pulling it toward zero removes it.

    The `wd_scheduled` flag is read back by the training loop, which rewrites
    `weight_decay` every step. Extra keys in a param group are preserved by
    `torch.optim`, so this travels with the group rather than in a side table.
    """
    decay, no_decay = [], []
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if "bias" in name or param.ndim == 1 else decay).append(param)
    return [
        {"params": decay, "wd_scheduled": True},
        {"params": no_decay, "weight_decay": 0.0, "wd_scheduled": False},
    ]
