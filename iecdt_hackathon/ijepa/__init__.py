"""I-JEPA pretraining: latent-space masked prediction instead of reconstruction.

`iecdt_hackathon.train_ijepa` drives it; `IJepa` is registered in
`iecdt_hackathon.models.build_model` as `"ijepa"`, which is what lets the
existing `embed.py` and `evaluate.py` consume its checkpoints unchanged.
"""

from .masking import MultiBlockMaskCollator
from .model import POOLS, IJepa
from .transforms import d4

__all__ = ["IJepa", "MultiBlockMaskCollator", "POOLS", "d4"]
