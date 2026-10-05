import torch
from torch import nn


def _block(in_ch, out_ch, groups=8):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
        nn.GroupNorm(min(groups, out_ch), out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(min(groups, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


def _up_block(in_ch, out_ch, groups=8):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(min(groups, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    """Baseline encoder/decoder. Any submission must expose `encode`.

    Args:
        in_channels: channels the dataset yields (6 bands, plus any extras).
        embedding_dim: size of the vector the probes are fit on.
        width: base channel count; blocks use width, 2w, 4w, 8w.
        seed_size: spatial size the decoder starts from.
    """

    def __init__(self, in_channels=6, embedding_dim=256, width=32, seed_size=8):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.seed_size = seed_size
        w = width
        chans = [w, 2 * w, 4 * w, 8 * w]

        self.encoder = nn.Sequential(
            _block(in_channels, chans[0]),
            _block(chans[0], chans[1]),
            _block(chans[1], chans[2]),
            _block(chans[2], chans[3]),
        )
        self.to_embedding = nn.Linear(chans[3], embedding_dim)

        self.from_embedding = nn.Linear(embedding_dim, chans[3] * seed_size**2)
        self.decoder = nn.Sequential(
            _up_block(chans[3], chans[2]),
            _up_block(chans[2], chans[1]),
            _up_block(chans[1], chans[0]),
            _up_block(chans[0], chans[0]),
            _up_block(chans[0], chans[0]),
            nn.Conv2d(chans[0], in_channels, kernel_size=3, padding=1),
        )

    def encode(self, x):
        """(B, C, H, W) -> (B, embedding_dim). The evaluation contract."""
        h = self.encoder(x)
        h = h.mean(dim=(2, 3))  # global average pool: size-agnostic
        return self.to_embedding(h)

    def decode(self, z, out_size):
        h = self.from_embedding(z)
        h = h.view(z.shape[0], -1, self.seed_size, self.seed_size)
        h = self.decoder(h)
        if h.shape[-2:] != tuple(out_size):
            h = nn.functional.interpolate(
                h, size=tuple(out_size), mode="bilinear", align_corners=False
            )
        return h

    def forward(self, x):
        return self.decode(self.encode(x), x.shape[-2:])


def build_model(name="conv_autoencoder", **kwargs):
    """Factory used by train.py and evaluate.py to rebuild from a checkpoint."""
    from .ijepa.model import IJepa  # local: keeps models.py cheap to import

    models = {"conv_autoencoder": ConvAutoencoder, "ijepa": IJepa}
    if name not in models:
        raise ValueError(f"Unknown model '{name}'; choose from {sorted(models)}")
    return models[name](**kwargs)
