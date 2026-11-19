"""TopK autoencoder with a recorded scalar input normalization."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

FIRING_THRESHOLD = 1e-8


class TopKSAE(nn.Module):
    """Keep k rectified encoder values and reconstruct in the original units."""

    def __init__(self, d_model: int, dict_size: int, k: int, scale: float = 1.0) -> None:
        super().__init__()
        if not 0 < k <= dict_size or d_model <= 0 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid SAE dimensions or normalization scale")
        self.d_model = d_model
        self.dict_size = dict_size
        self.k = k
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.encoder = nn.Linear(d_model, dict_size)
        self.decoder = nn.Linear(dict_size, d_model)
        with torch.no_grad():
            self.renorm_decoder()
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()
            self.decoder.bias.zero_()

    @torch.no_grad()
    def renorm_decoder(self) -> None:
        self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))

    def preactivations(self, x: Tensor) -> Tensor:
        return self.encoder(x * self.scale - self.decoder.bias)

    def select(self, pre: Tensor) -> Tensor:
        values, indices = pre.topk(self.k, dim=-1, sorted=False)
        # This keeps exactly k nonzeros even on degenerate inputs.
        values = values.clamp_min(torch.finfo(values.dtype).tiny)
        return torch.zeros_like(pre).scatter(-1, indices, values)

    def encode(self, x: Tensor) -> Tensor:
        return self.select(self.preactivations(x))

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents) / self.scale

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(x)
        return self.decode(latents), latents
