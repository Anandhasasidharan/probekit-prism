"""Cross-layer transcoders (CLTs), following Anthropic's circuit-tracing method
(Ameisen et al. 2025, "Circuit Tracing: Revealing Computational Graphs in
Language Models") adapted to GPT-2-Small scale.

A CLT sits in place of one MLP: it READS the residual stream at its own layer
(the MLP input) and conceptually WRITES to every subsequent layer. When all
MLPs in a range are replaced by CLTs, feature-to-feature interactions become
(approximately) linear for a given input, which is what makes attribution
graphs tractable.

Scale notes for RTX 3050 (4-6 GB VRAM): GPT-2 Small has d_model=768. A CLT
with n_features=4096 per layer costs ~2 x (768*4096) params ~= 6M floats
(~25 MB fp32) per layer — 12 layers fit easily.
"""

from __future__ import annotations

import torch
from probekit.core.device import get_device
from torch import nn


class CrossLayerTranscoder(nn.Module):
    """Transcoder for a single layer: relu((x - b_in) @ W_enc + b_enc) @ W_dec."""

    def __init__(self, d_model: int, n_features: int, device: str | None = None):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        dev = get_device(device)
        self.b_in = nn.Parameter(torch.zeros(d_model, device=dev))
        self.W_enc = nn.Parameter(torch.empty(d_model, n_features, device=dev))
        self.b_enc = nn.Parameter(torch.zeros(n_features, device=dev))
        self.W_dec = nn.Parameter(torch.empty(n_features, d_model, device=dev))
        self.b_out = nn.Parameter(torch.zeros(d_model, device=dev))

        with torch.no_grad():
            # Init decoder columns unit-norm; encoder tied to decoder transpose.
            w = torch.randn(n_features, d_model, device=dev)
            w /= w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.W_dec.copy_(w * 0.05)
            self.W_enc.copy_(w.T * (1.0 / max(d_model, 1) ** 0.5))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) MLP input -> (..., n_features) feature activations."""
        pre = (x - self.b_in) @ self.W_enc + self.b_enc
        return torch.relu(pre)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        return self.decode(acts), acts

    @torch.no_grad()
    def renorm_decoder(self) -> None:
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)


class CLTStack(nn.Module):
    """All transcoders for a contiguous range of model layers [start_layer, end_layer)."""

    def __init__(
        self,
        layer_ids: list[int],
        d_model: int,
        n_features: int,
        device: str | None = None,
    ):
        super().__init__()
        self.layer_ids = list(layer_ids)
        self.d_model = d_model
        self.n_features = n_features
        self.transcoders = nn.ModuleDict(
            {str(li): CrossLayerTranscoder(d_model, n_features, device=device) for li in layer_ids}
        )

    def __contains__(self, layer: int) -> bool:
        return str(layer) in self.transcoders

    def get(self, layer: int) -> CrossLayerTranscoder:
        tc: CrossLayerTranscoder = self.transcoders[str(layer)]  # type: ignore[assignment]
        return tc

    def encode_all(self, inputs: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Feature activations for every covered layer given MLP inputs."""
        return {
            li: self.get(li).encode(inputs[li]) for li in self.layer_ids if li in inputs
        }

    def save(self, path: str) -> None:
        torch.save(
            {"state_dict": self.state_dict(), "meta": {
                "layer_ids": self.layer_ids,
                "d_model": self.d_model,
                "n_features": self.n_features,
            }},
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | None = None) -> CLTStack:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        stack = cls(blob["meta"]["layer_ids"], blob["meta"]["d_model"],
                    blob["meta"]["n_features"], device=device)
        stack.load_state_dict(blob["state_dict"])
        stack.to(get_device(device)).eval()
        return stack
