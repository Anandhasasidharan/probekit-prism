"""Sparse autoencoder architectures: TopK, Gated, JumpReLU, Standard (ReLU).

All architectures share a linear encoder/decoder pair and expose:
  - forward(x) -> (reconstruction, feature_activations)
  - encode(x) -> feature_activations
  - decode(acts) -> reconstruction
  - decoder_directions() -> (dict_size, d_model) tensor for steering
"""

from __future__ import annotations

import torch
from probekit.core.device import get_device
from torch import nn


class BaseSAE(nn.Module):
    W_enc: nn.Parameter
    W_dec: nn.Parameter
    b_dec: nn.Parameter
    b_enc: nn.Parameter

    def __init__(self, d_model: int, dict_size: int):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def _init_weights(self) -> None:
        # Common init: decoder columns unit-norm, encoder = decoder^T
        with torch.no_grad():
            w = torch.randn(self.dict_size, self.d_model, device=self.W_enc.device)
            w = w / w.norm(dim=-1, keepdim=True)
            self.W_dec.data.copy_(w)
            self.W_enc.data.copy_(w.T)

    @torch.no_grad()
    def renorm_decoder(self) -> None:
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        return self.decode(acts), acts

    def decoder_directions(self) -> torch.Tensor:
        return self.W_dec.detach()


class StandardSAE(BaseSAE):
    """ReLU SAE with L1 sparsity (handled by trainer via sparsity_coef)."""

    def __init__(self, d_model: int, dict_size: int, sparsity_coef: float = 0.1,
                 device: str | None = None):
        super().__init__(d_model, dict_size)
        self.sparsity_coef = sparsity_coef
        dev = get_device(device)
        self.W_enc = nn.Parameter(torch.empty(d_model, dict_size, device=dev))
        self.b_enc = nn.Parameter(torch.zeros(dict_size, device=dev))
        self.W_dec = nn.Parameter(torch.empty(dict_size, d_model, device=dev))
        self._init_weights()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return torch.relu(pre)


class TopKSAE(BaseSAE):
    """TopK SAE (Gao et al. 2024): keep only the k largest activations.

    No L1 needed — sparsity is enforced architecturally, which avoids
    shrinkage of active features and the L0/L1 tradeoff.
    """

    def __init__(self, d_model: int, dict_size: int, k: int = 32,
                 device: str | None = None):
        super().__init__(d_model, dict_size)
        self.k = k
        dev = get_device(device)
        self.W_enc = nn.Parameter(torch.empty(d_model, dict_size, device=dev))
        self.b_enc = nn.Parameter(torch.zeros(dict_size, device=dev))
        self.W_dec = nn.Parameter(torch.empty(dict_size, d_model, device=dev))
        self._init_weights()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        k = min(self.k, pre.shape[-1])
        values, indices = torch.topk(pre, k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, indices, torch.relu(values))
        return acts


class GatedSAE(BaseSAE):
    """Gated SAE (Rajamanoharan et al. 2024): separate gate and magnitude paths."""

    def __init__(self, d_model: int, dict_size: int, sparsity_coef: float = 0.1,
                 device: str | None = None):
        super().__init__(d_model, dict_size)
        self.sparsity_coef = sparsity_coef
        dev = get_device(device)
        self.W_gate = nn.Parameter(torch.empty(d_model, dict_size, device=dev))
        self.b_gate = nn.Parameter(torch.zeros(dict_size, device=dev))
        self.W_mag = nn.Parameter(torch.empty(d_model, dict_size, device=dev))
        self.b_mag = nn.Parameter(torch.zeros(dict_size, device=dev))
        self.W_dec = nn.Parameter(torch.empty(dict_size, d_model, device=dev))

        with torch.no_grad():
            w = torch.randn(dict_size, d_model, device=dev)
            w = w / w.norm(dim=-1, keepdim=True)
            self.W_dec.copy_(w)
            self.W_gate.copy_(w.T)
            self.W_mag.copy_(w.T)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        gate_pre = (x - self.b_dec) @ self.W_gate + self.b_gate
        mag_pre = (x - self.b_dec) @ self.W_mag + self.b_mag
        gate = (gate_pre > 0).float()
        return gate * torch.relu(mag_pre)


class JumpReLUThreshold(torch.autograd.Function):
    """JumpReLU with straight-through gradient estimation on the threshold."""

    @staticmethod
    def forward(ctx, pre: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(pre, threshold)
        return (pre > threshold).float() * pre

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        pre, threshold = ctx.saved_tensors
        ste = (pre > threshold).float()
        grad_pre = grad_out * ste
        grad_thr = grad_out * pre * (-ste)  # simplified STE
        return grad_pre, grad_thr.sum() if threshold.dim() == 0 else grad_thr


class JumpReLUSAE(BaseSAE):
    """JumpReLU SAE (Decu et al. 2024): learned per-feature firing threshold."""

    def __init__(self, d_model: int, dict_size: int,
                 initial_threshold: float = 1e-3, device: str | None = None):
        super().__init__(d_model, dict_size)
        dev = get_device(device)
        self.log_threshold = nn.Parameter(
            torch.full((dict_size,), float(torch.log(torch.tensor(initial_threshold))),
                       device=dev)
        )
        self.W_enc = nn.Parameter(torch.empty(d_model, dict_size, device=dev))
        self.b_enc = nn.Parameter(torch.zeros(dict_size, device=dev))
        self.W_dec = nn.Parameter(torch.empty(dict_size, d_model, device=dev))
        self._init_weights()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        thr = torch.exp(self.log_threshold)
        return JumpReLUThreshold.apply(pre, thr)


def build_sae(
    architecture: str,
    d_model: int,
    dict_size: int,
    k: int = 32,
    sparsity_coef: float = 0.1,
    device: str | None = None,
) -> BaseSAE:
    arch = architecture.lower()
    if arch == "topk":
        sae: BaseSAE = TopKSAE(d_model, dict_size, k=k, device=device)
    elif arch == "gated":
        sae = GatedSAE(d_model, dict_size, sparsity_coef=sparsity_coef, device=device)
    elif arch == "jumprelu":
        sae = JumpReLUSAE(d_model, dict_size, device=device)
    elif arch in ("standard", "relu"):
        sae = StandardSAE(d_model, dict_size, sparsity_coef=sparsity_coef, device=device)
    else:
        raise ValueError(f"Unknown SAE architecture: {architecture}")
    return sae.to(get_device(device))
