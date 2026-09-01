"""SAE quality metrics: L0 sparsity, reconstruction MSE, dead feature rate,
dictionary utilization, and per-feature firing frequencies."""

from __future__ import annotations

import numpy as np
import torch
from probekit.core.device import get_device
from probekit.sae.architectures import BaseSAE


@torch.no_grad()
def evaluate_sae(
    sae: BaseSAE,
    data: np.ndarray,
    batch_size: int = 512,
    device: str | None = None,
) -> dict:
    # Device follows the SAE weights unless explicitly overridden.
    dev = next(sae.parameters()).device if device is None else get_device(device)
    sae.eval()
    matrix = torch.from_numpy(np.asarray(data, dtype=np.float32)).to(dev)

    total_sq_err = 0.0
    total_norm_sq = 0.0
    l0_total = 0
    max_act_per_feat = torch.zeros(sae.dict_size, device=dev)
    freq_counts = torch.zeros(sae.dict_size, device=dev)
    n_tokens = 0

    for start in range(0, matrix.shape[0], batch_size):
        batch = matrix[start : start + batch_size]
        recon, acts = sae(batch)
        diff = recon - batch
        total_sq_err += float(diff.pow(2).sum().item())
        total_norm_sq += float(batch.pow(2).sum().item())
        l0_total += int((acts > 0).sum(dim=-1).sum().item())
        max_act_per_feat = torch.maximum(max_act_per_feat, acts.amax(dim=0))
        freq_counts += (acts > 1e-6).sum(dim=0)
        n_tokens += batch.shape[0]

    mse = total_sq_err / (n_tokens * sae.d_model)
    explained_var = 1.0 - (total_sq_err / max(total_norm_sq, 1e-12))
    active = (max_act_per_feat > 1e-6)
    n_active = int(active.sum().item())
    freqs = (freq_counts / max(n_tokens, 1)).cpu().numpy()

    return {
        "l0": l0_total / n_tokens,
        "reconstruction_mse": mse,
        "explained_variance": explained_var,
        "dead_feature_rate": 1.0 - n_active / sae.dict_size,
        "dictionary_utilization": n_active / sae.dict_size,
        "n_tokens": n_tokens,
        "feature_frequencies": freqs.tolist(),
    }


@torch.no_grad()
def feature_frequencies(
    sae: BaseSAE, data: np.ndarray, device: str | None = None
) -> np.ndarray:
    """Per-feature activation frequency vector (used by drift monitoring)."""
    metrics = evaluate_sae(sae, data, device=device)
    return np.asarray(metrics["feature_frequencies"], dtype=np.float32)


@torch.no_grad()
def encode_matrix(
    sae: BaseSAE,
    data: np.ndarray,
    batch_size: int = 512,
    device: str | None = None,
) -> np.ndarray:
    """Project raw activations into SAE latent space -> (n_tokens, dict_size)."""
    from probekit.core.device import empty_cache

    dev = next(sae.parameters()).device if device is None else get_device(device)
    sae.eval()
    out = []
    for start in range(0, len(data), batch_size):
        batch = torch.from_numpy(
            np.asarray(data[start : start + batch_size], dtype=np.float32)
        ).to(dev)
        acts = sae.encode(batch)
        out.append(acts.cpu().numpy())
    empty_cache(dev)
    return np.concatenate(out, axis=0)
