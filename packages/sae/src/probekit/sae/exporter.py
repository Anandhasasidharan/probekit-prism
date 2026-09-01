"""Export SAE encoder to ONNX (hand-built graph — no training deps at runtime).

The exported graph computes: acts = TopK/ReLU((x - b_dec) @ W_enc + b_enc),
i.e. the latent projection used by SAE probes and the feature-serving API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from probekit.sae.architectures import BaseSAE, JumpReLUSAE, TopKSAE


def _topk_mask(pre: torch.Tensor, k: int) -> torch.Tensor:
    values, indices = torch.topk(pre, k, dim=-1)
    mask = torch.zeros_like(pre, dtype=torch.float32)
    return mask.scatter(-1, indices, values.new_ones(values.shape))


def export_sae_onnx(
    sae: BaseSAE,
    path: str | Path,
    opset: int = 17,
) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Freeze parameters so they export as constants.
    was_training = sae.training
    requires_grad_state = {n: p.requires_grad for n, p in sae.named_parameters()}
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    class EncoderModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pre = (x - sae.b_dec) @ sae.W_enc + sae.b_enc
            if isinstance(sae, JumpReLUSAE):
                thr = torch.exp(sae.log_threshold)
                return torch.relu(pre) * (pre > thr).float()
            if isinstance(sae, TopKSAE):
                k = min(sae.k, pre.shape[-1])
                values, indices = torch.topk(pre, k, dim=-1)
                acts = torch.zeros_like(pre)
                return acts.scatter(-1, indices, torch.relu(values))
            return torch.relu(pre)

    module = EncoderModule().eval()
    orig_device = next(sae.parameters()).device
    # ONNX export traces on CPU — temporarily move SAE there
    sae.to(torch.device("cpu"))
    dummy = torch.randn(1, sae.d_model, device=torch.device("cpu"))

    try:
        torch.onnx.export(
            module,
            (dummy,),
            str(path),
            input_names=["activations"],
            output_names=["latents"],
            dynamic_axes={"activations": {0: "batch"}, "latents": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    finally:
        # Restore training state, grad flags, and original device
        for n, p in sae.named_parameters():
            p.requires_grad_(requires_grad_state.get(n, True))
        sae.train(was_training)
        sae.to(orig_device)
    return str(path)


def verify_sae_onnx(path: str | Path, sae: BaseSAE, sample: np.ndarray) -> float:
    """Round-trip check with onnxruntime; returns max abs difference."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    # Reference computation runs on whatever device the SAE lives on.
    ref_device = next(sae.parameters()).device
    with torch.no_grad():
        expected = (
            sae.encode(torch.from_numpy(sample).float().to(ref_device))
            .cpu()
            .numpy()
        )
    got = sess.run(["latents"], {"activations": sample.astype(np.float32)})[0]
    return float(np.abs(got - expected).max())
