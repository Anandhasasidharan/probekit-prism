"""Collect (MLP input, MLP output) training pairs from any HF causal LM.

These pairs supervise cross-layer-transcoder training: each transcoder learns
to reproduce its layer's MLP output from the MLP input, so the CLT can later
substitute for the MLP while exposing interpretable feature activations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from transformers import PreTrainedTokenizerBase


def _decoder_blocks(model: Any) -> list[Any]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks: list[Any] = list(model.transformer.h)
        return blocks
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        llama_blocks: list[Any] = list(model.model.layers)
        return llama_blocks
    raise ValueError("Unsupported architecture")


def collect_mlp_pairs(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    layer_start: int = 0,
    layer_end: int | None = None,
    batch_size: int = 16,
    device: str | None = None,
    max_length: int = 256,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return {layer_index: (mlp_inputs[n, d], mlp_outputs[n, d])} over all tokens.

    Pad tokens are excluded. Matrices are float32 CPU numpy arrays ready for
    CLT training.
    """
    import torch as th

    target_device = th.device(device) if device else next(model.parameters()).device
    blocks = _decoder_blocks(model)
    end = layer_end if layer_end is not None else len(blocks)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    captured: dict[int, list[tuple[Any, Any]]] = {
        li: [] for li in range(layer_start, end)
    }
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, args, kwargs, output):
            x_in = args[0] if args else kwargs.get("hidden_states")
            y_out = output[0] if isinstance(output, tuple) else output
            captured[layer_idx].append(
                (x_in.detach().float().cpu(), y_out.detach().float().cpu())
            )
        return hook

    for li in range(layer_start, end):
        handles.append(blocks[li].mlp.register_forward_hook(make_hook(li), with_kwargs=True))

    try:
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(target_device)
            with th.no_grad():
                model(**enc)
            mask = enc["attention_mask"].bool()
            for li in captured:
                xs, ys = captured[li][-1]
                # Keep only non-pad token positions across the whole batch dim.
                keep = mask.reshape(-1).cpu()
                kept = (
                    xs.reshape(-1, xs.shape[-1])[keep],
                    ys.reshape(-1, ys.shape[-1])[keep],
                )
                captured[li][-1] = (kept[0], kept[1])
    finally:
        for h in handles:
            h.remove()

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for li, chunks in captured.items():
        if not chunks:
            continue
        xs = th.cat([c[0] for c in chunks], dim=0).numpy().astype(np.float32)
        ys = th.cat([c[1] for c in chunks], dim=0).numpy().astype(np.float32)
        out[li] = (xs, ys)
    return out
