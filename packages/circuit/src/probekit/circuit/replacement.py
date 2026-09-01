"""Replacement model: GPT-2 with all covered MLPs swapped for CLT outputs.

Semantics follow Anthropic's circuit-tracing setup: when every MLP in a range
is replaced by its transcoder, feature-to-feature interactions become linear
for a given input, enabling gradient-based attribution graphs.

Implementation note (documented approximation): transcoder inputs x_l are
cached from a clean pass and frozen during the replacement pass ("frozen-input
scheme"). This keeps the computation differentiable end-to-end w.r.t. feature
activations while remaining faithful for short prompts at GPT-2 scale.
"""

from __future__ import annotations

import torch
from probekit.circuit.clt import CLTStack


class ReplacementRun:
    """Result of one replacement-model pass."""

    def __init__(
        self,
        logits: torch.Tensor,
        feature_acts: dict[int, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        self.logits = logits
        self.feature_acts = feature_acts  # {layer: (B, P, n_features)}
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    @property
    def last_position(self) -> int:
        return int(self.attention_mask.sum(dim=1).max().item()) - 1


class ReplacementModel:
    def __init__(self, model, tokenizer, stack: CLTStack):
        self.model = model
        self.tokenizer = tokenizer
        self.stack = stack.to(next(model.parameters()).device)
        self.device = next(model.parameters()).device
        self._blocks = self._get_blocks()

    def _get_blocks(self):
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        raise ValueError("Unsupported architecture")

    # ------------------------------------------------------------------ passes

    @torch.no_grad()
    def _clean_pass(self, enc):
        """Pass 1: cache frozen transcoder inputs (MLP inputs) per layer."""
        cached: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook(_m, args, kwargs):
                x_in = args[0] if args else kwargs.get("hidden_states")
                cached[layer_idx] = x_in.detach().float()

            return hook

        for li in self.stack.layer_ids:
            handles.append(
                self._blocks[li].mlp.register_forward_pre_hook(make_hook(li), with_kwargs=True)
            )
        try:
            self.model(**enc)
        finally:
            for h in handles:
                h.remove()
        return cached

    def run(
        self,
        prompt: str | None = None,
        *,
        enc: dict | None = None,
        track_grad: bool = True,
    ) -> ReplacementRun:
        """Run the replacement model. Returns logits + feature activations where
        activations are graph leaves (requires_grad) so attribution can flow."""
        if enc is None:
            assert prompt is not None
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        cached_inputs = self._clean_pass(enc)
        with torch.no_grad():
            acts_frozen = {
                li: self.stack.get(li).encode(cached_inputs[li])
                for li in self.stack.layer_ids
            }

        # Leaf activations: gradients of the target logit land here.
        acts_leaf = {
            li: a.detach().clone().requires_grad_(track_grad)
            for li, a in acts_frozen.items()
        }
        decoded_cum: dict[int, torch.Tensor] = {}
        running: torch.Tensor | None = None
        for li in sorted(acts_leaf):
            out_l = self.stack.get(li).decode(acts_leaf[li]).to(self.model_dtype())
            running = out_l if running is None else running + out_l
            decoded_cum[li] = running.clone()

        handles = []
        replaced_layers = set(decoded_cum)

        def make_hook(layer_idx: int):
            def hook(_m, _args, _kwargs, output):
                y = output[0] if isinstance(output, tuple) else output
                if layer_idx not in replaced_layers:
                    return y
                # MLP_j output <- sum of ALL upstream transcoder writes (li <= j).
                new = decoded_cum[layer_idx].to(y.dtype).clone()
                if isinstance(output, tuple):
                    return (new,) + tuple(output[1:])
                return new

            return hook

        for j in range(max(replaced_layers) + 1):
            handles.append(
                self._blocks[j].mlp.register_forward_hook(make_hook(j), with_kwargs=True)
            )
        try:
            with torch.enable_grad():
                out = self.model(**enc)
        finally:
            for h in handles:
                h.remove()

        return ReplacementRun(out.logits, acts_leaf, enc["input_ids"], enc["attention_mask"])

    def model_dtype(self):
        return next(self.model.parameters()).dtype
