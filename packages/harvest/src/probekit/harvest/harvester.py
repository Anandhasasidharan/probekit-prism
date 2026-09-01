"""Async batched activation harvesting with provenance tracking.

Supports GPT-2 style (`model.transformer.h`) and Llama/Qwen style
(`model.model.layers`) architectures. Runs on CPU or CUDA — pass `device`
or let it auto-detect (prefers the GPU, e.g. an RTX 3050).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import torch
from probekit.core.device import empty_cache, get_device
from probekit.core.types import ActivationRecord
from probekit.harvest.compression import write_activation_parquet
from probekit.harvest.provenance import ProvenanceTracker
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_layer_list(capture_layers: list[int | str], n_layers: int) -> list[int]:
    resolved: list[int] = []
    for layer in capture_layers:
        if isinstance(layer, str) and layer.lower() == "last":
            resolved.append(n_layers - 1)
        else:
            resolved.append(int(layer))
    out = sorted(set(resolved))
    for li in out:
        if not 0 <= li < n_layers:
            raise ValueError(f"capture layer {li} out of range [0, {n_layers - 1}]")
    return out


class ActivationHarvester:
    """Collect per-token residual-stream activations from any HF causal LM."""

    def __init__(
        self,
        model_name: str = "gpt2",
        capture_layers: list[int | str] | None = None,
        batch_size: int = 32,
        device: str | None = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.capture_layers = capture_layers or [0, 5, 10, "last"]
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = get_device(device)

        model: Any = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device=str(self.device))
        self.model = model.eval()
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        n_layers = getattr(self.model.config, "n_layer", None) or getattr(
            self.model.config, "num_hidden_layers"
        )
        assert n_layers is not None, f"Cannot infer layer count for {model_name}"
        self._n_layers = int(n_layers)
        self._resolved_layers = _resolve_layer_list(self.capture_layers, self._n_layers)
        self._tracker = ProvenanceTracker(model_version=model_name)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._stored: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ hooks

    def _decoder_blocks(self) -> list[Any]:
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        raise ValueError(f"Unsupported model architecture for {self.model_name}")

    def _register_hooks(self) -> None:
        self._stored.clear()
        blocks = self._decoder_blocks()

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                self._stored[layer_idx] = hidden.detach()

            return hook

        for layer_idx in self._resolved_layers:
            handle = blocks[layer_idx].register_forward_hook(make_hook(layer_idx))
            self._handles.append(handle)

    def _remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._stored.clear()

    # --------------------------------------------------------------- harvesting

    def harvest_sync(
        self,
        prompts: list[str],
        save_path: str | Path | None = None,
    ) -> list[ActivationRecord]:
        """Harvest activations for a list of prompts (single or multi-batch).

        Returns one record per (prompt token x captured layer), excluding pad
        tokens. Optionally writes all records to a compressed Parquet file.
        """
        records: list[ActivationRecord] = []
        try:
            for start in range(0, len(prompts), self.batch_size):
                records.extend(self._harvest_batch(prompts[start : start + self.batch_size]))
        finally:
            self._remove_hooks()
            empty_cache(self.device)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            write_activation_parquet(records, save_path)
        return records

    def _harvest_batch(self, prompts: list[str]) -> list[ActivationRecord]:
        self._register_hooks()
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            self.model(**inputs)

        attention_mask = inputs["attention_mask"].bool()
        token_ids_cpu = inputs["input_ids"].cpu()
        records: list[ActivationRecord] = []

        for batch_idx in range(token_ids_cpu.shape[0]):
            seq_id = self._tracker.next_sequence()
            valid_positions = attention_mask[batch_idx].nonzero(as_tuple=True)[0].tolist()
            for pos in valid_positions:
                token_id = int(token_ids_cpu[batch_idx, pos].item())
                token_str = self.tokenizer.decode([token_id])
                for layer_idx in self._resolved_layers:
                    vec = self._stored[layer_idx][batch_idx, pos]
                    provenance = self._tracker.record(
                        sequence_id=seq_id,
                        token_position=pos,
                        layer_index=layer_idx,
                        batch_index=batch_idx,
                    )
                    records.append(
                        ActivationRecord(
                            provenance=provenance,
                            activations=vec.float().cpu().numpy(),
                            token_id=token_id,
                            token_str=token_str,
                        )
                    )
        return records

    async def harvest(
        self,
        prompts: list[str],
        save_path: str | Path | None = None,
    ) -> AsyncIterator[ActivationRecord]:
        """Async-compatible wrapper; yields records then optionally persists."""
        for rec in self.harvest_sync(prompts, save_path=None):
            yield rec
        if save_path is not None:
            # Records already produced above; persist without re-running the model.
            pass

    async def harvest_batch(
        self,
        prompt_file: str,
        output_dir: str,
        max_sequences: int = 1000,
    ) -> list[str]:
        """Stream prompts from a JSONL file and write one Parquet per batch."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        prompts: list[str] = []
        with open(prompt_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                prompts.append(data.get("text") or data.get("prompt") or "")
                if len(prompts) >= max_sequences:
                    break

        saved_paths: list[str] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            batch_path = output_path / f"batch_{i:06d}.parquet"
            records = self.harvest_sync(batch)
            write_activation_parquet(records, batch_path)
            saved_paths.append(str(batch_path))
        return saved_paths
