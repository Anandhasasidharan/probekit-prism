"""SAE training loop over harvested activation matrices.

Input can be a stacked numpy matrix (n_tokens, d_model) or a Parquet file
produced by probekit.harvest. Designed to fit a GPT-2-Small SAE comfortably
in an RTX 3050 (4-6 GB VRAM); batches are moved to GPU one at a time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from probekit.core.device import empty_cache, get_device
from probekit.sae.architectures import BaseSAE, build_sae


class SAETrainer:
    def __init__(
        self,
        sae: BaseSAE | None = None,
        *,
        architecture: str = "topk",
        d_model: int = 768,
        dict_size: int = 16384,
        k: int = 32,
        sparsity_coef: float = 0.1,
        lr: float = 3e-4,
        batch_size: int = 256,
        device: str | None = None,
        renorm_every: int = 100,
    ):
        self.sae = sae or build_sae(
            architecture, d_model=d_model, dict_size=dict_size, k=k,
            sparsity_coef=sparsity_coef, device=device,
        )
        self.lr = lr
        self.batch_size = batch_size
        self.device = get_device(device)
        self.renorm_every = renorm_every
        self.history: list[dict] = []

    def _iterate(self, data: np.ndarray | str | Path):
        if isinstance(data, (str, Path)):
            from probekit.harvest.compression import load_layer_matrix

            matrix, _ = load_layer_matrix(data, layer=-1)
        else:
            matrix = np.asarray(data, dtype=np.float32)
        n = matrix.shape[0]
        perm = np.random.permutation(n)
        for start in range(0, n, self.batch_size):
            idx = perm[start : start + self.batch_size]
            yield torch.from_numpy(matrix[idx]).to(self.device)

    def train(
        self,
        data: np.ndarray | str | Path,
        epochs: int = 5,
        log_every: int = 50,
    ) -> list[dict]:
        opt = torch.optim.AdamW(self.sae.parameters(), lr=self.lr, weight_decay=0.0)
        step = 0

        for epoch in range(epochs):
            for batch in self._iterate(data):
                opt.zero_grad(set_to_none=True)
                recon, acts = self.sae(batch)
                mse = torch.nn.functional.mse_loss(recon, batch)
                loss = mse
                if hasattr(self.sae, "sparsity_coef"):
                    loss = loss + self.sae.sparsity_coef * acts.abs().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
                opt.step()

                step += 1
                if isinstance(self.sae, BaseSAE) and step % self.renorm_every == 0:
                    self.sae.renorm_decoder()

                if step % log_every == 0 or epoch == epochs - 1:
                    self.history.append(
                        {
                            "step": step,
                            "epoch": epoch,
                            "loss": float(loss.item()),
                            "mse": float(mse.item()),
                        }
                    )
        empty_cache(self.device)
        return self.history

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.sae.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.sae.load_state_dict(state)
