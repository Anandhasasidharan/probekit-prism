"""Cross-layer transcoder training.

Each transcoder T_l minimises ||T_l(x_l) - y_l||^2 over collected (x_l = MLP
input, y_l = MLP output) pairs. Runs on CPU or CUDA; GPT-2-Small-scale CLTs
train in minutes on an RTX 3050.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from probekit.circuit.clt import CLTStack
from probekit.core.device import empty_cache, get_device


class CLTTrainer:
    def __init__(
        self,
        stack: CLTStack | None = None,
        *,
        layer_ids: list[int] | None = None,
        d_model: int = 768,
        n_features: int = 4096,
        lr: float = 1e-3,
        batch_size: int = 512,
        device: str | None = None,
        renorm_decoder: bool = False,
    ):
        self.stack = stack or CLTStack(
            layer_ids or [], d_model=d_model, n_features=n_features, device=device
        )
        self.lr = lr
        self.batch_size = batch_size
        self.device = get_device(device)
        self.history: list[dict] = []
        # NOTE: off by default — unit-renormalising decoder rows after training
        # destroys the learned output scale (MLP targets are not unit-norm).
        self.renorm_decoder = renorm_decoder

    def train_layer(
        self,
        layer: int,
        inputs: np.ndarray,
        outputs: np.ndarray,
        epochs: int = 10,
        log_every: int = 100,
    ) -> list[dict]:
        tc = self.stack.get(layer)
        opt = torch.optim.AdamW(tc.parameters(), lr=self.lr)
        n = inputs.shape[0]
        step = 0
        x_all = torch.from_numpy(inputs.astype(np.float32))
        y_all = torch.from_numpy(outputs.astype(np.float32))
        for epoch in range(epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = x_all[idx].to(self.device, non_blocking=True)
                yb = y_all[idx].to(self.device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                recon, _ = tc(xb)
                loss = torch.nn.functional.mse_loss(recon, yb) + 1e-4 * tc.b_out.pow(2).mean()
                loss.backward()
                opt.step()
                step += 1
                if step % log_every == 0:
                    self.history.append(
                        {"layer": layer, "step": step, "epoch": epoch, "loss": float(loss.item())}
                    )
        if self.renorm_decoder:
            with torch.no_grad():
                tc.renorm_decoder()
        empty_cache(self.device)
        return [h for h in self.history if h["layer"] == layer]

    def train(
        self,
        pairs: dict[int, tuple[np.ndarray, np.ndarray]],
        epochs: int = 10,
    ) -> dict[int, list[dict]]:
        results = {}
        for layer in sorted(pairs):
            xs, ys = pairs[layer]
            results[layer] = self.train_layer(layer, xs, ys, epochs=epochs)
        return results

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.stack.save(str(path))
