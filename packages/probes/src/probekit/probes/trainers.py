"""Three-way probe trainers: raw activations vs. SAE latents vs. circuit
features. All use L2-regularised logistic regression (Goodfire/Rakuten setup)
and expose identical fit/score/export interfaces.

Feature spaces:
  - raw:      harvested residual-stream activations directly
  - sae:      activations encoded through a trained SAE (sparse, monosemantic)
  - circuit:  CLT feature activations at a chosen layer (causally validated
              directions from the attribution pipeline)
"""

from __future__ import annotations

import numpy as np
from probekit.probes.dataset import ProbeDataset
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _make_lr(C: float = 1.0, seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=2000, random_state=seed)),
    ])


class BaseProbe:
    kind = "base"

    def __init__(self, concept: str, C: float = 1.0):
        self.concept = concept
        self.model = _make_lr(C=C)

    def fit(self, ds: ProbeDataset) -> BaseProbe:
        self.model.fit(ds.X, ds.y)
        return self

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        return self.model.decision_function(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------- transforms

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def score_dataset(self, ds: ProbeDataset) -> np.ndarray:
        return self.predict_proba(self._transform(ds.X))


class RawProbe(BaseProbe):
    """v1 baseline: logistic regression on raw activations."""

    kind = "raw"


class SAEProbe(BaseProbe):
    """Rakuten/Goodfire-style probe on SAE latent space."""

    kind = "sae"

    def __init__(self, concept: str, sae=None, C: float = 1.0, device: str | None = None):
        super().__init__(concept, C=C)
        self.sae = sae
        self.device = device

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.sae is None:
            raise ValueError("SAEProbe requires a trained SAE")
        from probekit.sae.evaluator import encode_matrix

        return encode_matrix(self.sae, X, device=self.device)

    def fit(self, ds: ProbeDataset) -> SAEProbe:
        Xs = self._transform(ds.X)
        wrapped = ProbeDataset(X=Xs, y=ds.y, texts=ds.texts, sources=ds.sources, concept=ds.concept)
        self.model.fit(wrapped.X, wrapped.y)
        return self


class CircuitProbe(BaseProbe):
    """Probe on circuit-derived features (CLT latent space)."""

    kind = "circuit"

    def __init__(self, concept: str, clt_layer_transcoder=None, C: float = 1.0,
                 device: str | None = None):
        super().__init__(concept, C=C)
        self.transcoder = clt_layer_transcoder
        self.device = device

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.transcoder is None:
            raise ValueError("CircuitProbe requires a trained CLT transcoder")
        import torch

        dev = torch.device(self.device) if self.device else next(
            self.transcoder.parameters()).device
        out = []
        with torch.no_grad():
            for start in range(0, len(X), 512):
                batch = torch.from_numpy(np.asarray(X[start:start + 512], dtype=np.float32)).to(dev)
                out.append(self.transcoder.encode(batch).cpu().numpy())
        return np.concatenate(out, axis=0)

    def fit(self, ds: ProbeDataset) -> CircuitProbe:
        Xc = self._transform(ds.X)
        self.model.fit(Xc, ds.y)
        return self
