"""Three-way probe comparator: raw vs. SAE vs. circuit feature spaces.

Runs the same concept dataset through all three probes and reports a
comparison table — the core experimental artifact of ProbeKit v2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from probekit.probes.dataset import ProbeDataset
from probekit.probes.evaluator import evaluate_predictions, format_comparison_table


class ThreeWayComparator:
    def __init__(
        self,
        sae=None,
        clt_transcoder=None,
        device: str | None = None,
        C: float = 1.0,
    ):
        self.sae = sae
        self.clt = clt_transcoder
        self.device = device
        self.C = C
        self.results: list[dict] = []

    def compare_concept(self, train: ProbeDataset, test: ProbeDataset) -> list[dict]:
        from probekit.probes.trainers import CircuitProbe, RawProbe, SAEProbe

        results = []
        probe: RawProbe | SAEProbe | CircuitProbe
        X_test: np.ndarray
        for kind in ("raw", "sae", "circuit"):
            if kind == "sae" and self.sae is None:
                continue
            if kind == "circuit" and self.clt is None:
                continue

            if kind == "raw":
                probe = RawProbe(train.concept, C=self.C)
                X_test = test.X
            elif kind == "sae":
                probe = SAEProbe(
                    train.concept, sae=self.sae, C=self.C, device=self.device
                )
                with torch.no_grad():
                    X_test = probe._transform(test.X)
            else:
                probe = CircuitProbe(
                    train.concept, clt_layer_transcoder=self.clt, C=self.C, device=self.device
                )
                X_test = probe._transform(test.X)

            probe.fit(train)
            scores = probe.predict_proba(X_test)
            metrics = evaluate_predictions(test.y, scores, concept=train.concept, probe_kind=kind)
            metrics["n_train"] = int(len(train.y))
            results.append(metrics)

        # Keep fitted probes around for export / serving.
        self._last_probes = {r["probe_kind"]: None for r in results}
        self.results.extend(results)
        return results

    def table(self) -> str:
        return format_comparison_table(self.results)

    def report(self, path: str | Path) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps({"results": self.results}, indent=2)
        Path(path).write_text(content)
        return content
