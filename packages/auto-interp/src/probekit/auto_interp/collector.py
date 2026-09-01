"""Top-activating and quantile example collection for AutoInterp.

Given a feature-activation matrix (n_tokens, n_features) produced by encoding
harvested activations through an SAE or CLT, plus the aligned token strings,
collect the examples a human (or LLM) would need to explain the feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FeatureExamples:
    feature_id: int
    top_examples: list[str] = field(default_factory=list)
    quantile_examples: list[str] = field(default_factory=list)  # 25/50/75th pct
    random_negatives: list[str] = field(default_factory=list)
    frequency: float = 0.0

    def all_positive(self) -> list[str]:
        seen, out = set(), []
        for t in self.top_examples + self.quantile_examples:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def as_prompt_block(self) -> str:
        pos = "\n".join(f"- {t!r}" for t in self.all_positive()[:20])
        neg = "\n".join(f"- {t!r}" for t in self.random_negatives[:10])
        return (
            f"Examples where the feature activates strongly:\n{pos}\n\n"
            f"Random examples where it does NOT activate:\n{neg}"
        )


def collect_examples(
    feature_id: int,
    acts_matrix: np.ndarray,
    tokens: list[str],
    top_k: int = 50,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    n_negatives: int = 10,
    seed: int = 0,
) -> FeatureExamples:
    """Collect top-activating + quantile + negative examples for one feature."""
    col = np.asarray(acts_matrix[:, feature_id], dtype=np.float32)
    order = np.argsort(-col)

    top_idx = [int(i) for i in order[:top_k] if col[i] > 1e-6]

    active_mask = col > max(col.max() * 0.1, 1e-8)
    active_vals = col[active_mask]
    quantile_idx: list[int] = []
    if active_vals.size >= 4:
        for q in quantiles:
            qv = float(np.quantile(active_vals, q))
            matches = np.where(np.abs(col - qv) < 1e-9)[0]
            if matches.size:
                quantile_idx.append(int(matches[0]))

    rng = np.random.default_rng(seed)
    inactive = np.where(~active_mask)[0]
    negative_idx = (
        [int(i) for i in rng.choice(inactive, size=min(n_negatives, inactive.size), replace=False)]
        if inactive.size
        else []
    )

    def _tok(i: int) -> str:
        return tokens[i] if 0 <= i < len(tokens) else ""

    freq = float((col > max(col.max() * 0.05, 1e-6)).mean())
    return FeatureExamples(
        feature_id=feature_id,
        top_examples=[_tok(i) for i in top_idx],
        quantile_examples=[_tok(i) for i in quantile_idx],
        random_negatives=[_tok(i) for i in negative_idx],
        frequency=freq,
    )
