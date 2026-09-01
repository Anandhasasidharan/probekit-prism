"""Steering evaluation: behavior shift vs. coherence drop
(the LessWrong 'Mind the Coherence Gap' methodology).

Behavior shift: mean probe-score change across a prompt set when steering.
Coherence drop: perplexity change of the model's own generations — a strong
intervention that crushes coherence is not usable regardless of behavior shift.
"""

from __future__ import annotations

import numpy as np
import torch
from probekit.steering.controller import SteeringController


@torch.no_grad()
def sequence_nll(model, tokenizer, text: str, device) -> float:
    enc = tokenizer(text, return_tensors="pt").to(device)
    if enc["input_ids"].shape[1] < 2:
        return 0.0
    out = model(**enc, labels=enc["input_ids"])
    return float(out.loss.item())


def evaluate_steering(
    controller: SteeringController,
    prompts: list[str],
    threshold: float = 0.5,
    max_new_tokens: int = 24,
) -> dict:
    """Run steered vs. unsteered passes; return aggregate metrics."""
    before_scores, after_scores = [], []
    ppl_clean, ppl_steered = [], []

    for prompt in prompts:
        result = controller.check_and_steer(prompt, threshold=threshold)
        before_scores.append(result.before_score)
        after_scores.append(result.after_score)

        clean_gen = controller.generate(
            prompt, max_new_tokens=max_new_tokens, steer=False
        )
        steered_gen = controller.generate(
            prompt, max_new_tokens=max_new_tokens, steer=True
        )
        ppl_clean.append(
            sequence_nll(
                controller.model, controller.tokenizer, clean_gen, controller.device
            )
        )
        ppl_steered.append(
            sequence_nll(
                controller.model, controller.tokenizer, steered_gen, controller.device
            )
        )

    before = np.asarray(before_scores, dtype=np.float64)
    after = np.asarray(after_scores, dtype=np.float64)

    def _ppl(nll: list[float]) -> float:
        arr = np.asarray([x for x in nll if np.isfinite(x)])
        return float(np.exp(arr.mean())) if arr.size else float("inf")

    return {
        "n_prompts": len(prompts),
        "mean_before": float(before.mean()) if before.size else 0.0,
        "mean_after": float(after.mean()) if after.size else 0.0,
        "behavior_shift": float((after - before).mean()) if before.size else 0.0,
        "steered_fraction": float(np.mean(after_scores != before_scores)),
        "perplexity_clean": _ppl(ppl_clean),
        "perplexity_steered": _ppl(ppl_steered),
        "coherence_drop": (
            _ppl(ppl_steered) / _ppl(ppl_clean) if _ppl(ppl_clean) > 0 else float("inf")
        ),
    }
