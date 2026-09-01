"""End-to-end AutoInterp pipeline: feature -> examples -> label -> scores ->
registry-ready Explanation. Results are cached in SQLite."""

from __future__ import annotations

from probekit.auto_interp.cache import ExplanationCache, fingerprint_examples
from probekit.auto_interp.collector import collect_examples
from probekit.auto_interp.explainer import build_explainer
from probekit.auto_interp.scorer import detection_score, fuzzing_score


class AutoInterpPipeline:
    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        cache_db: str = "./data/features/explanation_cache.sqlite",
        top_k_examples: int = 50,
        quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
        distractor_words: list[str] | None = None,
        **explainer_kwargs,
    ):
        self.explainer = build_explainer(provider, model=model, **explainer_kwargs)
        self.cache = ExplanationCache(cache_db)
        self.top_k_examples = top_k_examples
        self.quantiles = quantiles
        self.distractor_words = distractor_words or [
            "banana", "Tuesday", "purple", "quietly", "keyboard", "cloud",
            "seven", "gently", "window", "paper",
        ]

    def run_feature(
        self,
        feature_id: int,
        acts_matrix,
        tokens: list[str],
        score: bool = True,
    ) -> dict:
        examples = collect_examples(
            feature_id,
            acts_matrix,
            tokens,
            top_k=self.top_k_examples,
            quantiles=self.quantiles,
        )
        key = self.cache.make_key(feature_id, self.explainer.name, fingerprint_examples(examples))
        cached = self.cache.get(key)
        if cached:
            return {**cached, "examples": examples}

        text = self.explainer.explain(examples)
        det = fuzz = None
        by_name = getattr(self.explainer, "name", "")
        if score and isinstance(by_name, str) and by_name != "mock":
            negatives = examples.random_negatives or ["the", "and", "of"]
            det = detection_score(self.explainer, text, examples.all_positive(), negatives)
            fuzz = fuzzing_score(
                self.explainer, text, examples.all_positive(), self.distractor_words
            )
        elif score and by_name == "mock":
            # Deterministic pseudo-scores for offline demos/tests.
            det = 1.0 if examples.all_positive() else 0.0
            fuzz = 1.0

        result = {
            "feature_id": feature_id,
            "text": text,
            "detection_score": det,
            "fuzzing_score": fuzz,
            "explained_by": self.explainer.name,
            "cost_usd": getattr(self.explainer, "cost_usd_total", 0.0),
            "examples": examples,
        }
        self.cache.put(
            key,
            feature_id=feature_id,
            text=text,
            detection_score=det,
            fuzzing_score=fuzz,
            explained_by=result["explained_by"],
            cost_usd=result["cost_usd"],
        )
        return result

    def close(self) -> None:
        self.cache.close()
