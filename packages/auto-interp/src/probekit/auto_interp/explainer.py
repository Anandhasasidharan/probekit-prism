"""LLM-based feature explainers.

Providers:
  - "openai"    -> OpenAI Chat Completions API (gpt-4o-mini default)
  - "anthropic" -> Anthropic Messages API (claude-haiku default)
  - "mock"      -> deterministic offline explainer for tests / demos

All HTTP calls go through httpx — no heavyweight vendor SDK required.
"""

from __future__ import annotations

import os

import httpx

EXPLAIN_SYSTEM_PROMPT = (
    "You are an expert mechanistic interpretability researcher. You will see "
    "examples of text where a sparse-autoencoder feature activates strongly, "
    "and random examples where it does not. Write a single-sentence "
    "description of what the feature detects, in the form: 'Activates on ...'. "
    "Be specific and concrete; do not speculate beyond the examples."
)

ANSWER_SYSTEM_PROMPT = "You are an expert mechanistic interpretability researcher."


class MockExplainer:
    """Deterministic offline explainer: labels a feature by its most frequent
    activating token pattern. Used in tests and CPU-only demos."""

    name = "mock"

    def explain(self, examples) -> str:
        pos = examples.all_positive()
        if not pos:
            return "No activation examples found."
        from collections import Counter

        counts = Counter(t.strip().lower() for t in pos if t.strip())
        top = counts.most_common(3)
        listed = ", ".join(f"'{t}'" for t, _ in top)
        return f"Activates on text containing {listed}"

    def answer(self, question: str) -> str:
        first = question.strip().splitlines()[0] if question.strip() else "(empty)"
        return f"[mock circuit narration] {first[:200]}"


class OpenAIExplainer:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1", timeout: float = 60.0):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cost_usd_total = 0.0

    @property
    def name(self) -> str:
        return self.model

    def _chat(self, system: str, user: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
                "max_tokens": 300,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        # Rough cost model for gpt-4o-mini class models.
        self.cost_usd_total += 0.15 * usage.get("prompt_tokens", 0) / 1e6
        self.cost_usd_total += 0.60 * usage.get("completion_tokens", 0) / 1e6
        return data["choices"][0]["message"]["content"].strip()

    def explain(self, examples) -> str:
        return self._chat(EXPLAIN_SYSTEM_PROMPT, examples.as_prompt_block())

    def answer(self, question: str) -> str:
        return self._chat(ANSWER_SYSTEM_PROMPT, question)


class AnthropicExplainer(OpenAIExplainer):
    def __init__(self, model: str = "claude-3-5-haiku-latest",
                 api_key: str | None = None, timeout: float = 60.0):
        super().__init__(model=model, api_key=api_key)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.timeout = timeout

    def _chat(self, system: str, user: str) -> str:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": 300,
                "temperature": 0.0,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        self.cost_usd_total += 0.80 * usage.get("input_tokens", 0) / 1e6
        self.cost_usd_total += 4.00 * usage.get("output_tokens", 0) / 1e6
        return "".join(b.get("text", "") for b in data.get("content", [])).strip()


def build_explainer(provider: str, model: str | None = None, **kwargs):
    provider = provider.lower()
    if provider == "openai":
        return OpenAIExplainer(model=model or "gpt-4o-mini", **kwargs)
    if provider == "anthropic":
        return AnthropicExplainer(model=model or "claude-3-5-haiku-latest", **kwargs)
    if provider == "mock":
        return MockExplainer()
    raise ValueError(f"Unknown explainer provider: {provider}")
