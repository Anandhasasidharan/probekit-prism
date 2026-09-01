"""Explanation quality scoring (EleutherAI Delphi methodology).

Detection score: can an LLM, given only the explanation, identify which of a
pair of sentences strongly/weakly activates the feature? 1.0 = perfect.

Fuzzing score: token-level variant — replace one token in an activating
sentence and check whether the LLM identifies the substituted token as not
activating.
"""

from __future__ import annotations

import re

import numpy as np

DETECT_PROMPT = """Explanation of a feature: "{explanation}"

Sentence A: {a!r}
Sentence B: {b!r}

Which sentence activates the feature more strongly? Answer with exactly 'A' or 'B'."""

FUZZ_PROMPT = """Explanation of a feature: "{explanation}"

Original sentence: {original!r}
Modified sentence: {modified!r}

In the modified sentence, does the replaced word still activate the feature?
Answer with exactly 'YES' or 'NO'."""


def _ask_llm(explainer, prompt: str) -> str:
    try:
        return explainer.answer(prompt)
    except Exception:
        return ""


def detection_score(
    explainer,
    explanation: str,
    positive_examples: list[str],
    negative_examples: list[str],
    n_pairs: int = 10,
    seed: int = 0,
) -> float:
    """Fraction of A/B trials where the LLM picks the true activator."""
    rng = np.random.default_rng(seed)
    n = min(n_pairs, len(positive_examples), len(negative_examples))
    if n == 0:
        return 0.0
    correct = 0
    total = 0
    for _ in range(n):
        p = positive_examples[int(rng.integers(len(positive_examples)))]
        q = negative_examples[int(rng.integers(len(negative_examples)))]
        flip = bool(rng.random() < 0.5)
        a, b = (q, p) if flip else (p, q)
        expected = "B" if flip else "A"
        prompt = DETECT_PROMPT.format(explanation=explanation, a=a, b=b)
        out = _ask_llm(explainer, prompt).strip().upper()
        if out.startswith(expected):
            correct += 1
        total += 1
    return correct / max(total, 1)


def fuzzing_score(
    explainer,
    explanation: str,
    positive_examples: list[str],
    distractor_words: list[str],
    n_trials: int = 10,
    seed: int = 0,
) -> float:
    """Fraction of fuzzed tokens correctly identified as non-activating."""
    rng = np.random.default_rng(seed)
    usable = [s for s in positive_examples if len(s.split()) >= 3]
    if not usable or not distractor_words:
        return 0.0
    correct, total = 0, 0
    for _ in range(n_trials):
        sent = usable[int(rng.integers(len(usable)))]
        words = sent.split()
        i = int(rng.integers(len(words)))
        original_word = words[i]
        replacement = distractor_words[int(rng.integers(len(distractor_words)))]
        modified = " ".join(words[:i] + [replacement] + words[i + 1:])
        out = _ask_llm(
            explainer,
            FUZZ_PROMPT.format(explanation=explanation, original=sent, modified=modified),
        ).strip().upper()
        # Correct behaviour: NO — the swapped-in distractor should NOT activate.
        if out.startswith("NO"):
            correct += 1
        total += 1
    del original_word
    return correct / max(total, 1)


def extract_activating_token(sentence: str) -> str:
    """Utility used by tests/mocks: crude 'most salient' token extraction."""
    words = re.findall(r"[a-zA-Z']+", sentence)
    return words[-1] if words else ""
