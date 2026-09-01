"""Shared test fixtures: a tiny random-weight GPT-2 so no downloads are needed."""

from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

TINY_KWARGS = dict(
    vocab_size=128,
    n_positions=64,
    n_embd=32,
    n_layer=4,
    n_head=4,
    bos_token_id=0,
    eos_token_id=0,
)


@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    config = GPT2Config(**TINY_KWARGS)
    model = GPT2LMHeadModel(config).eval()
    return model


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory):
    """A real (fast) tokenizer with a small vocab, built locally."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    # Build a small BPE over a toy corpus so common words exist.
    corpus = (
        "the cat sat on the mat\nthe dog ran in the park\ndelete all files now\n"
        "hello world this is a test\nmy name is john smith\nemail me at john@example com\n"
        "the sky is blue today\ni love pizza for dinner\nsecret password is hunter2\n"
        "call me at 555 123 4567 tomorrow\n" * 20
    )
    path = tmp_path_factory.mktemp("tok")
    tok_dir = path / "toy-tokenizer"
    tok_dir.mkdir()
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()
    from tokenizers.trainers import BpeTrainer

    trainer = BpeTrainer(vocab_size=120, special_tokens=["<|endoftext|>"])
    tokenizer.train_from_iterator([corpus], trainer=trainer)
    tokenizer.save(str(tok_dir / "tokenizer.json"))

    fast = GPT2TokenizerFast(tokenizer_file=str(tok_dir / "tokenizer.json"),
                             unk_token="<|endoftext|>", pad_token="<|endoftext|>")
    return fast
