import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from probekit.config import ModelConfig
from probekit.harvest import ActivationHarvester


@pytest.fixture
def tiny_harvester() -> ActivationHarvester:
    vocabulary = ["<pad>", "<unk>", "a", "b", "c", "d", "e", "f", "g"]
    backend = Tokenizer(
        WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>"
    )
    torch.manual_seed(42)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=16,
            n_embd=16,
            n_layer=3,
            n_head=2,
        )
    )
    config = ModelConfig(revision="tiny-test", layer=1, device="cpu", batch_size=4, cpu_threads=2)
    return ActivationHarvester(config, model, tokenizer)
