import pytest
import torch

from conform.proxy_model import DEFAULT_TARGETS, PROXY_SHAPE, QWEN_TYPE, build_proxy


def build_or_skip(model_id, seed=0):
    try:
        return build_proxy(model_id, seed=seed)
    except OSError:
        pytest.skip(f"{model_id} not downloaded; run scripts/download_targets.py")


@pytest.mark.parametrize("model_id", DEFAULT_TARGETS)
def test_proxy_is_tiny_and_keeps_real_vocab(model_id):
    model, tokenizer, info = build_or_skip(model_id)
    assert info.num_hidden_layers == PROXY_SHAPE["num_hidden_layers"]
    assert info.hidden_size == PROXY_SHAPE["hidden_size"]
    assert model.get_input_embeddings().num_embeddings == info.vocab_size
    assert info.vocab_size >= len(tokenizer)
    assert all(p.dtype == torch.float32 for p in model.parameters())


@pytest.mark.parametrize("model_id", DEFAULT_TARGETS)
def test_proxy_runs_a_forward_pass(model_id):
    model, tokenizer, info = build_or_skip(model_id)
    input_ids = torch.tensor([tokenizer("What is 2+2?")["input_ids"]])
    logits = model(input_ids).logits
    assert logits.shape == (1, input_ids.shape[1], info.vocab_size)


@pytest.mark.parametrize("model_id", DEFAULT_TARGETS)
def test_proxy_records_hub_revision(model_id):
    _, _, info = build_or_skip(model_id)
    assert info.revision is not None
    assert len(info.revision) == 40
    assert all(c in "0123456789abcdef" for c in info.revision)


def test_same_seed_gives_identical_weights():
    model_a, _, _ = build_or_skip(QWEN_TYPE, seed=0)
    model_b, _, _ = build_or_skip(QWEN_TYPE, seed=0)
    for a, b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(a, b)
