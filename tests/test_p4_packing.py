import tempfile
from functools import lru_cache

import pytest
import torch

from conform.noise_floor import ABOVE
from conform.probes.p4_packing import PACK_LENGTH, documents, measure_leak, run
from conform.proxy_model import DEFAULT_TARGETS, LLAMA_TYPE, build_proxy
from conform.result import Status
from conform.stack import SAMPLES, attention_kernel, build_trainer, conversations


@lru_cache(maxsize=None)
def cached_run(model_id, trainer="trl"):
    result = run(model_id, trainer=trainer)
    if result.status is Status.NOT_TESTED and (result.reason or "").startswith("OSError"):
        pytest.skip(f"{model_id} not downloaded; run scripts/download_targets.py")
    return result


def test_documents_from_position_ids():
    assert documents(torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3])) == [(0, 3), (3, 5), (5, 9)]


@pytest.mark.parametrize("model_id", DEFAULT_TARGETS)
def test_current_trl_packing_isolates_documents(model_id):
    result = cached_run(model_id)
    assert result.status is Status.PASS, result.to_dict()
    assert result.observed["documents"] >= 2 and result.observed["gap"] == 0


def test_measurement_detects_a_leak_when_boundaries_are_withheld():
    try:
        model, tokenizer, _ = build_proxy(LLAMA_TYPE)
    except OSError:
        pytest.skip("tokenizer not downloaded")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as out:
        built, _ = build_trainer("trl", model, tokenizer, conversations(SAMPLES), output_dir=out, seed=0,
                                 trainer_args={"packing": True, "max_length": PACK_LENGTH})
        batch = next(iter(built.get_train_dataloader()))
        leak = measure_leak(built.model, batch, len(tokenizer), drop_boundaries=True)
    assert leak["verdict"] == ABOVE and leak["gap"] > 1e-3, leak


def test_private_access_is_marked_fragile_exactly_when_used():
    model, _, _ = build_proxy(LLAMA_TYPE)
    _, fragile = attention_kernel(model)
    result = cached_run(LLAMA_TYPE)
    assert any(item.startswith("fragile:") for item in result.limitations) == fragile


def test_transformers_trainer_is_not_tested():
    assert run(LLAMA_TYPE, trainer="transformers").status is Status.NOT_TESTED
