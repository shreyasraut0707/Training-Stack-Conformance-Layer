from functools import lru_cache

import pytest
from trl import SFTTrainer

from conform.probes.p7_eos_stop import run
from conform.proxy_model import LLAMA_TYPE, QWEN_TYPE
from conform.result import EvidenceClass, Status


class NoTrainingSFTTrainer(SFTTrainer):
    """Injected stand-in for a run too short to learn anything: training does nothing."""

    def train(self, *args, **kwargs):
        return None


@lru_cache(maxsize=None)
def cached_run(model_id, trainer, trainer_cls=None):
    result = run(model_id, trainer=trainer, trainer_cls=trainer_cls)
    if result.status is Status.NOT_TESTED and (result.reason or "").startswith("OSError"):
        pytest.skip(f"{model_id} not downloaded; run scripts/download_targets.py")
    return result


def test_trl_learns_to_stop():
    result = cached_run(LLAMA_TYPE, "trl")
    assert result.status is Status.PASS, result.to_dict()
    assert result.observed["after"]["eos_probability"] > result.observed["before"]["eos_probability"]
    assert all(g["stopped"] for g in result.observed["generations"])


def test_pad_equals_eos_collator_does_not_learn_to_stop():
    """The measured harm behind P2's inference: content is learned, stopping is not."""
    result = cached_run(LLAMA_TYPE, "transformers")
    assert result.status is Status.FAIL, result.to_dict()
    observed = result.observed
    assert observed["after"]["content_probability"] > observed["before"]["content_probability"]
    assert observed["after"]["eos_probability"] < observed["before"]["eos_probability"]
    assert not any(g["stopped"] for g in observed["generations"])


def test_collator_learns_to_stop_when_pad_differs_from_eos():
    assert cached_run(QWEN_TYPE, "transformers").status is Status.PASS


def test_untrained_run_is_unknown_not_fail():
    result = cached_run(LLAMA_TYPE, "trl", NoTrainingSFTTrainer)
    assert result.status is Status.INFO, result.to_dict()
    assert any(f.evidence_class is EvidenceClass.UNKNOWN and "did not learn the completions" in f.claim
               for f in result.findings)
