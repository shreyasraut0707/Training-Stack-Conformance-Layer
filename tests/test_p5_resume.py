import json
from functools import lru_cache

import pytest
from transformers import Trainer

from conform.probes.p5_resume import run
from conform.proxy_model import LLAMA_TYPE
from conform.result import EvidenceClass, Status


class NoOptimizerRestoreTrainer(Trainer):
    """Injected defect: a resume that does not restore the optimizer and scheduler state."""

    def _load_optimizer_and_scheduler(self, checkpoint):
        return None


@lru_cache(maxsize=None)
def cached_run(trainer="transformers", trainer_cls=None, trainer_args_json=None):
    result = run(LLAMA_TYPE, trainer=trainer, trainer_cls=trainer_cls,
                 trainer_args=json.loads(trainer_args_json) if trainer_args_json else None)
    if result.status is Status.NOT_TESTED and (result.reason or "").startswith("OSError"):
        pytest.skip("tokenizer not downloaded; run scripts/download_targets.py")
    return result


def _verdicts(result):
    return {f.evidence["metric"]: f.evidence["verdict"] for f in result.findings if "metric" in f.evidence}


@pytest.mark.parametrize("trainer", ["transformers", "trl"])
def test_current_resume_is_equivalent(trainer):
    result = cached_run(trainer)
    assert result.status is Status.PASS, result.to_dict()
    assert _verdicts(result) == {"parameters": "inside", "learning_rate": "inside", "data_order": "inside"}


def test_unrestored_optimizer_and_scheduler_fail():
    result = cached_run(trainer_cls=NoOptimizerRestoreTrainer)
    assert result.status is Status.FAIL, result.to_dict()
    verdicts = _verdicts(result)
    assert verdicts["parameters"] == "above" and verdicts["learning_rate"] == "above"
    assert any(f.evidence_class is EvidenceClass.INFERRED for f in result.findings)


def test_restarting_the_data_on_resume_fails():
    result = cached_run(trainer_args_json=json.dumps({"ignore_data_skip": True}))
    assert result.status is Status.FAIL, result.to_dict()
    assert _verdicts(result)["data_order"] == "above"
