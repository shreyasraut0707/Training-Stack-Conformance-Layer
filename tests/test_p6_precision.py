import json
from functools import lru_cache

import pytest

from conform.probes.p6_precision import run
from conform.proxy_model import LLAMA_TYPE
from conform.result import EvidenceClass, Status


@lru_cache(maxsize=None)
def cached_run(trainer, trainer_args_json=None):
    result = run(LLAMA_TYPE, trainer=trainer,
                 trainer_args=json.loads(trainer_args_json) if trainer_args_json else None)
    if result.status is Status.NOT_TESTED and result.reason.startswith("OSError"):
        pytest.skip("tokenizer not downloaded; run scripts/download_targets.py")
    return result


@pytest.mark.parametrize("trainer", ["transformers", "trl"])
def test_p6_never_fails_and_reports_only_seen_facts(trainer):
    result = cached_run(trainer)
    assert result.status is Status.INFO, result.to_dict()
    assert all(f.evidence_class is EvidenceClass.SEEN for f in result.findings)
    for key in ("parameters", "compute", "gradients", "optimizer_state"):
        assert result.observed[key], key


def test_transformers_trainer_defaults_to_float32():
    observed = cached_run("transformers").observed
    assert observed["compute"] == ["float32"]
    assert observed["settings"]["bf16"] == {"value": False, "source": "library default"}


def test_trl_turns_bf16_on_by_default():
    result = cached_run("trl")
    assert result.observed["compute"] == ["bfloat16"]
    assert result.observed["parameters"] == ["float32"]
    silent = next(f for f in result.findings if "silent_defaults" in f.evidence)
    assert silent.evidence["silent_defaults"] == ["bf16"]


def test_recipe_setting_is_marked_and_applied():
    result = cached_run("trl", json.dumps({"bf16": False}))
    assert result.observed["settings"]["bf16"] == {"value": False, "source": "recipe"}
    assert result.observed["compute"] == ["float32"]
    assert "--trainer-args" in result.reproducer
