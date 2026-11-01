import dataclasses
from functools import lru_cache

import pytest
import torch

from conform.probes.p2_loss_tokens import _supported, analyse_row, run
from conform.proxy_model import LLAMA_TYPE, QWEN_TYPE
from conform.result import EvidenceClass, Status

BOS, EOS, PAD = 1, 2, 2  # a PAD = EOS tokenizer, like SmolLM2


def facts(ids, labels, mask=None, prompt_tokens=None):
    return analyse_row(
        torch.tensor(ids), torch.tensor(labels), None if mask is None else torch.tensor(mask),
        prompt_tokens=prompt_tokens, bos_id=BOS, eos_id=EOS, pad_id=PAD,
    )


@lru_cache(maxsize=None)
def cached_run(model_id, trainer, intent_items=()):
    result = run(model_id, trainer=trainer, intent=dict(intent_items) or None)
    if result.status is Status.NOT_TESTED and result.reason.startswith("OSError"):
        pytest.skip(f"{model_id} not downloaded; run scripts/download_targets.py")
    return result


def violated_intents(result):
    return {f.evidence["intent"] for f in result.findings
            if f.evidence_class is EvidenceClass.SEEN and "intent" in f.evidence}


def test_analyse_row_spans_and_eos():
    row = facts(ids=[1, 10, 11, 2, 12, 13, 2, 2],
                labels=[-100, -100, -100, -100, 12, 13, 2, -100],
                mask=[1, 1, 1, 1, 1, 1, 1, 0], prompt_tokens=4)
    assert row.real_tokens == 7
    assert row.supervised_spans == [(4, 7)]
    assert row.final_eos_supervised is True
    assert row.leading_bos == 1
    assert row.supervised_prompt_tokens == 0
    assert row.pad_id_real_tokens == 2  # both real EOS tokens share the pad ID
    assert row.pad_id_real_tokens_masked == 1


def test_masked_final_eos_is_reported():
    row = facts(ids=[1, 10, 11, 2, 12, 13, 2, 2],
                labels=[-100, -100, -100, -100, 12, 13, -100, -100],
                mask=[1, 1, 1, 1, 1, 1, 1, 0])
    assert row.final_eos_supervised is False
    assert row.pad_id_real_tokens_masked == 2


def test_position_zero_never_carries_loss_and_double_bos():
    row = facts(ids=[1, 1, 10, 11, 2], labels=[1, 1, 10, 11, 2], prompt_tokens=2)
    assert row.supervised_spans == [(1, 5)]
    assert row.leading_bos == 2
    assert row.supervised_prompt_tokens == 1


def test_transformers_collator_masks_eos_when_pad_equals_eos():
    result = cached_run(LLAMA_TYPE, "transformers", (("supervise_eos", True),))
    assert result.status is Status.FAIL, result.to_dict()
    eos = next(f for f in result.findings if "final EOS" in f.claim)
    assert eos.evidence == {"rows_checked": 4, "eos_supervised": 0}
    assert violated_intents(result) == {"supervise_eos"}
    assert any(f.evidence_class is EvidenceClass.INFERRED and "P7" in f.claim for f in result.findings)


def test_trl_supervises_eos_and_masks_prompt():
    intent = (("supervise_eos", True), ("mask_prompt", True), ("single_bos", True))
    result = cached_run(LLAMA_TYPE, "trl", intent)
    assert result.status is Status.PASS, result.to_dict()


def test_qwen_transformers_supervises_eos_but_trains_prompt():
    result = cached_run(QWEN_TYPE, "transformers", (("supervise_eos", True), ("mask_prompt", True)))
    assert result.status is Status.FAIL, result.to_dict()
    assert violated_intents(result) == {"mask_prompt"}


def test_supported_keeps_known_options_and_lists_the_rest():
    @dataclasses.dataclass
    class OldConfig:
        packing: bool = True

    kept, dropped = _supported(OldConfig, {"packing": False, "padding_free": False})
    assert kept == {"packing": False}
    assert dropped == ["padding_free"]


def test_trl_messages_format_trains_whole_conversation():
    result = run(LLAMA_TYPE, trainer="trl", data_format="messages",
                 intent={"supervise_eos": True, "mask_prompt": True})
    if result.status is Status.NOT_TESTED and result.reason.startswith("OSError"):
        pytest.skip("tokenizer not downloaded")
    assert violated_intents(result) == {"mask_prompt"}  # EOS trained, prompt trained too
    assert result.observed["data_format"] == "messages"


def test_no_intent_reports_facts_only():
    result = cached_run(QWEN_TYPE, "trl")
    assert result.status is Status.INFO, result.to_dict()
    unknowns = [f for f in result.findings if f.evidence_class is EvidenceClass.UNKNOWN]
    assert [f.claim for f in unknowns] == ["No intent declared: P2 reports facts only and cannot pass or fail"]
    assert all(f.evidence_class is EvidenceClass.SEEN for f in result.findings if f not in unknowns)
