from functools import lru_cache

import pytest
from trl import SFTTrainer

from conform.probes.p3_template import compare_row, match_rows, run
from conform.proxy_model import DEFAULT_TARGETS, LLAMA_TYPE
from conform.result import EvidenceClass, Status


class DoubleBOSSFTTrainer(SFTTrainer):
    """Injected defect: prepend an extra BOS to every prepared training row."""

    def _prepare_dataset(self, dataset, processing_class, *args, **kwargs):
        dataset = super()._prepare_dataset(dataset, processing_class, *args, **kwargs)
        bos = processing_class.bos_token_id  # self.processing_class is not set yet at this point
        return dataset.map(lambda ex: {"input_ids": [bos] + ex["input_ids"], "labels": [-100] + ex["labels"]})


@lru_cache(maxsize=None)
def cached_run(model_id, data_format="prompt_completion", trainer_cls=None, trainer="trl"):
    result = run(model_id, trainer=trainer, data_format=data_format, trainer_cls=trainer_cls)
    if result.status is Status.NOT_TESTED and (result.reason or "").startswith("OSError"):
        pytest.skip(f"{model_id} not downloaded; run scripts/download_targets.py")
    return result


def test_compare_row_facts():
    prompt, full = [1, 5, 6], [1, 5, 6, 7, 2]
    assert compare_row([1, 5, 6, 7, 2], prompt, full)["after_template"] == "identical to the template"
    extra = compare_row([1, 5, 6, 7, 2, 2], prompt, full)
    assert extra["prefix_match"] and extra["extra_tokens"] == [2]
    doubled = compare_row([1, 1, 5, 6, 7, 2], prompt, full)
    assert not doubled["prefix_match"] and doubled["first_difference"] == 1


def test_match_rows_survives_a_broken_prefix_and_refuses_ties():
    fulls = [[1, 5, 6, 7, 2], [1, 8, 9, 3, 2]]
    assert match_rows([[1, 5, 6, 7, 2], [1, 8, 9, 3, 2]], fulls) == [0, 1]
    assert match_rows([[1, 1, 8, 9, 3, 2]], fulls) == [1]  # extra BOS still matched
    assert match_rows([[4]], [[5], [6]]) == [None]


@pytest.mark.parametrize("model_id", DEFAULT_TARGETS)
@pytest.mark.parametrize("data_format", ["prompt_completion", "messages"])
def test_current_trl_passes(model_id, data_format):
    result = cached_run(model_id, data_format)
    assert result.status is Status.PASS, result.to_dict()


def test_injected_double_bos_fails():
    result = cached_run(LLAMA_TYPE, trainer_cls=DoubleBOSSFTTrainer)
    assert result.status is Status.FAIL, result.to_dict()
    differences = [f.evidence for f in result.findings if "first_difference" in f.evidence]
    assert differences and all(d["first_difference"] == 1 for d in differences)
    assert any(f.evidence_class is EvidenceClass.INFERRED for f in result.findings)


def test_transformers_trainer_is_not_tested():
    result = run(LLAMA_TYPE, trainer="transformers")
    assert result.status is Status.NOT_TESTED
    assert "applies no chat template" in result.reason
