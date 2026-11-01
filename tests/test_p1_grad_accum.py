from functools import lru_cache

import pytest
import torch
from transformers import Trainer
from trl import SFTTrainer

from conform.noise_floor import ABOVE, INSIDE
from conform.probes.p1_grad_accum import _CapturingSGD, build_rows, run, supervised_targets
from conform.proxy_model import DEFAULT_TARGETS, LLAMA_TYPE
from conform.result import EvidenceClass, Status


class MeanOfMeansTrainer(Trainer):
    """Injected defect: withhold num_items_in_batch, so each micro-batch is averaged on
    its own and the Trainer divides by the number of accumulation steps (mean of means,
    the October 2024 behaviour)."""

    def training_step(self, model, inputs, num_items_in_batch=None):
        return super().training_step(model, inputs, num_items_in_batch=None)


class MeanOfMeansSFTTrainer(SFTTrainer):
    """Same injected defect as MeanOfMeansTrainer, on TRL's own loss path."""

    def training_step(self, model, inputs, num_items_in_batch=None):
        return super().training_step(model, inputs, num_items_in_batch=None)


class PositionZeroCountTrainer(Trainer):
    """Injected defect: count every non-ignored label, including position 0, which is
    never a prediction target (the behaviour of transformers 4.46 to 5.10)."""

    def _get_num_items_in_batch(self, batch_samples, device):
        return sum(b["labels"].ne(-100).sum() for b in batch_samples).to(device)


@lru_cache(maxsize=None)
def cached_run(model_id, trainer="transformers", trainer_cls=None):
    result = run(model_id, trainer=trainer, trainer_cls=trainer_cls)
    if result.status is Status.NOT_TESTED and result.reason.startswith("OSError"):
        pytest.skip(f"{model_id} not downloaded; run scripts/download_targets.py")
    return result


def assert_mean_of_means_pattern(result):
    """The defect breaks the accumulated run and leaves the one-batch run correct."""
    assert result.status is Status.FAIL, result.to_dict()
    verdicts = {(f.evidence["run"], f.evidence["metric"]): f.evidence["verdict"]
                for f in result.findings if "verdict" in f.evidence}
    assert verdicts[("accumulated (4 micro-batches x 1 row)", "loss")] == ABOVE
    assert verdicts[("accumulated (4 micro-batches x 1 row)", "update")] == ABOVE
    assert verdicts[("one batch (1 x 4 rows)", "loss")] == INSIDE
    assert verdicts[("one batch (1 x 4 rows)", "update")] == INSIDE
    assert verdicts[("accumulated vs one batch", "loss and update")] == ABOVE
    inferred = [f.claim for f in result.findings if f.evidence_class is EvidenceClass.INFERRED]
    assert inferred == ["Accumulating micro-batches does not give the same result as one large batch"]


def test_capturing_sgd_records_gradient_before_update():
    layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0]]))
    layer.weight.grad = torch.tensor([[0.5, -1.0]])
    optimizer = _CapturingSGD(layer.parameters(), lr=0.1)
    optimizer.step()
    assert optimizer.gradient.tolist() == [0.5, -1.0]
    assert torch.allclose(layer.weight, torch.tensor([[0.95, 2.1]]))


def test_rows_have_unequal_supervised_counts():
    assert supervised_targets(build_rows(1000, 0, seed=0)) == [14, 8, 4, 3]


@pytest.mark.parametrize("model_id", DEFAULT_TARGETS)
def test_current_trainer_passes(model_id):
    result = cached_run(model_id)
    assert result.status is Status.PASS, result.to_dict()
    assert all(f.evidence_class is EvidenceClass.SEEN for f in result.findings)


def test_injected_mean_of_means_defect_fails():
    assert_mean_of_means_pattern(cached_run(LLAMA_TYPE, trainer_cls=MeanOfMeansTrainer))


def test_injected_position_zero_count_is_a_normalisation_fail():
    result = cached_run(LLAMA_TYPE, trainer_cls=PositionZeroCountTrainer)
    assert result.status is Status.FAIL, result.to_dict()
    verdicts = {(f.evidence["run"], f.evidence["metric"]): f.evidence["verdict"]
                for f in result.findings if "verdict" in f.evidence}
    assert verdicts[("accumulated (4 micro-batches x 1 row)", "loss")] == ABOVE
    assert verdicts[("one batch (1 x 4 rows)", "loss")] == ABOVE
    assert verdicts[("accumulated vs one batch", "loss and update")] == INSIDE
    inferred = [f.claim for f in result.findings if f.evidence_class is EvidenceClass.INFERRED]
    assert inferred == ["The loss normalisation differs from the reference in both runs, so the cause "
                        "is not accumulation itself"]


@pytest.mark.parametrize("model_id", DEFAULT_TARGETS)
def test_current_sft_trainer_does_not_fail(model_id):
    """TRL's float32 log-prob kernel rounds more than the measured floor allows for, so
    its accumulated update may land in the unclear zone (docs/m0_findings.md). It must
    never fail, and anything outside the pass band must be labelled unknown, not seen."""
    result = cached_run(model_id, "trl")
    assert result.status in (Status.PASS, Status.INFO), result.to_dict()
    assert result.observed["trainer"] == "TRL SFTTrainer"
    assert any("TRL's chunked log-prob kernel" in item for item in result.limitations)
    for f in result.findings:
        verdict = f.evidence["verdict"]
        assert verdict != ABOVE
        assert (f.evidence_class is EvidenceClass.SEEN) == (verdict == INSIDE)


def test_injected_defect_fails_on_sft_trainer():
    assert_mean_of_means_pattern(cached_run(LLAMA_TYPE, "trl", MeanOfMeansSFTTrainer))


def test_result_records_environment_and_limits():
    result = cached_run(LLAMA_TYPE)
    assert result.environment["transformers"] is not None
    assert result.environment["trl"] is not None
    assert result.reproducer.startswith("python -m conform.probes.p1_grad_accum")
    assert any("calibrated" in item and "docs/calibration.md" in item for item in result.limitations)
