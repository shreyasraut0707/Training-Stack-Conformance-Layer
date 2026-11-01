import pytest

from conform.probes.p1_multiprocess import run
from conform.proxy_model import LLAMA_TYPE
from conform.result import EvidenceClass, Status


def _run(**kwargs):
    result = run(LLAMA_TYPE, **kwargs)
    if result.status is Status.NOT_TESTED and "OSError" in (result.reason or ""):
        pytest.skip("tokenizer not downloaded; run scripts/download_targets.py")
    return result


def test_two_processes_average_the_loss_correctly():
    result = _run()
    assert result.status is Status.PASS, result.to_dict()
    tokens = result.observed["tokens_per_process"]
    assert len(tokens) == 2 and tokens[0] != tokens[1]  # unequal counts, or the defect could not show


def test_per_process_normalisation_fails():
    result = _run(trainer_args={"average_tokens_across_devices": False})
    assert result.status is Status.FAIL, result.to_dict()
    assert any(f.evidence_class is EvidenceClass.INFERRED for f in result.findings)
