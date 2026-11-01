import pytest

from conform.mutations import MUTANTS, PROBE_MODULES, run_mutant
from conform.proxy_model import LLAMA_TYPE, build_proxy
from conform.registry import PROBES


@pytest.fixture(scope="module", autouse=True)
def tokenizer_available():
    try:
        build_proxy(LLAMA_TYPE)
    except OSError:
        pytest.skip("tokenizer not downloaded; run scripts/download_targets.py")


def _mutant(mid, probe):
    return next(m for m in MUTANTS if m[0] == mid and m[1] == probe)


def test_every_mutant_targets_a_real_probe_and_trainer():
    for mid, probe, trainers, factory, kwargs, expect in MUTANTS:
        assert probe in PROBES and probe in PROBE_MODULES, mid
        assert set(trainers) <= {"transformers", "trl"} and expect in ("fail", "report"), mid
    covered = {m[1] for m in MUTANTS}
    assert covered == {"P1", "P2", "P3", "P7"}  # P6 never fails, so it has no mutants


@pytest.mark.parametrize("mid, probe, trainer", [
    ("loss-scale-1e-3", "P1", "transformers"),
    ("unmask-prompt", "P2", "trl"),
    ("double-bos", "P3", "trl"),
    ("change-turn-boundary", "P3", "trl"),
])
def test_mutant_is_caught(mid, probe, trainer):
    row = run_mutant(_mutant(mid, probe), LLAMA_TYPE, trainer)
    assert row["caught"], row


def test_extra_trailing_eos_is_reported_not_failed():
    row = run_mutant(_mutant("extra-trailing-eos", "P3"), LLAMA_TYPE, "trl")
    assert row["caught"] and row["status"] != "fail", row
