import pytest

from conform.result import (
    EvidenceClass, Finding, ProbeResult, Status, exit_code, summarize,
)

ENV = {"python": "3.11.9", "device": "cpu", "dtype": "float32"}


def make(status, findings=(), reason=None):
    return ProbeResult(
        probe="P0-test", status=status, oracle="test", environment=ENV,
        reproducer="none", findings=list(findings), reason=reason,
    )


def test_fail_needs_a_seen_finding():
    inferred = Finding(EvidenceClass.INFERRED, "may not stop", rule="no EOS gradient")
    with pytest.raises(ValueError):
        make(Status.FAIL, [inferred])
    make(Status.FAIL, [Finding(EvidenceClass.SEEN, "EOS label is -100 in 8 of 8 rows")])


def test_inferred_needs_a_rule():
    with pytest.raises(ValueError):
        Finding(EvidenceClass.INFERRED, "may not stop")


def test_not_tested_needs_a_reason():
    with pytest.raises(ValueError):
        make(Status.NOT_TESTED)


def test_summary_has_no_overall_pass():
    seen = Finding(EvidenceClass.SEEN, "x")
    results = [make(Status.PASS), make(Status.FAIL, [seen]),
               make(Status.NOT_TESTED, reason="no GPU")]
    assert summarize(results) == "1 pass, 1 fail, 1 not tested"


def test_exit_codes():
    passing = [make(Status.PASS), make(Status.NOT_TESTED, reason="no GPU")]
    assert exit_code(passing) == 0
    assert exit_code(passing, strict=True) == 1


def test_to_dict_uses_plain_strings():
    d = make(Status.PASS, [Finding(EvidenceClass.SEEN, "x")]).to_dict()
    assert d["status"] == "pass"
    assert d["findings"][0]["evidence_class"] == "seen"
