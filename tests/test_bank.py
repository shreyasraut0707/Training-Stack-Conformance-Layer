import importlib
import json

import pytest

from conform.bank import (
    BankRun, extract_result, judge, load_cases, load_controls, run_and_report, run_case, stability,
)


def test_case_files_are_valid():
    cases = load_cases()
    assert {c["id"] for c in cases} == {
        "ga-normalisation-2024", "eos-masked-pad-eq-eos-2025", "eos-masked-pad-eq-eos-2025-behaviour",
        "position-zero-label-count",
    }
    for c in cases:
        # Cases may share an environment (case A's fixed pin is case C's), never within a case.
        assert c["environments"]["defective"]["venv"] != c["environments"]["fixed"]["venv"]
        importlib.import_module(f"conform.probes.{c['probe']}")
        assert c["environments"]["defective"]["expect"] == "fail"
        assert c["environments"]["fixed"]["expect"] == "not_fail"


def test_extract_result_ignores_trainer_logs():
    stdout = "{'loss': '1.0'}\n" + json.dumps({"probe": "P1", "status": "pass"}, indent=2) + "\n"
    assert extract_result(stdout)["status"] == "pass"


def test_judge():
    assert judge("fail", "fail") == "match"
    assert judge("info", "not_fail") == "match"
    assert judge("pass", "fail") == "mismatch"
    assert judge("fail", "not_fail") == "mismatch"
    assert judge("not_tested", "fail") == "not tested"


def test_controls_file_is_valid():
    controls = load_controls()
    assert len(controls["models"]) == 3
    ids = [spec["id"] for spec in controls["runs"]]
    assert len(ids) == len(set(ids))
    for spec in controls["runs"]:
        importlib.import_module(f"conform.probes.{spec['probe']}")
    collator = next(spec for spec in controls["runs"] if spec["id"] == "p2-transformers")
    assert "supervise_eos=true" not in collator["args"]  # documented-only intents (G3 decision)
    assert "p7-transformers" not in ids  # fails by design with PAD = EOS: an observation, not a control
    assert {"p3-trl", "p6-transformers", "p6-trl", "p7-trl"} <= set(ids)


def _run(status, verdicts):
    findings = [{"evidence_class": "seen", "evidence": {"verdict": v, "deviation": 1e-7}}
                for v in verdicts]
    return BankRun("c", "e", "m", "not_fail", status, judge(status, "not_fail"), {}, None,
                   {"status": status, "findings": findings})


def test_stability_flags_any_changed_verdict():
    same = [[_run("pass", ["inside"])] for _ in range(3)]
    assert stability(same)[0]["distinct_signatures"] == 1
    changed = same[:2] + [[_run("info", ["unclear"])]]
    assert stability(changed)[0]["distinct_signatures"] == 2


def _collect_in_turn(statuses):
    """A collect() that returns one run per call with the next status; "crash" raises,
    standing in for an interruption part-way through that repeat."""
    remaining = iter(statuses)

    def collect():
        status = next(remaining)
        if status == "crash":
            raise RuntimeError("simulated interruption")
        return [_run(status, ["inside"])]

    return collect


def _saved_repeats(out_dir):
    return sorted(p.name for p in out_dir.glob("repeat-*"))


def test_each_repeat_is_saved_as_it_finishes(tmp_path):
    assert run_and_report(_collect_in_turn(["pass"] * 3), repeat=3, out_dir=tmp_path) == 0
    assert _saved_repeats(tmp_path) == ["repeat-01", "repeat-02", "repeat-03"]
    assert all((tmp_path / name / "summary.md").exists() for name in _saved_repeats(tmp_path))
    report = (tmp_path / "stability.md").read_text(encoding="utf-8")
    assert "Repeats completed: 3 of 3" in report
    assert "Stable (identical verdicts in every repeat): yes" in report


def test_interruption_keeps_completed_repeats(tmp_path):
    with pytest.raises(RuntimeError):
        run_and_report(_collect_in_turn(["pass", "pass", "crash"]), repeat=5, out_dir=tmp_path)
    assert _saved_repeats(tmp_path) == ["repeat-01", "repeat-02"]
    report = (tmp_path / "stability.md").read_text(encoding="utf-8")
    assert "Repeats completed: 2 of 5" in report
    assert "Stable (identical verdicts in every repeat)" not in report  # never claimed if incomplete


@pytest.mark.parametrize("statuses, saved", [
    (["fail", "pass", "pass"], ["repeat-01"]),               # a run that does not match
    (["pass", "info", "pass"], ["repeat-01", "repeat-02"]),  # a verdict that changed
])
def test_stops_at_first_unexpected_result(tmp_path, statuses, saved):
    assert run_and_report(_collect_in_turn(statuses), repeat=3, out_dir=tmp_path) == 1
    assert _saved_repeats(tmp_path) == saved
    assert f"Repeats completed: {len(saved)} of 3" in (tmp_path / "stability.md").read_text(encoding="utf-8")


def test_missing_environment_is_not_tested(tmp_path):
    case = load_cases({"ga-normalisation-2024"})[0]
    runs = run_case(case, venv_root=tmp_path)
    assert {r.outcome for r in runs} == {"not tested"}
    assert all("not created" in r.reason for r in runs)
