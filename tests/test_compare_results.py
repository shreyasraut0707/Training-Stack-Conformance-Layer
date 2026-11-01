import importlib.util
import json
import shutil

from conform.bank import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location("compare_results", PROJECT_ROOT / "scripts" / "compare_results.py")
compare_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_results)


def _write(folder, name, status, verdict="inside", deviation=1e-7):
    folder.mkdir(parents=True, exist_ok=True)
    result = {"status": status, "findings": [{"evidence_class": "seen", "claim": "x",
                                              "evidence": {"verdict": verdict, "deviation": deviation}}]}
    (folder / name).write_text(json.dumps(result), encoding="utf-8")


def test_small_numeric_differences_are_allowed(tmp_path):
    _write(tmp_path / "a", "r.json", "pass", deviation=1e-7)
    _write(tmp_path / "b", "r.json", "pass", deviation=1.3e-7)
    assert compare_results.compare(tmp_path / "a", tmp_path / "b")[1]


def test_a_changed_verdict_or_missing_run_is_reported(tmp_path):
    _write(tmp_path / "a", "r.json", "pass")
    _write(tmp_path / "b", "r.json", "info", verdict="unclear")
    lines, ok = compare_results.compare(tmp_path / "a", tmp_path / "b")
    assert not ok and lines[0].startswith("DIFFERENT")
    _write(tmp_path / "c", "other.json", "pass")
    assert not compare_results.compare(tmp_path / "a", tmp_path / "c")[1]


def test_repeat_folders_are_compared_by_their_first_repeat(tmp_path):
    _write(tmp_path / "a" / "repeat-01", "r.json", "pass")
    _write(tmp_path / "b", "r.json", "pass")
    assert compare_results.compare(tmp_path / "a", tmp_path / "b")[1]


def test_the_real_g4_repeats_match_each_other(tmp_path):
    folder = PROJECT_ROOT / "bank" / "results" / "controls_2026-11-01_091556"
    if not folder.exists():
        return
    shutil.copytree(folder / "repeat-02", tmp_path / "second")
    assert compare_results.compare(folder, tmp_path / "second")[1]
