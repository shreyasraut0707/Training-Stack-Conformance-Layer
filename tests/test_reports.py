import json
import xml.etree.ElementTree as ET

from conform.reports import exit_code, not_tested, summary, to_junit, to_markdown, write_reports


def _result(probe, status, findings=()):
    return {"probe": probe, "status": status, "findings": list(findings), "environment": {"torch": "x"},
            "reproducer": f"python -m conform.probes.{probe}", "limitations": ["proxy model"], "oracle": "o",
            "observed": {}, "reason": None}


SEEN = {"evidence_class": "seen", "claim": "EOS carries loss in 0 of 4 rows", "evidence": {}}
INFERRED = {"evidence_class": "inferred", "claim": "may not stop", "rule": "no gradient", "evidence": {}}
RESULTS = [_result("P1", "pass"), _result("P2", "fail", [SEEN, INFERRED]), _result("P6", "info"),
           not_tested("P4", "arrives in v0.2")]


def test_summary_has_no_overall_pass_label():
    assert summary(RESULTS) == "1 pass, 1 fail, 1 info, 1 not tested"


def test_exit_codes():
    assert exit_code(RESULTS) == 1
    clean = [r for r in RESULTS if r["status"] != "fail"]
    assert exit_code(clean) == 0
    assert exit_code(clean, strict=True) == 1


def test_markdown_separates_evidence_classes():
    md = to_markdown(RESULTS, {"command": "check"})
    assert "## P2: fail" in md and "**seen**" in md and "**inferred**" in md
    assert "(rule: no gradient)" in md
    assert "Reason: arrives in v0.2" in md


def test_junit_counts_and_strict():
    suite = ET.fromstring(to_junit(RESULTS, {}))
    assert (suite.get("tests"), suite.get("failures"), suite.get("skipped")) == ("4", "1", "1")
    strict = ET.fromstring(to_junit(RESULTS, {}, strict=True))
    assert (strict.get("failures"), strict.get("skipped")) == ("2", "0")


def test_write_reports_creates_all_three(tmp_path):
    paths = write_reports(RESULTS, tmp_path, {"command": "check"})
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["summary"].startswith("1 pass")
    assert paths["markdown"].read_text(encoding="utf-8").startswith("# Conformance report")
    ET.parse(paths["junit"])
