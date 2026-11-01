import json
import xml.etree.ElementTree as ET

import pytest

from conform.cli import main
from conform.proxy_model import LLAMA_TYPE, build_proxy


def _recipe(tmp_path, **fields):
    path = tmp_path / "recipe.yaml"
    path.write_text(json.dumps({"model": LLAMA_TYPE, **fields}), encoding="utf-8")  # JSON is valid YAML
    return str(path)


@pytest.fixture(scope="module", autouse=True)
def tokenizer_available():
    try:
        build_proxy(LLAMA_TYPE)
    except OSError:
        pytest.skip("tokenizer not downloaded; run scripts/download_targets.py")


def test_list_probes(capsys):
    assert main(["list-probes"]) == 0
    out = capsys.readouterr().out
    assert "P7  EOS stopping [v0.1] - available" in out
    assert "P4  Packing isolation [v0.2] - available" in out
    assert "P1M  Multi-process loss averaging [v0.2] - available" in out


def test_probes_need_the_installed_stack(monkeypatch, capsys):
    import conform.registry as registry

    monkeypatch.setattr(registry, "_installed", lambda package: package not in ("torch", "trl"))
    assert registry.missing("P3") == ["torch", "trl"]
    assert main(["list-probes"]) == 0
    assert "P1  Gradient-accumulation equivalence [v0.1] - not available: the installed stack lacks torch" \
        in capsys.readouterr().out


@pytest.mark.parametrize("fields", [{"model": None}, {"colour": "red"}, {"trainer": "keras"},
                                    {"probes": ["P9"]}])
def test_bad_recipe_is_a_usage_error(tmp_path, fields, capsys):
    path = tmp_path / "bad.yaml"
    recipe = {"model": LLAMA_TYPE, **fields}
    if recipe["model"] is None:
        del recipe["model"]
    path.write_text(json.dumps(recipe), encoding="utf-8")
    assert main(["check", "--recipe", str(path), "--out", str(tmp_path / "out")]) == 2
    assert "conform:" in capsys.readouterr().err


def test_unknown_command_is_a_usage_error():
    assert main(["frobnicate"]) == 2


def test_diagnose_dry_run_lists_related_probes(tmp_path, capsys):
    assert main(["diagnose", "--symptom", "no-stop", "--recipe", _recipe(tmp_path), "--dry-run"]) == 0
    assert "P7 P2 P3" in capsys.readouterr().out


def test_unavailable_probe_is_not_tested_and_strict_counts_it(tmp_path, monkeypatch):
    from conform.registry import PROBES

    monkeypatch.setitem(PROBES, "PX", {"module": None, "name": "Placeholder", "release": "later",
                                       "question": "?", "uses_rows": False, "unavailable": "not built yet"})
    recipe = _recipe(tmp_path, probes=["PX"])
    assert main(["check", "--recipe", recipe, "--out", str(tmp_path / "a")]) == 0
    assert main(["check", "--recipe", recipe, "--out", str(tmp_path / "b"), "--strict"]) == 1
    report = json.loads((tmp_path / "a" / "report.json").read_text(encoding="utf-8"))
    assert report["results"][0]["status"] == "not_tested"


def test_check_runs_probes_and_writes_three_reports(tmp_path):
    recipe = _recipe(tmp_path, trainer="transformers", probes=["P6", "P3"],
                     rows=[{"prompt": "Hi?", "completion": "Hello."}, {"prompt": "Yes?", "completion": "Yes."}])
    out = tmp_path / "out"
    assert main(["check", "--recipe", recipe, "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    statuses = {r["probe"]: r["status"] for r in report["results"]}
    assert statuses == {"P6-precision": "info", "P3-train-inference-tokens": "not_tested"}
    assert "--rows" in report["results"][0]["reproducer"]  # the recipe's own rows were used
    suite = ET.parse(out / "report.xml").getroot()
    assert (suite.get("tests"), suite.get("skipped")) == ("2", "1")
    assert (out / "report.md").exists()


def test_diff_same_environment_changes_nothing(tmp_path, capsys):
    import sys

    recipe = _recipe(tmp_path, probes=["P6"])
    # The running interpreter, so the test works on any computer (CI has no ~/.venvs/conform).
    assert main(["diff", "--from", sys.executable, "--to", sys.executable, "--recipe", recipe,
                 "--out", str(tmp_path / "diff")]) == 0
    assert "No probe changed verdict" in capsys.readouterr().out
