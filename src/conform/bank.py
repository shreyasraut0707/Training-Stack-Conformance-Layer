"""Regression bank (spec Section 10): pinned environments where a documented defect
exists and where it is fixed. Each case runs its probe inside each environment as a
separate, time-limited process, so old library versions never mix with the current
ones. Environments use the project's src folder via PYTHONPATH, so they hold only the
pinned libraries under test.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = PROJECT_ROOT / "bank" / "cases"
DEFAULT_VENV_ROOT = Path.home() / ".venvs"
EXPECTATIONS = ("fail", "not_fail")
REQUIRED_KEYS = ("id", "title", "confidence", "sources", "probe", "models", "args", "environments")
VERSION_KEYS = ("torch", "transformers", "trl", "accelerate", "datasets")


def load_cases(case_ids=None) -> list[dict]:
    """Load and validate bank/cases/*.yaml, optionally only the given ids."""
    cases = []
    for path in sorted(CASES_DIR.glob("*.yaml")):
        case = yaml.safe_load(path.read_text(encoding="utf-8"))
        missing = [k for k in REQUIRED_KEYS if k not in case]
        if missing:
            raise ValueError(f"{path.name}: missing keys {missing}")
        if set(case["environments"]) != {"defective", "fixed"}:
            raise ValueError(f"{path.name}: needs exactly a 'defective' and a 'fixed' environment")
        for name, env in case["environments"].items():
            if env.get("expect") not in EXPECTATIONS:
                raise ValueError(f"{path.name}: {name}.expect must be one of {EXPECTATIONS}")
        if case_ids is None or case["id"] in case_ids:
            cases.append(case)
    return cases


def venv_python(venv_root: Path, venv: str) -> Path:
    return Path(venv_root) / venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def extract_result(stdout: str) -> dict:
    """The probe's JSON result: the last printed object that starts with "probe".
    Trainers also print log lines to stdout, so the rest is ignored."""
    start = stdout.rfind('{\n  "probe": ')
    if start < 0:
        raise ValueError("no probe result found in output")
    return json.loads(stdout[start:])


def judge(status: str, expect: str) -> str:
    """'match', 'mismatch' or 'not tested' for one probe status against the expectation."""
    if status == "not_tested":
        return "not tested"
    met = status == "fail" if expect == "fail" else status in ("pass", "info")
    return "match" if met else "mismatch"


@dataclass
class BankRun:
    case: str
    environment: str
    model: str
    expect: str
    status: str
    outcome: str
    versions: dict
    reason: str | None
    result: dict | None


def _child_env() -> dict:
    return {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src"), "PYTHONIOENCODING": "utf-8",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def run_probe(python: Path, probe: str, model: str, args: list[str], *, timeout: int):
    """Run one probe with `python` in its own time-limited process.
    Returns (result, reason); result is None when the probe gave no result."""
    python = Path(python)
    if not python.exists():
        return None, f"environment {python.parents[1].name} not created"
    command = [str(python), "-m", f"conform.probes.{probe}", model, *args]
    try:
        done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, cwd=PROJECT_ROOT, env=_child_env())
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s"
    try:
        return extract_result(done.stdout), None
    except ValueError:
        last = (done.stderr.strip().splitlines() or ["no error output"])[-1]
        known = WINDOWS_EXIT_CODES.get(done.returncode % 2**32)
        return None, f"probe crashed (exit {done.returncode}{': ' + known if known else ''}): {last}"


# Windows status codes seen when a process is killed rather than exiting normally.
WINDOWS_EXIT_CODES = {
    0xC000012D: "out of virtual memory (Windows commit limit)",
    0xC0000017: "out of memory",
    0x40010004: "terminated by the system",
    0xC000013A: "interrupted (Ctrl+C)",
}


def _bank_run(case, environment, model, expect, result, reason) -> BankRun:
    if result is None:
        return BankRun(case, environment, model, expect, status="not_tested", outcome="not tested",
                       versions={}, reason=reason, result=None)
    return BankRun(case, environment, model, expect, status=result["status"],
                   outcome=judge(result["status"], expect),
                   versions={k: result["environment"].get(k) for k in VERSION_KEYS},
                   reason=result.get("reason"), result=result)


def run_case(case: dict, *, venv_root=DEFAULT_VENV_ROOT, timeout: int = 900) -> list[BankRun]:
    """Run the case's probe for every model in both environments."""
    runs = []
    for env_name, env in case["environments"].items():
        python = venv_python(venv_root, env["venv"])
        for model in case["models"]:
            result, reason = run_probe(python, case["probe"], model, case["args"], timeout=timeout)
            runs.append(_bank_run(case["id"], env_name, model, env["expect"], result, reason))
    return runs


CONTROLS_FILE = PROJECT_ROOT / "bank" / "controls.yaml"


def load_controls() -> dict:
    controls = yaml.safe_load(CONTROLS_FILE.read_text(encoding="utf-8"))
    for key in ("venv", "models", "runs"):
        if key not in controls:
            raise ValueError(f"controls.yaml: missing key {key!r}")
    return controls


def run_controls(controls: dict, *, venv_root=DEFAULT_VENV_ROOT, timeout: int = 900,
                 python=None, only=None) -> list[BankRun]:
    """G3: every control run on current releases must not fail. `python` overrides the
    environment named in controls.yaml (for example a CI runner's interpreter); `only`
    limits the run to the given control ids."""
    python = Path(python) if python else venv_python(venv_root, controls["venv"])
    unknown = set(only or ()) - {spec["id"] for spec in controls["runs"]}
    if unknown:
        raise ValueError(f"unknown control ids {sorted(unknown)}")
    runs = []
    for spec in controls["runs"]:
        if only and spec["id"] not in only:
            continue
        for model in controls["models"]:
            result, reason = run_probe(python, spec["probe"], model, spec["args"], timeout=timeout)
            runs.append(_bank_run("controls", spec["id"], model, "not_fail", result, reason))
    return runs


def write_report(runs: list[BankRun], out_dir: Path) -> Path:
    """Save each probe's full JSON result and a summary.md table; return the summary path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in runs:
        if r.result is not None:
            name = f"{r.case}__{r.environment}__{r.model.replace('/', '_')}.json"
            (out_dir / name).write_text(json.dumps(r.result, indent=2), encoding="utf-8")
    lines = ["| Case | Environment | Model | transformers | trl | torch | Status | Expected | Outcome |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        v = r.versions
        lines.append(f"| {r.case} | {r.environment} | {r.model} | {v.get('transformers') or '–'} | "
                     f"{v.get('trl') or '–'} | {v.get('torch') or '–'} | {r.status} | {r.expect} | "
                     f"{r.outcome} |")
    notes = [f"- {r.case} / {r.environment} / {r.model}: {r.reason}" for r in runs if r.reason]
    if notes:
        lines += ["", "Reasons:", *notes]
    summary = out_dir / "summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def signature(run: BankRun) -> tuple:
    """What must be identical across repeats: the status, and each finding's class and verdict."""
    if run.result is None:
        return (run.status,)
    return (run.status, *((f["evidence_class"], f["evidence"].get("verdict"))
                          for f in run.result["findings"]))


def _deviations(run: BankRun) -> list[float]:
    if run.result is None:
        return []
    return [f["evidence"]["deviation"] for f in run.result["findings"] if "deviation" in f["evidence"]]


def stability(repeats: list[list[BankRun]]) -> list[dict]:
    """For each run position: how many distinct verdict signatures the repeats gave (1 = stable)."""
    rows = []
    for same_runs in zip(*repeats):
        first = same_runs[0]
        deviations = [_deviations(r) for r in same_runs]
        spread = (max(max(col) - min(col) for col in zip(*deviations))
                  if all(deviations) else None)
        rows.append({"case": first.case, "environment": first.environment, "model": first.model,
                     "repeats": len(same_runs),
                     "distinct_signatures": len({signature(r) for r in same_runs}),
                     "statuses": sorted({r.status for r in same_runs}),
                     "max_deviation_spread": spread})
    return rows


def write_stability_report(rows: list[dict], out_dir: Path, *, completed: int, planned: int) -> Path:
    """Write stability.md. A stability verdict appears only once all planned repeats are done."""
    out_dir.mkdir(parents=True, exist_ok=True)
    header = f"Repeats completed: {completed} of {planned}"
    if completed < planned:
        header += " (in progress, interrupted or stopped early: no stability verdict)"
    lines = [header, "",
             "| Case | Environment | Model | Repeats | Distinct verdicts | Statuses | Max deviation spread |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        spread = "–" if r["max_deviation_spread"] is None else f"{r['max_deviation_spread']:.3g}"
        lines.append(f"| {r['case']} | {r['environment']} | {r['model']} | {r['repeats']} | "
                     f"{r['distinct_signatures']} | {', '.join(r['statuses'])} | {spread} |")
    same = all(r["distinct_signatures"] == 1 for r in rows)
    if completed == planned:
        lines += ["", f"Stable (identical verdicts in every repeat): {'yes' if same else 'NO'}"]
    else:
        lines += ["", f"Identical verdicts in the {completed} completed repeats: {'yes' if same else 'NO'}"]
    path = out_dir / "stability.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_and_report(collect, *, repeat: int, out_dir: Path) -> int:
    """Run `collect` `repeat` times. With repeats, each repeat is saved (repeat-NN/) as soon as
    it finishes and stability.md is rewritten after every repeat, so an interruption keeps
    every completed repeat. The run stops at the first repeat with a run that does not match
    its expectation or a verdict that differs from an earlier repeat. Exit code 0 only if
    every run matches and every verdict is identical across all the repeats."""
    if repeat == 1:
        runs = collect()
        print(write_report(runs, out_dir).read_text(encoding="utf-8"))
        return 0 if all(r.outcome == "match" for r in runs) else 1
    repeats = []
    for i in range(1, repeat + 1):
        runs = collect()
        repeats.append(runs)
        write_report(runs, out_dir / f"repeat-{i:02d}")
        rows = stability(repeats)
        path = write_stability_report(rows, out_dir, completed=i, planned=repeat)
        mismatched = [r for r in runs if r.outcome != "match"]
        changed = [row for row in rows if row["distinct_signatures"] > 1]
        print(f"repeat {i}/{repeat} saved: {len(runs) - len(mismatched)}/{len(runs)} runs match, "
              f"{len(changed)} changed verdicts", flush=True)
        if mismatched or changed:
            for r in mismatched:
                print(f"  STOP: {r.case} / {r.environment} / {r.model}: {r.status} "
                      f"(expected {r.expect}): {r.reason or ''}", flush=True)
            for row in changed:
                print(f"  STOP: verdict changed: {row['case']} / {row['environment']} / {row['model']} "
                      f"(statuses seen: {', '.join(row['statuses'])})", flush=True)
            print(path.read_text(encoding="utf-8"))
            return 1
    print(path.read_text(encoding="utf-8"))
    return 0
