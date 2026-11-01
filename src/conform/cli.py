"""The `conform` command line (spec Sections 5, 6 and 8).

    conform list-probes
    conform check --recipe sft.yaml [--probes P2 P7] [--out DIR] [--strict] [--download]
    conform diagnose --symptom no-stop --recipe sft.yaml [--dry-run]
    conform diff --from ENV --to ENV --recipe sft.yaml
    conform bank run [CASE ...] [--repeat N]

Each probe runs in its own time-limited process, offline. Exit codes: 0 when no probe
fails, 1 when any probe fails (with --strict, also when any is not tested), 2 for a usage
or internal error.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from conform import __version__
from conform.bank import (
    DEFAULT_VENV_ROOT, PROJECT_ROOT, load_cases, run_and_report, run_case, run_probe, venv_python,
)
from conform.registry import (
    PROBES, SYMPTOMS, RecipeError, available, load_recipe, probe_args, selected_probes, unavailable_reason,
)
from conform.reports import not_tested, summary, write_reports
from conform.reports import exit_code as report_exit_code

USAGE_ERROR = 2


def _rows_file(recipe: dict, workdir: Path) -> str | None:
    if recipe.get("rows_file"):
        return recipe["rows_file"]
    if recipe.get("rows"):
        path = workdir / "rows.json"
        path.write_text(json.dumps(recipe["rows"]), encoding="utf-8")
        return str(path)
    return None


def run_probes(recipe: dict, probe_ids: list[str], *, python=sys.executable, timeout=None) -> list[dict]:
    """Run each probe in its own process and collect its result dictionary."""
    timeout = int(timeout or recipe.get("timeout", 900))
    results = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        rows = _rows_file(recipe, Path(work))
        for pid in probe_ids:
            info = PROBES[pid]
            if not available(pid):
                results.append(not_tested(pid, f"{info['name']}: {unavailable_reason(pid)}"))
                continue
            args = probe_args(pid, recipe, rows)
            result, reason = run_probe(Path(python), info["module"], recipe["model"], args, timeout=timeout)
            results.append(result if result is not None else not_tested(
                pid, reason, f"python -m conform.probes.{info['module']} {recipe['model']} {' '.join(args)}"))
    return results


def _out_dir(args, label: str) -> Path:
    if args.out:
        return Path(args.out)
    return Path.cwd() / "conform-reports" / datetime.now().strftime(f"{label}_%Y-%m-%d_%H%M%S")


def _finish(results, args, meta, label) -> int:
    paths = write_reports(results, _out_dir(args, label), meta, strict=args.strict)
    print(summary(results))
    for r in results:
        line = f"  {r['probe']}: {r['status']}"
        if r.get("reason"):
            line += f" ({r['reason']})"
        print(line)
    print(f"Reports: {paths['markdown']}  {paths['json']}  {paths['junit']}")
    return report_exit_code(results, strict=args.strict)


def _download(recipe) -> None:
    from conform.proxy_model import download_target

    print(f"Downloading config and tokenizer files (no weights) for {recipe['model']} ...")
    download_target(recipe["model"])


def cmd_list_probes(args) -> int:
    for pid, info in PROBES.items():
        state = "available" if available(pid) else f"not available: {unavailable_reason(pid)}"
        print(f"{pid}  {info['name']} [{info['release']}] - {state}\n    {info['question']}")
    print("Symptoms for 'diagnose': " + ", ".join(f"{s} -> {' '.join(p)}" for s, p in SYMPTOMS.items()))
    return 0


def cmd_check(args) -> int:
    recipe = load_recipe(args.recipe)
    probe_ids = selected_probes(recipe, args.probes)
    if args.download:
        _download(recipe)
    results = run_probes(recipe, probe_ids, timeout=args.timeout)
    meta = {"command": "check", "recipe": str(args.recipe), "model": recipe["model"],
            "trainer": recipe.get("trainer", "trl"), "conform": __version__}
    return _finish(results, args, meta, "check")


def cmd_diagnose(args) -> int:
    recipe = load_recipe(args.recipe)
    probe_ids = SYMPTOMS[args.symptom]
    print(f"Symptom '{args.symptom}': running the most related probes, most direct first: {' '.join(probe_ids)}")
    if args.dry_run:
        return 0
    if args.download:
        _download(recipe)
    results = run_probes(recipe, probe_ids, timeout=args.timeout)
    meta = {"command": "diagnose", "symptom": args.symptom, "recipe": str(args.recipe),
            "model": recipe["model"], "conform": __version__}
    return _finish(results, args, meta, f"diagnose-{args.symptom}")


def verdict_signature(result: dict) -> tuple:
    """Status plus each finding's evidence class and verdict: what `diff` compares."""
    return (result["status"], *((f["evidence_class"], f.get("evidence", {}).get("verdict"))
                                for f in result.get("findings", [])))


def _python_for(env: str) -> Path:
    path = Path(env)
    return path if path.suffix == ".exe" or path.name.startswith("python") else venv_python(DEFAULT_VENV_ROOT, env)


def cmd_diff(args) -> int:
    recipe = load_recipe(args.recipe)
    probe_ids = selected_probes(recipe, args.probes)
    sides = {}
    for label, env in (("from", args.from_env), ("to", args.to_env)):
        python = _python_for(env)
        if not python.exists():
            raise RecipeError(f"environment {env!r} not found (looked for {python})")
        sides[label] = run_probes(recipe, probe_ids, python=python, timeout=args.timeout)
    changed = []
    for before, after in zip(sides["from"], sides["to"]):
        if verdict_signature(before) != verdict_signature(after):
            changed.append({"probe": before["probe"], "from": before["status"], "to": after["status"],
                            "from_versions": {k: before.get("environment", {}).get(k) for k in ("transformers", "trl", "torch")},
                            "to_versions": {k: after.get("environment", {}).get(k) for k in ("transformers", "trl", "torch")}})
    out = _out_dir(args, "diff")
    out.mkdir(parents=True, exist_ok=True)
    (out / "diff.json").write_text(json.dumps({"from": args.from_env, "to": args.to_env, "changed": changed,
                                               "results": sides}, indent=2), encoding="utf-8")
    if not changed:
        print(f"No probe changed verdict between {args.from_env} and {args.to_env} ({len(probe_ids)} probes).")
    for c in changed:
        print(f"{c['probe']}: {c['from']} -> {c['to']}   ({c['from_versions']} -> {c['to_versions']})")
    print(f"Details: {out / 'diff.json'}")
    return 1 if changed else 0


def cmd_bank_run(args) -> int:
    cases = load_cases(set(args.cases) if args.cases else None)
    if not cases:
        raise RecipeError("no matching bank cases")
    out = PROJECT_ROOT / "bank" / "results" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return run_and_report(lambda: [r for c in cases for r in run_case(c, timeout=args.timeout)],
                          repeat=args.repeat, out_dir=out)


def cmd_setup(args) -> int:
    from conform.setup_agents import apply, describe, plan

    python = args.python or sys.executable
    changes = plan(args.agent, python=python, home=args.home, project=args.project)
    print(f"Proposed changes for {args.agent}:")
    print(describe(changes, python))
    if not args.apply:
        print("Nothing was changed. Review the plan above, then re-run with --apply to write it "
              "(existing files are backed up first).")
        return 0
    backups = apply(changes)
    print("Applied." + (" Backups: " + ", ".join(str(b) for b in backups) if backups else " No files needed backups."))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conform", description="Executable conformance probes for "
                                     "fine-tuning stacks (TRL SFTTrainer, Transformers Trainer).")
    parser.add_argument("--version", action="version", version=f"conform {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-probes", help="list the probes and their availability").set_defaults(func=cmd_list_probes)

    def reporting(p):
        p.add_argument("--recipe", required=True, help="recipe file (YAML)")
        p.add_argument("--out", help="directory for report.json, report.md and report.xml")
        p.add_argument("--strict", action="store_true", help="count not_tested as a failure")
        p.add_argument("--timeout", type=int, help="seconds per probe (default: recipe or 900)")
        p.add_argument("--download", action="store_true",
                       help="download the model's config and tokenizer files first (never weights)")

    check = sub.add_parser("check", help="run probes for a recipe")
    reporting(check)
    check.add_argument("--probes", nargs="+", metavar="PROBE", help="probe ids (default: recipe or all available)")
    check.set_defaults(func=cmd_check)

    diagnose = sub.add_parser("diagnose", help="run the probes most related to a symptom")
    reporting(diagnose)
    diagnose.add_argument("--symptom", required=True, choices=sorted(SYMPTOMS))
    diagnose.add_argument("--dry-run", action="store_true", help="only print which probes would run")
    diagnose.set_defaults(func=cmd_diagnose)

    diff = sub.add_parser("diff", help="probes whose verdict changes between two environments")
    diff.add_argument("--from", dest="from_env", required=True, help="environment name in ~/.venvs or a python path")
    diff.add_argument("--to", dest="to_env", required=True, help="environment name in ~/.venvs or a python path")
    diff.add_argument("--recipe", required=True)
    diff.add_argument("--probes", nargs="+", metavar="PROBE")
    diff.add_argument("--out")
    diff.add_argument("--timeout", type=int)
    diff.set_defaults(func=cmd_diff)

    setup = sub.add_parser("setup", help="register the MCP server and skill for a coding agent")
    setup.add_argument("--agent", required=True, choices=["claude-code", "codex"])
    setup.add_argument("--apply", action="store_true", help="write the changes (default: only show them)")
    setup.add_argument("--python", help="interpreter that has conform and mcp installed (default: this one)")
    setup.add_argument("--project", type=Path, help="project folder for Claude Code (default: current folder)")
    setup.add_argument("--home", type=Path, help="home folder for Codex (default: your home folder)")
    setup.set_defaults(func=cmd_setup)

    bank = sub.add_parser("bank", help="regression bank")
    bank_sub = bank.add_subparsers(dest="bank_command", required=True)
    bank_run = bank_sub.add_parser("run", help="run bank cases in their pinned environments")
    bank_run.add_argument("cases", nargs="*")
    bank_run.add_argument("--repeat", type=int, default=1)
    bank_run.add_argument("--timeout", type=int, default=900)
    bank_run.set_defaults(func=cmd_bank_run)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse already printed the usage error
        return USAGE_ERROR if exc.code else 0
    try:
        return args.func(args)
    except RecipeError as exc:
        print(f"conform: {exc}", file=sys.stderr)
        return USAGE_ERROR
    except Exception as exc:  # internal error: exit code 2, never a silent pass
        print(f"conform: internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return USAGE_ERROR


if __name__ == "__main__":
    sys.exit(main())
