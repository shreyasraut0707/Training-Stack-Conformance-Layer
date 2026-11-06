"""MCP server for coding agents (spec Section 9): a thin layer over the CLI.

Local stdio transport only. Every tool returns a compact summary and the path of the full
report (spec: "keep output small"), keeps unknown findings visible, and includes the exact
equivalent CLI command so an agent in a confirmation mode can show it before running.

Run with:  python -m conform.mcp_server
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from conform.registry import PROBES, SYMPTOMS, available, load_recipe, selected_probes, unavailable_reason

REPORTS_ROOT = Path.cwd() / "conform-reports"
EVIDENCE_CLASSES = ("seen", "inferred", "unknown")
# Spec Section 9: the agent quotes the evidence class and reads every unknown item aloud. Given
# both in the tool descriptions and first in each result, since an agent may not open the skill.
HOW_TO_REPORT = ("Tell the user each probe's status and quote each finding's evidence class (seen, "
                 "inferred or unknown). Read every 'unknown' item to the user. Claim harm only when a "
                 "behavioural probe (P7) confirmed it. Give the user the path of the full report "
                 "(the 'report' field).")


def _compact(result: dict) -> dict:
    findings = result.get("findings", [])
    return {"probe": result["probe"], "status": result["status"], "reason": result.get("reason"),
            "evidence": {c: sum(f["evidence_class"] == c for f in findings) for c in EVIDENCE_CLASSES},
            "seen": [f["claim"] for f in findings if f["evidence_class"] == "seen"][:5],
            "inferred": [f["claim"] for f in findings if f["evidence_class"] == "inferred"],
            "unknown": [f["claim"] for f in findings if f["evidence_class"] == "unknown"],
            "reproducer": result.get("reproducer")}


def _run(recipe_path: str, probe_ids: list[str], label: str, meta: dict) -> dict:
    from conform.cli import run_probes
    from conform.reports import summary, write_reports

    recipe = load_recipe(recipe_path)
    results = run_probes(recipe, probe_ids)
    out = REPORTS_ROOT / datetime.now().strftime(f"{label}_%Y-%m-%d_%H%M%S")
    paths = write_reports(results, out, {**meta, "recipe": recipe_path, "model": recipe["model"]})
    note = f"{HOW_TO_REPORT} Full report: {paths['markdown']}"
    return {"note": note, "summary": summary(results), "results": [_compact(r) for r in results],
            "report": str(paths["markdown"]), "report_json": str(paths["json"])}


def list_probes() -> dict:
    """Probes available for the installed stack, and why any are unavailable."""
    return {"probes": [{"id": pid, "name": p["name"], "release": p["release"], "question": p["question"],
                        "available": available(pid), "why_unavailable": unavailable_reason(pid)}
                       for pid, p in PROBES.items()],
            "symptoms": SYMPTOMS}


def run_probe(probe_id: str, recipe_path: str) -> dict:
    """Run one probe against a recipe file and its sample rows."""
    if probe_id not in PROBES:
        return {"error": f"unknown probe {probe_id!r}; choose from {sorted(PROBES)}"}
    result = _run(recipe_path, [probe_id], f"probe-{probe_id}", {"command": "run_probe"})
    return {"note": result.pop("note"), "command": f"conform check --recipe {recipe_path} --probes {probe_id}",
            **result}


def run_all(recipe_path: str) -> dict:
    """Run every available probe for a recipe; returns a summary and the report path."""
    probe_ids = selected_probes(load_recipe(recipe_path))
    result = _run(recipe_path, probe_ids, "check", {"command": "run_all"})
    return {"note": result.pop("note"), "command": f"conform check --recipe {recipe_path}", **result}


def diagnose(symptom: str, recipe_path: str) -> dict:
    """Run the probes most related to a symptom (no-stop, worse-than-base), most direct first."""
    if symptom not in SYMPTOMS:
        return {"error": f"unknown symptom {symptom!r}; choose from {sorted(SYMPTOMS)}"}
    result = _run(recipe_path, SYMPTOMS[symptom], f"diagnose-{symptom}", {"command": "diagnose", "symptom": symptom})
    return {"note": result.pop("note"), "command": f"conform diagnose --symptom {symptom} --recipe {recipe_path}",
            "probes_in_order": SYMPTOMS[symptom], **result}


def diff_environments(from_env: str, to_env: str, recipe_path: str) -> dict:
    """Compare verdicts between two pinned environments; returns changed verdicts only."""
    import contextlib
    import io
    import json

    from conform.cli import main

    out = REPORTS_ROOT / datetime.now().strftime("diff_%Y-%m-%d_%H%M%S")
    with contextlib.redirect_stdout(io.StringIO()):
        code = main(["diff", "--from", from_env, "--to", to_env, "--recipe", recipe_path, "--out", str(out)])
    if code == 2 or not (out / "diff.json").exists():
        return {"error": "diff could not run (check the environment names and the recipe)"}
    changed = json.loads((out / "diff.json").read_text(encoding="utf-8"))["changed"]
    return {"command": f"conform diff --from {from_env} --to {to_env} --recipe {recipe_path}",
            "changed": changed, "details": str(out / "diff.json")}


TOOLS = (list_probes, run_probe, run_all, diagnose, diff_environments)
for _tool in (run_probe, run_all, diagnose):  # the tool description is what every agent sees
    _tool.__doc__ = f"{_tool.__doc__} {HOW_TO_REPORT}"


def build_server():
    try:  # mcp 2.x renamed FastMCP to MCPServer
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server

    server = Server("conform")
    for tool in TOOLS:
        server.tool()(tool)
    return server


def main() -> None:
    build_server().run()  # stdio transport (spec Section 12.2: stdio only)


if __name__ == "__main__":
    main()
