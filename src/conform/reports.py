"""JSON for agents, Markdown for people, JUnit XML for CI, all from one result list (spec Section 8).

Results are probe result dictionaries (`ProbeResult.to_dict()`). There is deliberately no
overall pass label: the summary reads like "4 pass, 1 fail, 2 not tested".
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

STATUS_ORDER = ("pass", "fail", "info", "not_tested")
CLASS_ORDER = ("seen", "inferred", "unknown")


def summary(results: list[dict]) -> str:
    counts = Counter(r["status"] for r in results)
    parts = [f"{counts[s]} {s.replace('_', ' ')}" for s in STATUS_ORDER if counts[s]]
    return ", ".join(parts) or "no probes run"


def exit_code(results: list[dict], strict: bool = False) -> int:
    statuses = {r["status"] for r in results}
    return 1 if "fail" in statuses or (strict and "not_tested" in statuses) else 0


def not_tested(probe: str, reason: str, reproducer: str = "") -> dict:
    """A result for a probe that produced none (crash, timeout, unavailable)."""
    return {"probe": probe, "status": "not_tested", "reason": reason, "findings": [], "environment": {},
            "reproducer": reproducer, "limitations": [], "oracle": "", "observed": {}}


def to_json(results: list[dict], meta: dict) -> str:
    return json.dumps({"summary": summary(results), "meta": meta, "results": results}, indent=2)


def to_markdown(results: list[dict], meta: dict) -> str:
    lines = [f"# Conformance report", "", f"**{summary(results)}** (no overall pass label by design)", ""]
    for key, value in meta.items():
        lines.append(f"- {key}: {value}")
    env = next((r["environment"] for r in results if r.get("environment")), {})
    if env:
        lines += ["", "| Environment | |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in env.items() if k != "seeds"]
    for r in results:
        lines += ["", f"## {r['probe']}: {r['status']}", ""]
        if r.get("reason"):
            lines.append(f"Reason: {r['reason']}")
            lines.append("")
        for cls in CLASS_ORDER:
            items = [f for f in r.get("findings", []) if f["evidence_class"] == cls]
            if items:
                lines.append(f"**{cls}**")
                lines += [f"- {f['claim']}" + (f" (rule: {f['rule']})" if f.get("rule") else "") for f in items]
                lines.append("")
        if r.get("oracle"):
            lines.append(f"Oracle: {r['oracle']}")
        if r.get("limitations"):
            lines.append("Limitations: " + "; ".join(r["limitations"]))
        if r.get("reproducer"):
            lines.append(f"Reproduce: `{r['reproducer']}`")
    return "\n".join(lines) + "\n"


def to_junit(results: list[dict], meta: dict, strict: bool = False) -> str:
    def text(r):
        return "\n".join(f"[{f['evidence_class']}] {f['claim']}" for f in r.get("findings", []))

    failing = [r for r in results if r["status"] == "fail" or (strict and r["status"] == "not_tested")]
    skipped = [r for r in results if r["status"] == "not_tested" and not strict]
    suite = ET.Element("testsuite", name="conform", tests=str(len(results)), failures=str(len(failing)),
                       skipped=str(len(skipped)), errors="0")
    ET.SubElement(ET.SubElement(suite, "properties"), "property", name="summary", value=summary(results))
    for r in results:
        case = ET.SubElement(suite, "testcase", classname="conform", name=r["probe"])
        if r in failing:
            message = r.get("reason") or next((f["claim"] for f in r.get("findings", [])
                                                if f["evidence_class"] == "seen"), r["status"])
            ET.SubElement(case, "failure", message=message).text = text(r) or message
        elif r in skipped:
            ET.SubElement(case, "skipped", message=r.get("reason") or "not tested")
        out = text(r)
        if out:
            ET.SubElement(case, "system-out").text = out
    return ET.tostring(suite, encoding="unicode")


def write_reports(results: list[dict], out_dir: Path, meta: dict, strict: bool = False) -> dict:
    """Write report.json, report.md and report.xml; return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / "report.json", "markdown": out_dir / "report.md", "junit": out_dir / "report.xml"}
    paths["json"].write_text(to_json(results, meta), encoding="utf-8")
    paths["markdown"].write_text(to_markdown(results, meta), encoding="utf-8")
    paths["junit"].write_text('<?xml version="1.0" encoding="utf-8"?>\n' + to_junit(results, meta, strict),
                              encoding="utf-8")
    return paths
