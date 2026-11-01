"""`conform setup`: register the MCP server and install the skill for coding agents.

Modelled on the spec's description (Section 9): the user chooses the agent, reviews the
proposed changes, approves them, and existing files are backed up first. By default this
only prints the plan; `--apply` writes it. Files outside the chosen locations are never
touched.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Shipped inside the package so `conform setup` works after a plain `pip install`; the
# repository's skills/conform/SKILL.md is the same file (a test keeps them identical).
SKILL_SOURCE = Path(__file__).resolve().parent / "skills" / "conform" / "SKILL.md"
AGENTS = ("claude-code", "codex")


@dataclass
class Change:
    path: Path
    new_text: str
    note: str

    @property
    def action(self) -> str:
        return "modify" if self.path.exists() else "create"


def _server_command(python: str) -> dict:
    return {"command": python, "args": ["-m", "conform.mcp_server"]}


def _claude_mcp_json(project: Path, python: str) -> Change:
    path = project / ".mcp.json"
    config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    config.setdefault("mcpServers", {})["conform"] = _server_command(python)
    return Change(path, json.dumps(config, indent=2) + "\n",
                  "Claude Code project-scope MCP server registration (.mcp.json)")


def _codex_config(home: Path, python: str) -> Change:
    path = home / ".codex" / "config.toml"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if old:
        tomllib.loads(old)  # refuse to edit a file that is not valid TOML
    block = (f"[mcp_servers.conform]\ncommand = {json.dumps(python)}\n"
             f'args = ["-m", "conform.mcp_server"]\n')
    pattern = re.compile(r"^\[mcp_servers\.conform\]\n(?:(?!\[).*\n?)*", re.MULTILINE)
    new = pattern.sub(block, old) if pattern.search(old) else (old.rstrip("\n") + "\n\n" if old else "") + block
    tomllib.loads(new)
    return Change(path, new, "Codex MCP server registration (~/.codex/config.toml); format from the "
                             "Codex documentation, not yet verified against a running Codex")


def _skill(target_dir: Path, note: str) -> Change:
    return Change(target_dir / "conform" / "SKILL.md", SKILL_SOURCE.read_text(encoding="utf-8"), note)


def plan(agent: str, *, python: str = sys.executable, home: Path | None = None,
         project: Path | None = None) -> list[Change]:
    home, project = Path(home or Path.home()), Path(project or Path.cwd())
    if agent == "claude-code":
        return [_claude_mcp_json(project, python),
                _skill(project / ".claude" / "skills", "Claude Code project skill")]
    if agent == "codex":
        # Codex reads user skills from ~/.agents/skills (its documentation, read 2026-11-01).
        return [_codex_config(home, python),
                _skill(home / ".agents" / "skills", "Codex user skill (~/.agents/skills, per the Codex "
                                                    "documentation)")]
    raise ValueError(f"unknown agent {agent!r}; choose from {AGENTS}")


def apply(changes: list[Change]) -> list[Path]:
    """Write the changes, backing up every existing file first; returns the backups made."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backups = []
    for change in changes:
        if change.path.exists():
            if change.path.read_text(encoding="utf-8") == change.new_text:
                continue
            backup = change.path.with_name(f"{change.path.name}.bak-{stamp}")
            shutil.copy2(change.path, backup)
            backups.append(backup)
        change.path.parent.mkdir(parents=True, exist_ok=True)
        change.path.write_text(change.new_text, encoding="utf-8")
    return backups


def describe(changes: list[Change], python: str) -> str:
    lines = []
    if importlib.util.find_spec("mcp") is None:
        lines.append(f"warning: the 'mcp' package is not installed for {python}; "
                     "install it there before starting the server")
    for change in changes:
        lines.append(f"- {change.action} {change.path}  ({change.note})")
        if change.path.suffix in (".json", ".toml"):
            lines += [f"    {line}" for line in change.new_text.splitlines()]
    return "\n".join(lines)
