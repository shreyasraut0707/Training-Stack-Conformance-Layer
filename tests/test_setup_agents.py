import json
import tomllib

from conform.bank import PROJECT_ROOT
from conform.cli import main
from conform.setup_agents import SKILL_SOURCE, apply, plan


def test_packaged_skill_matches_the_repository_skill():
    repo_copy = PROJECT_ROOT / "skills" / "conform" / "SKILL.md"
    assert SKILL_SOURCE.read_bytes() == repo_copy.read_bytes()


def test_dry_run_changes_nothing(tmp_path, capsys):
    assert main(["setup", "--agent", "claude-code", "--project", str(tmp_path)]) == 0
    assert "Nothing was changed" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_claude_code_merges_existing_config_and_backs_it_up(tmp_path):
    existing = {"mcpServers": {"other": {"command": "x"}}}
    (tmp_path / ".mcp.json").write_text(json.dumps(existing), encoding="utf-8")
    backups = apply(plan("claude-code", python="py", project=tmp_path))
    config = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert config["mcpServers"]["other"] == {"command": "x"}
    assert config["mcpServers"]["conform"] == {"command": "py", "args": ["-m", "conform.mcp_server"]}
    assert len(backups) == 1 and json.loads(backups[0].read_text(encoding="utf-8")) == existing
    assert (tmp_path / ".claude" / "skills" / "conform" / "SKILL.md").read_text(encoding="utf-8").startswith("---")


def test_codex_adds_or_replaces_its_table_only(tmp_path):
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('model = "x"\n\n[mcp_servers.conform]\ncommand = "old"\n\n[other]\nkey = 1\n',
                      encoding="utf-8")
    apply(plan("codex", python="py", home=tmp_path))
    data = tomllib.loads(config.read_text(encoding="utf-8"))
    assert data["model"] == "x" and data["other"] == {"key": 1}
    assert data["mcp_servers"]["conform"] == {"command": "py", "args": ["-m", "conform.mcp_server"]}
    assert (tmp_path / ".agents" / "skills" / "conform" / "SKILL.md").exists()  # Codex's documented location
    assert not (tmp_path / ".codex" / "skills").exists()


def test_applying_twice_is_idempotent(tmp_path):
    apply(plan("claude-code", python="py", project=tmp_path))
    assert apply(plan("claude-code", python="py", project=tmp_path)) == []
