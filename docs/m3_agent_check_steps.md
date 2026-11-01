# M3 agent check: step by step

M3's exit criterion (spec Section 11.1): **a clean install and a `diagnose` run work end to
end in both agents** (Claude Code and Codex).

Already verified on this computer: a clean install of the built wheel in a fresh environment
(command, skill and MCP server work); an MCP client running `diagnose` end to end over stdio
(`tests/test_mcp_server.py`). What remains is the run **inside** each agent. It needs your
sign-in and your trust of a folder, so it is yours to do. Both checks use one computer.

A test folder is already prepared: `C:\Users\Shreyas\conform-agent-test` (made by
`conform setup --agent claude-code --apply`). It contains `.mcp.json` (registers the
`conform` MCP server, using the `conform-agent` environment), the skill in
`.claude\skills\conform\SKILL.md`, and the example recipe `sft.yaml` with `rows.json`.

## Claude Code (no installation needed: the desktop app is Claude Code)

1. In the Claude desktop app, open the **Code** tab and start a **new session** in the folder
   `C:\Users\Shreyas\conform-agent-test`. When asked whether to trust the folder, choose to
   trust it (it only contains the files listed above).
2. When Claude Code asks whether to use the project's MCP server `conform`, approve it.
3. Type exactly:
   > My fine-tuned model never stops generating. Use the conform MCP server's diagnose tool
   > with symptom no-stop and the recipe sft.yaml in this folder. Do not run Python yourself.
4. Wait (a few minutes: real probes run).
5. **Pass if:** the agent calls `diagnose` from the `conform` server; reports P7, P2 and P3
   with their statuses (expected: all three pass for this recipe) and evidence classes
   (seen / inferred / unknown); reads out any unknown item; and gives the report path (a
   `conform-reports\diagnose-no-stop_…\report.md` file inside the test folder).
6. Tell me the result, or copy the agent's last message to me. I will record it.

## Codex

1. **Install** (PowerShell; this is OpenAI's official installer from the Codex README):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://chatgpt.com/codex/install.ps1 | iex"
```

   Alternative, since Node.js is already installed: `npm install -g @openai/codex`.

2. **Sign in:** run `codex` in any folder and choose **Sign in with ChatGPT** (needs a
   ChatGPT Plus, Pro, Business, Edu or Enterprise plan), or sign in with an OpenAI API key.
3. **Register the server and skill** (first look, then apply; the existing
   `~/.codex/config.toml` is backed up first):

```powershell
& $HOME\.venvs\conform-agent\Scripts\conform.exe setup --agent codex
```

```powershell
& $HOME\.venvs\conform-agent\Scripts\conform.exe setup --agent codex --apply
```

   This adds an `[mcp_servers.conform]` table to `~/.codex/config.toml` and the skill to
   `~/.agents/skills/conform/SKILL.md` (the locations in the Codex documentation, read on
   2026-11-01; not yet confirmed against a running Codex). Codex's own command
   `codex mcp add conform -- <python> -m conform.mcp_server` is an alternative for the server.

4. **Run it:**

```powershell
cd $HOME\conform-agent-test
```

```powershell
codex
```

   and type the same request as for Claude Code.
5. **Pass if:** the same as for Claude Code. If Codex does not see the server or the skill,
   copy its message to me: that is exactly what this check is for.

## Undo

`conform setup` wrote backups next to every file it changed (`*.bak-<timestamp>`). To undo:
delete the folder `C:\Users\Shreyas\conform-agent-test`; remove the `[mcp_servers.conform]`
table from `~/.codex/config.toml` (or restore its backup); delete `~/.agents/skills/conform`.
