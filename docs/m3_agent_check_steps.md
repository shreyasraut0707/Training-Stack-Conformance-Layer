# M3 agent check: step by step

M3's exit criterion (spec Section 11.1): **a clean install and a `diagnose` run work end to
end in both agents** (Claude Code and Codex). Both checks use this one computer.

Already verified: a clean install of the built wheel in a fresh environment (command, skill
and MCP server work); an MCP client running `diagnose` end to end over stdio
(`tests/test_mcp_server.py`, 76 s). What remains is the same run **inside** each agent, with
you at the keyboard (it needs your sign-in and your approval of the folder and the server).

**The test folder** `C:\Users\Shreyas\conform-agent-test` is ready (checked 2026-11-01). It
contains `.mcp.json` (registers the `conform` MCP server, run by the `conform-agent`
environment, which uses this project's current code), the skill in
`.claude\skills\conform\SKILL.md`, and the example recipe `sft.yaml` with `rows.json`.

**The request to type** (the same in both agents):

> My fine-tuned model never stops generating. Use the conform MCP server's diagnose tool
> with symptom no-stop and the recipe sft.yaml in this folder. Do not run Python yourself.

**Pass if** the agent (1) calls `diagnose` from the `conform` server; (2) reports P7, P2 and
P3 with their statuses (expected for this recipe: all three pass) and evidence classes
(seen / inferred / unknown); (3) reads out any unknown item rather than hiding it; and
(4) gives the report path: a `conform-reports\diagnose-no-stop_…\report.md` file in the
test folder. A run takes about 1 to 3 minutes.

## Part 1: Claude Code (no installation: this desktop app is Claude Code)

1. In the Claude desktop app, go to the **Code** tab and start a **new session**.
2. Choose the folder `C:\Users\Shreyas\conform-agent-test` for it (not the project folder).
3. If asked whether to trust the folder, choose **Trust** (it holds only the four files above).
4. If asked whether to use the project's MCP server **conform**, choose **Allow** / **Use**.
   (If you are not asked, type `/mcp` in that session to see whether `conform` is listed and
   connected.)
5. Paste the request above and press Enter.
6. If the agent asks permission to run the `diagnose` tool, allow it.
7. Wait for its answer (about 1 to 3 minutes).
8. Copy the agent's **whole final message** and paste it to me in this session. Also tell me
   if anything was refused or failed.

I then check the report file it wrote and record the result.

## Part 2: Codex

**Step 1. Install** (PowerShell; the official installer from OpenAI's Codex README):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://chatgpt.com/codex/install.ps1 | iex"
```

   Alternative (Node.js 24 is already installed): `npm install -g @openai/codex`.
   Close and reopen PowerShell afterwards, then check:

```powershell
codex --version
```

**Step 2. Sign in:** run `codex` and choose **Sign in with ChatGPT** (needs a ChatGPT Plus,
Pro, Business, Edu or Enterprise plan) or sign in with an OpenAI API key. Then quit Codex
(type `/quit` or press Ctrl+C).

**Step 3. Register the server and the skill.** First look at the plan (changes nothing):

```powershell
& $HOME\.venvs\conform-agent\Scripts\conform.exe setup --agent codex
```

   Then apply it:

```powershell
& $HOME\.venvs\conform-agent\Scripts\conform.exe setup --agent codex --apply
```

   This creates `~/.codex/config.toml` with an `[mcp_servers.conform]` table (backing up any
   existing file first) and copies the skill to `~/.agents/skills/conform/SKILL.md`. The
   table sets `tool_timeout_sec = 3600` and `startup_timeout_sec = 30`: Codex's defaults
   (60 s per tool call, 10 s to start a server) are shorter than a real `diagnose` run. These
   locations and keys come from the Codex documentation; this check is what confirms them.

**Step 4. Run it:**

```powershell
cd $HOME\conform-agent-test
```

```powershell
codex
```

   If Codex asks whether to trust the folder, choose yes. Type `/mcp` to check that
   `conform` is listed, then paste the request above. Approve the `diagnose` tool call if
   asked.

**Step 5.** Copy Codex's whole final message to me, or any error. If Codex does not see the
server or the skill, that message is exactly what this check is for.

## Part 3: the same checks with a clean install (closes M3)

Parts 1 and 2 passed on 2026-11-03, but both agents used `conform-agent`, an editable
development install. M3 asks for a **clean install** and a `diagnose` run in both agents, so
the two checks are repeated against `C:\Users\Shreyas\.venvs\conform-clean`: a fresh
environment with the built wheel installed normally (not editable; the same 78 packages,
hash-pinned from `bank/envs/conform-agent.lock`).

A second test folder is ready: `C:\Users\Shreyas\conform-clean-test` (the same `sft.yaml`
and `rows.json`; `.mcp.json` and the skill written by the clean environment's
`conform setup`). The first test folder and its reports stay as they are.

**Claude Code:** as in Part 1, but start the new session in
`C:\Users\Shreyas\conform-clean-test`. Trust the folder, allow the `conform` server, paste
the same request, and copy the final reply to me.

**Codex:** `~/.codex/config.toml` already points at the clean environment (done on
2026-11-03 with the clean environment's `conform setup`; the previous file is backed up as
`config.toml.bak-20261103-235618`; `codex mcp list` shows `conform` enabled). Then:

```powershell
cd $HOME\conform-clean-test
```

```powershell
codex
```

   Paste the same request and copy the final reply to me.

**Pass if:** the same four conditions as in Parts 1 and 2, and the report appears under
`C:\Users\Shreyas\conform-clean-test\conform-reports\`.

## Undo

`conform setup` writes a backup next to every file it changes (`*.bak-<timestamp>`). To
undo: delete the folder `C:\Users\Shreyas\conform-agent-test`; delete
`~/.codex/config.toml` (or restore its backup, if one was made); delete
`~/.agents/skills/conform`. Uninstall Codex with `npm uninstall -g @openai/codex` if you
installed it with npm.
