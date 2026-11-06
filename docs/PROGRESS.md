# Progress and milestone status

Source of truth: `Training_Stack_Conformance_Layer_Illustrated_Spec.pdf` (Sections 5–14).
A milestone is marked complete only when every part of its exit criterion (spec Section
11.1) is met. Evidence: `docs/m0_findings.md`, `docs/bank_results.md`, `bank/results/`.
Status as of 2026-11-04.

## Status

| Milestone | Exit criterion (spec) | Status | What is met | What is missing |
|---|---|---|---|---|
| **M0 Spike** | G1–G4 met | **Not complete: conditional go** | G1 (3 of 3 cases detected), G2 (0 fails at fixed pins), G3 (0 fails in controls, three tokenizer families), G4 10 repeats identical on this machine (220 of 220); extra same-computer evidence: a fresh environment from the lockfile reproduces every verdict and number, one CPU thread changes 1 of 72 verdicts (borderline P1-on-TRL runs recorded) | G4's **second-machine spot check: pending**. The owner has one computer, so it cannot be done now; it is not replaced by a virtual machine or by CI |
| **M1 v0.1** | all v0.1 probes pass their own unit tests and run on CPU against the current stable release | **Complete (0.1.0)** | P1, P2, P3, P6, P7, CLI, JSON / Markdown / JUnit reports, proxy factory for Llama- and Qwen-type families; tests pass; controls 24 of 24 on transformers 5.22.1 / trl 1.17.1 | — |
| **M2 Bank** | detection and false-positive results published, including cases the probes missed | **Complete** (exit criterion met) | `docs/bank_results.md` (4 of 4 historical case checks detected, 0 fails at fixed pins, misses listed; case C confirmed upstream by PR 46204); calibration done (fail 10× confirmed, pass band 2× kept by decision); mutation suite 49 of 49 applicable; hash-pinned lockfiles; **CI matrix ran on GitHub: all jobs green** (run 41326929545: 122 tests, 72 control runs with 0 fails, bank 12 of 12 identical to this computer); **weekly upstream run** works (first run 41328069985 on transformers 5.23.0.dev0 / trl 1.18.0.dev0: 36 controls, 0 fails; scheduled Mondays 06:00 UTC) | — |
| **M3 Agents** | a clean install and a `diagnose` run work end to end in both agents | **Complete** (2026-11-04) | clean install of the built wheel verified in a fresh environment (command, skill, MCP server); MCP client ran `diagnose` end to end over stdio (test passes); skill, `conform setup` (plan, `--apply`, backups), release workflow with Trusted Publishing, `SECURITY.md`; **Claude Code check passed 2026-11-03**; **Codex check passed 2026-11-03** (development install); **clean-install runs 2026-11-04** (`conform-clean`, wheel `f04d00c9…`): one Claude Code run and one Codex run with only the standard request each met all four pass conditions (evidence in the log below) | — |
| **M4 v0.2** | new probes meet the same stability and control thresholds as v0.1 | **Not complete: met on one machine** | P4, P5, P1M built and tested; controls 12 runs × 10 repeats, 120 of 120 match, identical verdicts, spread 0 (`controls_2026-11-01_171529/`); the same 12 identical in a fresh lockfile environment and on one CPU thread; DPO decision recorded (out of scope until a confirmed DPO bug) | the v0.1 stability threshold includes a **second-machine spot check: pending** (one computer only) |

Tests: **125 pass** (124 in the main environment, plus the MCP end-to-end test in
`conform-agent`; 2026-11-04). Against the installed wheel (`conform-clean`): 115 pass; the
other 5 need the repository's own files (bank, controls, scripts). Version: **0.2.0.dev0** (pre-release; 0.1.0 is the completed M1).
Final report: `docs/COMPLETION_REPORT.md`.

## Open decisions (yours)

1. ~~P1 pass band~~: **decided 2026-11-01: keep 2× (fail multiplier 10× confirmed)**;
   recorded in `docs/calibration.md`.
2. Spec Section 14: contact TRL and Transformers maintainers before public claims
   (recommended: yes); first public artifact (the tool or the bank write-up).
3. ~~The unused `bank-ga-fixed` environment~~: **decided 2026-11-01: keep it until all final
   verification is complete.**
4. **Final name:** `conform` is taken on PyPI and GitHub; `training-stack-conform` and
   `conform-probes` are free on PyPI today (`docs/publishing_steps.md`). No upload is
   possible until this is decided.
5. **Borderline P1-on-TRL runs (G4):** they move between pass and unknown with the CPU
   thread count and on GitHub's runners (TinyLlama 1.96× → 3.55×). The spec counts any
   changed verdict as a probe defect (Section 10.2, step 2). Recommended: option A, fix the
   probe's noise floor for TRL's kernel without changing 2× / 10×; alternatives B and C in
   `docs/m0_findings.md`, "G4 against the spec's text". Nothing changed until you decide.
6. **When the repository becomes public** (recommended: just before the first real release,
   after contacting maintainers; `docs/publishing_steps.md`, Step B).

## Tasks that need you in person

1. **Second-machine spot check (closes M0's G4 and M4's stability threshold): pending.**
   You have one computer, so this stays open. If you ever have the use of another Windows
   10/11 computer (borrowed for about two hours), follow `docs/g4_second_machine_steps.md`
   and bring the zip back; the folders are compared with `scripts/compare_results.py`.
2. ~~Agent check (closes M3)~~: **done 2026-11-04.** On the rebuilt clean install, one
   Claude Code run and one Codex run in `C:\Users\Shreyas\conform-clean-test`, standard
   request only, each met all four pass conditions. M3 is complete (log entry "M3
   complete", 2026-11-04).
3. **PyPI and TestPyPI accounts and the final name** (before any upload):
   `docs/publishing_steps.md`, sections 1 and 3.
4. **Maintainer contact and a user pilot**, if you choose to (`docs/publishing_steps.md`, section 4).

## Environments on this computer

| Environment | Purpose | Record |
|---|---|---|
| `conform` | main environment (current releases) | `bank/envs/conform.txt` |
| `conform-agent` | the same plus `mcp` 2.3.0, for M3 | `bank/envs/conform-agent.txt` |
| `bank-ga-defective`, `bank-eos-defective`, `bank-eos-fixed`, `bank-p0-defective`, `bank-p0-fixed` | bank pins | `bank/envs/*.txt` |
| `bank-ga-fixed` | no longer used (transformers 4.46.3) | `bank/envs/bank-ga-fixed.txt` |

## Log

- 2026-11-01: M0 evidence complete except the second-machine check; conditional go.
- 2026-11-01: M1 complete (0.1.0): P3, P6, P7, CLI, reports; controls 24 of 24; bank 10 of 10.
- 2026-11-01: M2: mutation suite (first run 49 of 51; the 2 misses were a mutant fault;
  corrected run 49 of 49 applicable), calibration measured, behavioural bank case for TRL
  PR 3200 (P7 detects it), CI workflow, published results with misses.
- 2026-11-01: M3: MCP server (mcp 2.x renamed FastMCP to MCPServer; both supported), skill,
  setup, release workflow, security notice; end-to-end MCP test passes.
- 2026-11-01: M4: P4, P5, P1M. First 10-repeat run stopped in repeat 3 when this 7.3 GB
  laptop ran out of memory (recorded in `docs/m0_findings.md`); after reducing P1M's and
  P5's memory use, the rerun passed 120 of 120 with identical verdicts.
- 2026-11-01: full test suite: 116 pass.
- 2026-11-01: P1 pass band kept at 2× by decision. Clean install from the wheel verified
  (fixed: the skill now ships in the package; availability now checks the installed stack);
  hash-pinned lockfiles; P4's private access marked fragile; case C confirmed by PR 46204;
  name check; guides for every external task; `scripts/compare_results.py`. Version
  0.2.0.dev0. Full test suite: 123 pass.
- 2026-11-01 (evening): one computer only, so the second-machine check stays pending.
  Extra same-computer evidence: fresh lockfile environment 36 of 36 identical; one CPU
  thread 35 of 36 identical (`p1-trl` SmolLM2 info → pass; borderline runs recorded).
  Codex skill location fixed; packaging metadata modernised (builds without warnings,
  `twine check --strict` passes); TestPyPI rehearsal added to the release workflow; CI job
  time limits. Local Git repository initialised (spec PDF excluded). Full test suite: 123
  pass (122 + the MCP end-to-end test in `conform-agent`).
- 2026-11-01: private GitHub repository `shreyasraut0707/Training-Stack-Conformance-Layer`
  created; first commit `e51c4a3` (657 files, about 2.4 MB). Secret scan of the working tree
  and of the pushed commit: no tokens, keys, passwords or binary files; personal details
  present (your Windows user folder name in 7 example paths in 3 docs; your name and e-mail
  on the commit), acceptable while private, to tidy before any public release.
- 2026-11-01: first CI run 41326929545: **all jobs success** (tests pinned 8 min 43 s,
  tests latest 11 min 5 s, bank on Windows 9 min 56 s). Compared with this computer: bank
  12 of 12 identical; controls 35 of 36 identical in each test job, 0 fails; the one change
  is the borderline `p1-trl` TinyLlama run (1.96× → 3.55×, pass → info), as recorded in
  advance. Cloud virtual machines: extra evidence, **not** the second-computer check.
  The upstream job moved to its own workflow (`upstream.yml`) so it can run alone.
- 2026-11-01: second CI run 41328058562 (commit `581d4a0`, the separated workflows): all
  jobs success; verdicts identical to the first CI run in all 84 runs (36 + 36 controls,
  12 bank), 0 fails.
- 2026-11-01: first upstream run 41328069985 (transformers 5.23.0.dev0, trl 1.18.0.dev0
  from `main`): success; 36 controls, 0 fails, verdicts identical to the CI runs on the
  current releases.
- 2026-11-01: M3 preparation: the agent test folder checked (files present, server uses
  this project's current code); `conform setup --agent codex` now sets Codex's tool and
  startup time limits (defaults 60 s and 10 s are shorter than a `diagnose` run);
  `docs/m3_agent_check_steps.md` rewritten. G4 reviewed against the spec's text (any
  changed verdict is a probe defect); recommendation recorded, criteria unchanged.
  Tests: 123 pass. CI run 41329885350 (commit `4f115ed`): all jobs success.
- 2026-11-03: **M3 agent check, Claude Code: passed.** Claude desktop app, Code tab, new
  session in `C:\Users\Shreyas\conform-agent-test`; model Claude Opus 5.5
  (`claude-opus-5-5`); app version not recorded (no `claude` command on PATH). The request
  from `docs/m3_agent_check_steps.md` typed as written. The agent loaded the `conform` skill
  and called `diagnose` (symptom `no-stop`, recipe `sft.yaml`) on the `conform` MCP server
  (stdio, `conform-agent` environment); it ran no Python for the check. Result: "3 pass",
  P7, P2, P3 in that order, every finding labelled seen (no inferred or unknown items in
  this run), a reproducer per probe, the recipe's limits stated (example recipe,
  `trainer_args: {}`), and the report path
  `conform-reports\diagnose-no-stop_2026-11-03_230813\report.md`. Checked afterwards: the
  folder holds `report.md`, `report.json` and `report.xml` (23:08:13 IST); 3 of 3 statuses
  pass in each, and every number matches the agent's message. Pass conditions (1) to (4):
  all met; (3) checked further by a follow-up `run_all` in the same session
  (`check_2026-11-03_231039`: 5 pass, 2 info, 1 not tested), where the agent read out both
  P1 unknown items word for word and gave P1M's not-tested reason. Not observed from inside
  the agent: whether the app asked to trust the folder or allow the server. One stray
  `python -` with no script was run during this verification (after the check; it did
  nothing). Codex: not yet run, so M3 stays **not complete**.
- 2026-11-03: Claude Code check re-verified from that session's own log
  (`~/.claude/projects/C--Users-Shreyas-conform-agent-test/b144baac-….jsonl`): the request
  as written; the `conform` skill loaded; one `mcp__conform__diagnose` call (`no-stop`,
  `sft.yaml`) at 17:36:53 UTC; reply with P7, P2, P3, all seen, "none of the probes reported
  an unknown item", and the report path. The `run_all` follow-up: P1 info (TRL kernel
  noise, as known), P1M not tested with its stated reason (Transformers Trainer only),
  P6 info (observation only), the rest pass.
- 2026-11-03: **M3 agent check, Codex: passed.** Codex CLI 0.166.0 (the npm copy is the one
  on PATH), model `gpt-6-luna`, session started in `C:\Users\Shreyas\conform-agent-test`.
  `~/.codex/config.toml` holds the `[mcp_servers.conform]` table written by `conform setup`
  (with `tool_timeout_sec = 3600`, `startup_timeout_sec = 30`); the skill is at
  `~/.agents/skills/conform/SKILL.md`. **Verified against the running Codex:** its session
  context lists `C:/Users/Shreyas/.agents/skills` as skill root `r0` and the skill as
  `r0/conform/SKILL.md`, so the documented locations are right. Session log
  (`~/.codex/sessions/2026/11/03/rollout-2026-11-03T23-25-03-….jsonl`): the owner's own
  request (use the conform MCP `diagnose` tool, not Python; report P7, P2, P3 with evidence
  classes; read out unknowns; give the report path). Codex's first `diagnose` call passed a
  sentence as the symptom; the server answered "unknown symptom …; choose from ['no-stop',
  'worse-than-base']". Codex then read the skill file (PowerShell `Get-Content`, no Python)
  and called `diagnose` again with `no-stop` and `sft.yaml` (MCP call completed 17:59:10
  UTC, about 60 s: Codex's default 60 s tool limit would have been borderline). Final
  message: 3 pass, 0 fail, 0 not tested; P7, P2, P3 each pass with seen evidence; "Unknown
  items: None"; report paths given. Checked afterwards:
  `conform-reports\diagnose-no-stop_2026-11-03_232910\` holds `report.md`, `report.json`
  and `report.xml` (23:29:10 IST); 3 of 3 pass, every finding seen, every number matching
  Codex's message and identical to the Claude Code run's report. Pass conditions (1) to
  (4): all met. Observed: the server reports an unknown symptom as an error message inside
  a normal (not `isError`) result; Codex recovered, but marking it as an error would be
  clearer (not changed: source changes not requested).
- 2026-11-03: **M3 still not complete**: both agent runs used the editable development
  install (`conform-agent`), not a clean install from the built package; the clean wheel
  install of 2026-11-01 was checked separately and its environment no longer exists.
- 2026-11-03: **clean install prepared for the final M3 runs.** Wheel built from commit
  `4f115ed` (no uncommitted source changes; sha256 `90f3676e…`). New environment
  `~/.venvs/conform-clean`: dependencies from `bank/envs/conform-agent.lock` with
  `--require-hashes`, then the wheel with `[stack,mcp]` (not editable); `pip check` clean;
  78 of 78 packages identical to `bank/envs/conform-agent.txt`; `conform` loads from the
  environment's `site-packages`; the shipped skill is byte-identical to `skills/`; the
  command works from outside the project. **Test suite against the installed package: 115
  pass, 4 fail, 1 collection error**, all five because `PROJECT_ROOT` (`conform/bank.py`)
  then points at `~/.venvs/conform-clean/Lib`: the bank cases, the controls file, a script
  and the repository skill are not in the package. The MCP end-to-end test passes from the
  clean install. New test folder `C:\Users\Shreyas\conform-clean-test` (same inputs;
  `.mcp.json` and skill written by the clean environment's `conform setup`).
- 2026-11-03: **two findings from the clean install (source not changed):**
  (1) **`conform setup --agent codex` fails when `~/.codex/config.toml` already has a
  `conform` table and the Python path contains backslashes** (every Windows path):
  `TOMLDecodeError: Invalid hex value`. Cause: `re.sub` reads backslashes in the
  replacement text as escapes, so `C:\\Users` becomes `C:\Users`; the existing test uses
  `py` and missed it. The safety check refused to write, so no file was changed. The first
  setup worked because it appends instead. (2) `conform bank run` and the controls work only
  from a repository checkout (the bank's case files are not shipped in the package); this
  is how they are used (CI and the scripts), but it is not stated in the docs.
- 2026-11-03: **finding (1) fixed** (owner's approval): `_codex_config` now replaces the
  table with a function (`pattern.sub(lambda _: block, old)`), so backslashes are kept; the
  setup note now says the format was checked with Codex CLI 0.166.0. New regression test
  `test_codex_replaces_a_windows_path` (two setups with Windows paths, then idempotent):
  **fails on the old code** (`TOMLDecodeError: Invalid hex value`, run against the
  `4f115ed` wheel) and passes on the fix. Full test suite: **124 pass** (123 in the main
  environment, 1 skipped there and passed in `conform-agent`: the MCP end-to-end test).
  Wheel rebuilt from `4f115ed` plus this uncommitted fix (sha256 `24e43f2a…`) and installed
  into `conform-clean` with `--force-reinstall --no-deps` (dependencies untouched: 78 of 78
  as recorded; `pip check` clean); the regression and MCP tests pass from it. Agents
  pointed at the clean environment with its own `conform setup`: the Claude Code test
  folder needed no change; `~/.codex/config.toml` now runs
  `~/.venvs/conform-clean/Scripts/python.exe` (other sections kept; backup
  `config.toml.bak-20261103-235618`); `codex mcp list` shows `conform` enabled with that
  command. Seen while doing so: the plan labels an existing, already identical file
  "modify" although `--apply` leaves it untouched (cosmetic; not changed).
- 2026-11-04: **M3 clean-install runs** in `C:\Users\Shreyas\conform-clean-test`, the
  request from `docs/m3_agent_check_steps.md` as written. Both agents used the clean
  environment: it is the only `conform` server registered for either (no user-scope entry
  in `~/.claude.json`), both configurations were last changed before the runs (`.mcp.json`
  23:47:16, `~/.codex/config.toml` 23:56:18), and both session logs mention `conform-clean`
  (55 and 56 times) and never `conform-agent`.
  - **Claude Code** (log `~/.claude/projects/C--Users-Shreyas-conform-clean-test/49068d8f-….jsonl`):
    one `mcp__conform__diagnose` call (`no-stop`, `sft.yaml`; 18:30:19 UTC); no Python
    (it listed the folder with Glob); the skill was not loaded this time. Reply: all 3 pass,
    P7, P2 and P3 each described, "All evidence above was directly observed, none
    inferred", "The tool reported no unknowns", report path
    `conform-reports/diagnose-no-stop_2026-11-04_000114/report.md`. Conditions (1) to (4)
    met.
  - **Codex** (log `~/.codex/sessions/2026/11/04/rollout-2026-11-04T00-02-26-….jsonl`):
    the conform skill listed at session start but not opened; one PowerShell command
    (`Get-Location; Get-ChildItem -Name sft.yaml`, not Python); one `diagnose` call on the
    `conform` server (`no-stop`, `sft.yaml`; 18:33:25 UTC). Reply: P7, P2 and P3 pass,
    "Unknown items: None reported", report path `…_000325\report.md`; **no evidence class
    quoted for P2 and P3** (P7 only "behavior confirmed"). Conditions (1), (3), (4) met;
    **(2) only partly met**.
  - Reports: `report.md`, `report.json`, `report.xml` in each folder (00:01:14 and
    00:03:25 IST); 3 of 3 pass with every finding seen; `report.md` identical line for line
    (apart from the folder name) to the development-install runs.
- 2026-11-04: **M3 not marked complete.** The mechanics of the exit criterion hold in both
  agents with a clean install (install, registration by `conform setup`, `diagnose` end to
  end, report written). One required behaviour is missing in the Codex clean run: the spec
  says the agent quotes the evidence class (Section 9, the skill; Figure 9, step 7: "what
  was seen, what is inferred, what is unknown"), and Codex did not, apparently because it
  did not open the skill. (In the earlier Codex run the owner's own prompt asked for the
  classes, so that run does not show the skill doing it.)
- 2026-11-04: **reporting instruction moved where every agent sees it** (owner's choice:
  fix, then rerun Codex). `conform.mcp_server`: the descriptions of `diagnose`, `run_all`
  and `run_probe` now end with "Tell the user each probe's status and quote each finding's
  evidence class (seen, inferred or unknown). Read every 'unknown' item to the user. Claim
  harm only when a behavioural probe (P7) confirmed it."; the same `note` now comes
  **first** in each result instead of last; each probe's result carries an `evidence` count
  per class (spec Section 9: `diagnose` returns "ordered results with evidence classes").
  The MCP end-to-end test checks all three; it **fails on the previous wheel** and passes
  now. Full test suite: **124 pass**. Wheel rebuilt from `4f115ed` plus both uncommitted
  changes (sha256 `ac9b53c0…`) and put into `conform-clean` (`--force-reinstall
  --no-deps`; 78 of 78 dependencies unchanged; `pip check` clean); the MCP and setup
  regression tests pass from it. Read back from the clean server over stdio, exactly as
  Codex receives it, the `diagnose` description contains the instruction. Codex's
  registration unchanged (`codex mcp list`: `conform` enabled, clean environment).
- 2026-11-04: **Codex rerun with the updated clean install** (log
  `~/.codex/sessions/2026/11/04/rollout-2026-11-04T00-16-34-01a1a2bc-….jsonl`; session in
  `conform-clean-test`; the log mentions `conform-clean` 56 times, `conform-agent` never).
  Codex received the new instruction (it is in the log), ran one PowerShell command
  (`Get-Item` / `Get-Content` of `sft.yaml`, not Python) and one `diagnose` call (`no-stop`,
  `sft.yaml`; 18:47:37 UTC); the result began with the note and held the per-class counts.
  Final message: a table of P7, P2 and P3, each "pass" with "seen (3)", "seen (4)",
  "seen (1)", "There are **no unknown items**", and the report path
  `…_001737\report.md`. Report checked: `report.md`, `.json`, `.xml` (00:17:37); 3 pass, all
  seen; identical to the earlier runs. Configurations and installed code unchanged since
  the rebuild. **But the request typed was not the standard one:** after the standard
  sentence it added "Report P7, P2, and P3 results, including each probe's status and
  evidence class (seen, inferred, or unknown). Clearly read out all unknown items … Provide
  the generated report path." So the run shows Codex reporting the classes when asked, not
  that conform's own instruction makes it do so: the point the fix was for.
- 2026-11-04: Claude Code clean-install evidence re-checked: one prompt, the standard
  request exactly; one `mcp__conform__diagnose` call; no Python; all four conditions met;
  on the earlier clean build (`24e43f2a…`, before the reporting change).
- 2026-11-04: **M3 not marked complete.** Shown in both agents with a clean install:
  install, registration by `conform setup`, `diagnose` end to end, report written, results
  reported. Not yet shown under the documented procedure (the standard request alone):
  Codex quoting the evidence classes. Closing step: one Codex run in
  `conform-clean-test` with only the standard request (about 3 minutes); recommended in
  the same sitting, one Claude Code run there too, so that both agents' final evidence
  comes from the same build.
- 2026-11-04: **Codex, standard request only, updated clean install** (log
  `~/.codex/sessions/2026/11/04/rollout-2026-11-04T00-22-22-01a1a2c1-….jsonl`; session in
  `conform-clean-test`; `conform-clean` 68 mentions, `conform-agent` none). The first
  message is **exactly the standard request** (checked character by character; a second
  message, "done", was typed into Codex afterwards and answered "Got it."). Codex said it
  was applying the `conform` skill and read `~/.agents/skills/conform/SKILL.md`
  (PowerShell `Get-Content`, not Python), then made one `diagnose` call on the `conform`
  server (`no-stop`, `C:/Users/Shreyas/conform-clean-test/sft.yaml`; 18:54:32 UTC); the
  result began with the note and held the per-class counts. Reply: "3 pass, 0 fail, 0 not
  tested"; P7, P2 and P3 each "pass" with "**Seen:**" and the observed facts; "There were
  **no inferred or unknown findings**"; the proxy-model limit stated. **No report path in
  any of its messages.** Report `…_002432\` (`report.md`, `.json`, `.xml`, 00:24:32): 3 pass,
  all seen; identical to the earlier runs except the recipe path's `/` separators.
  Conditions (1), (2), (3) met; **(4) not met**.
- 2026-11-04: **M3 not marked complete.** No standard-request Codex run meets all four
  conditions together: the run at 00:03 missed (2) (old build), this one misses (4). Cause
  on conform's side: neither the skill's "How to report the results (required)" section nor
  the tool's note asks the agent to give the user the report path, although the spec's
  intended session ends with "a short summary and a path to the full report" (Figure 9,
  step 6); earlier runs gave it by the agent's own choice. Also seen: the skill still says
  P1's thresholds are "provisional until calibrated" (calibrated in M2; outdated wording).
  Claude Code's clean-install run (standard request, all four conditions) stands, on the
  earlier build `24e43f2a…`.
- 2026-11-04: **report-path rule added** (owner's choice: fix, then rerun both agents).
  `HOW_TO_REPORT` (tool descriptions of `diagnose`, `run_all`, `run_probe`) now ends "Give
  the user the path of the full report (the 'report' field)"; the result's `note` also
  ends "Full report: <path>". The skill (`skills/` and the packaged copy, identical) gains
  the rule "Give the user the path of the full report (the `report` field)" under "How to
  report the results (required)", and its limits now say P1 passes up to 2× and fails above
  10× (the outdated "provisional until calibrated" removed; "single-process CPU" corrected,
  since P1M uses two processes). Tests: the MCP test checks the description and the note's
  path; new `test_skill_states_the_reporting_rules`; both **fail on the previous wheel**
  and pass now. Full test suite: **125 pass**. Wheel rebuilt (sha256 `f04d00c9…`, from
  `4f115ed` plus the uncommitted changes) and put into `conform-clean` (78 of 78
  dependencies unchanged, `pip check` clean); the three regression tests pass from it.
  Skill copies refreshed by the clean environment's `conform setup` (backups
  `SKILL.md.bak-20261104-003823` in `conform-clean-test\.claude\skills\conform\` and
  `~/.agents/skills/conform/`); server registrations unchanged. Read back from the clean
  server: the `diagnose` description carries all four rules. Waiting for: one Claude Code
  run and one Codex run in `conform-clean-test`, standard request only.
- 2026-11-04: **M3 complete.** The two runs left after the report-path fix, both in
  `C:\Users\Shreyas\conform-clean-test` with only the standard request, on the rebuilt
  clean install (`conform-clean`: wheel `f04d00c9…` installed at 00:37, before both runs;
  its installed `mcp_server.py` carries the report-path rule). The only `conform` server
  either agent could reach was the clean one (the test folder's `.mcp.json`; no entry in
  `~/.claude.json`; `~/.codex/config.toml` unchanged since 23:56:18).
  - **Claude Code** (Claude desktop app, Code tab; Claude Code 2.1.309, from the session
    log's version field; model `claude-opus-5-5`; log
    `~/.claude/projects/C--Users-Shreyas-conform-clean-test/150768fc-….jsonl`): request at
    00:41:24 IST; the `conform` skill loaded; `sft.yaml` read; one `mcp__conform__diagnose`
    call (`no-stop`, `sft.yaml`) at 00:41:38; no Python. Reply: "3 pass, 0 fail, 0 not
    tested"; P7, P2, P3 each "Pass" with every finding labelled "seen"; "No finding came
    back as unknown"; report path `…_004230/report.md` (as a link). Conditions (1) to (4):
    all met.
  - **Codex** (Codex 0.166.0, model `gpt-6-luna`; log
    `~/.codex/sessions/2026/11/04/rollout-2026-11-04T00-42-57-01a1a2d4-….jsonl`): request
    at 00:43:07; one `mcp__conform__diagnose` call (`no-stop`, `sft.yaml`) at 00:43:15,
    made through Codex's `exec` tool; its other calls listed its tools and ran a PowerShell
    `Get-ChildItem` to find `sft.yaml`; no Python. It did not open the skill (the skill's
    text, such as "How to report the results", is not in its log), so its reporting
    followed the tool's description and note. Reply: "3 passes"; P7, P2, P3 each "pass"
    with "Evidence class **seen**"; "no unknown findings"; report path
    `…_004406\report.md`. Conditions (1) to (4): all met. Its log mentions `conform-clean`
    59 times and `conform-agent` never.
  - Reports `…_004230\` and `…_004406\` (`report.md`, `report.json`, `report.xml`): 3 pass,
    every finding seen; the two `report.md` files are identical; every number in both
    replies matches them.
  - Recorded at the owner's request by the same Claude Code session that made the Claude
    Code run: a self-check, made against both session logs and the report files (the owner
    can confirm it from that session's first reply). Later turns in that session (the
    owner's follow-up questions: a read-only search for a training script, reading these
    documents) were not part of the check and ran no Python.

