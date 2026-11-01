# Progress and milestone status

Source of truth: `Training_Stack_Conformance_Layer_Illustrated_Spec.pdf` (Sections 5–14).
A milestone is marked complete only when every part of its exit criterion (spec Section
11.1) is met. Evidence: `docs/m0_findings.md`, `docs/bank_results.md`, `bank/results/`.
Status as of 2026-11-01.

## Status

| Milestone | Exit criterion (spec) | Status | What is met | What is missing |
|---|---|---|---|---|
| **M0 Spike** | G1–G4 met | **Not complete: conditional go** | G1 (3 of 3 cases detected), G2 (0 fails at fixed pins), G3 (0 fails in controls, three tokenizer families), G4 10 repeats identical on this machine (220 of 220); extra same-computer evidence: a fresh environment from the lockfile reproduces every verdict and number, one CPU thread changes 1 of 72 verdicts (borderline P1-on-TRL runs recorded) | G4's **second-machine spot check: pending**. The owner has one computer, so it cannot be done now; it is not replaced by a virtual machine or by CI |
| **M1 v0.1** | all v0.1 probes pass their own unit tests and run on CPU against the current stable release | **Complete (0.1.0)** | P1, P2, P3, P6, P7, CLI, JSON / Markdown / JUnit reports, proxy factory for Llama- and Qwen-type families; tests pass; controls 24 of 24 on transformers 5.22.1 / trl 1.17.1 | — |
| **M2 Bank** | detection and false-positive results published, including cases the probes missed | **Complete** (exit criterion met) | `docs/bank_results.md` (4 of 4 historical case checks detected, 0 fails at fixed pins, misses listed; case C confirmed upstream by PR 46204); calibration done (fail 10× confirmed, pass band 2× kept by decision); mutation suite 49 of 49 applicable; hash-pinned lockfiles; **CI matrix ran on GitHub: all jobs green** (run 41326929545: 122 tests, 72 control runs with 0 fails, bank 12 of 12 identical to this computer); **weekly upstream run** works (first run 41328069985 on transformers 5.23.0.dev0 / trl 1.18.0.dev0: 36 controls, 0 fails; scheduled Mondays 06:00 UTC) | — |
| **M3 Agents** | a clean install and a `diagnose` run work end to end in both agents | **Not complete** | clean install of the built wheel verified in a fresh environment (command, skill, MCP server); MCP client ran `diagnose` end to end over stdio (test passes); skill, `conform setup` (plan, `--apply`, backups), release workflow with Trusted Publishing, `SECURITY.md` | the run **inside Claude Code and inside Codex** (`docs/m3_agent_check_steps.md`); publishing (`docs/publishing_steps.md`) |
| **M4 v0.2** | new probes meet the same stability and control thresholds as v0.1 | **Not complete: met on one machine** | P4, P5, P1M built and tested; controls 12 runs × 10 repeats, 120 of 120 match, identical verdicts, spread 0 (`controls_2026-11-01_171529/`); the same 12 identical in a fresh lockfile environment and on one CPU thread; DPO decision recorded (out of scope until a confirmed DPO bug) | the v0.1 stability threshold includes a **second-machine spot check: pending** (one computer only) |

Tests: **123 pass** (122 in the main environment, plus the MCP end-to-end test in
`conform-agent`; 2026-11-01). Version: **0.2.0.dev0** (pre-release; 0.1.0 is the completed M1).
Final report: `docs/COMPLETION_REPORT.md`.

## Open decisions (yours)

1. ~~P1 pass band~~: **decided 2026-11-01: keep 2× (fail multiplier 10× confirmed)**;
   recorded in `docs/calibration.md`.
2. Spec Section 14: contact TRL and Transformers maintainers before public claims
   (recommended: yes); first public artifact (the tool or the bank write-up).
3. The unused `bank-ga-fixed` environment (transformers 4.46.3, about 1.4 GB): keep or delete.
4. **Final name:** `conform` is taken on PyPI and GitHub; `training-stack-conform` and
   `conform-probes` are free on PyPI today (`docs/publishing_steps.md`). No upload is
   possible until this is decided.
5. **Borderline P1-on-TRL runs at a future second-computer check:** whether a pass ↔ info
   (unknown) change on `p1-trl` for SmolLM2, TinyLlama or Qwen2.5 counts against G4. They
   sit near the 2× pass band and moved with the CPU thread count on this computer
   (`docs/m0_findings.md`, "Extra single-computer evidence for G4"). Recorded before any
   second-computer run; the criterion and thresholds are unchanged.
6. **When the repository becomes public** (recommended: just before the first real release,
   after contacting maintainers; `docs/publishing_steps.md`, Step B).

## Tasks that need you in person

1. **Second-machine spot check (closes M0's G4 and M4's stability threshold): pending.**
   You have one computer, so this stays open. If you ever have the use of another Windows
   10/11 computer (borrowed for about two hours), follow `docs/g4_second_machine_steps.md`
   and bring the zip back; the folders are compared with `scripts/compare_results.py`.
2. **Agent check (closes M3).** Claude Code (no install: a session of this app in the test
   folder) and Codex (install and sign in): `docs/m3_agent_check_steps.md`.
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
