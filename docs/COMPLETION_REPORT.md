# Training-Stack Conformance Layer: completion report

Date: 2026-11-04 · Version: 0.2.0.dev0 · Source of truth:
`Training_Stack_Conformance_Layer_Illustrated_Spec.pdf` · Detailed status: `docs/PROGRESS.md`

## Bottom line

**The project is not yet complete by the spec's acceptance criteria.** Everything that can
be built and verified on this computer is done and tested (126 tests pass). One
acceptance check remains open: the **second-computer spot check** (M0's G4 and M4's
stability threshold) is **pending**: the owner has only one computer, and it has not been
replaced by a virtual machine, by CI or by more runs on this computer. **M3 is complete**
(2026-11-04): a clean install and a `diagnose` run work end to end in both agents, and
with only the standard request each agent met all four pass conditions.

| Milestone | Status |
|---|---|
| M0 Spike | Not complete: conditional go. G1, G2, G3 met; G4 met on this computer, with extra same-computer evidence; the second-computer spot check is pending (one computer only) |
| M1 v0.1 | **Complete** (0.1.0) |
| M2 Bank | **Complete**: results published with misses; the CI matrix ran on GitHub with every job green (runs 41326929545 and 41328058562), and the weekly upstream run works (41328069985) |
| M3 Agents | **Complete** (2026-11-04): clean install plus `diagnose` end to end in both agents; with only the standard request, Claude Code and Codex each met all four pass conditions |
| M4 v0.2 | Not complete: P4, P5, P1M built; stability and controls met on this computer (120 of 120, plus a fresh environment and one CPU thread); second-computer check pending |

## What was built

**Probes** (each has a plain question, an oracle independent of the library under test,
and a verdict rule; each runs in its own time-limited process; a probe that cannot run
says `not_tested` and why):

| Probe | Question | Oracle |
|---|---|---|
| P1 | Does gradient accumulation match one big batch? | float64 reference maths with a measured noise floor |
| P1M | Is the loss averaged correctly across processes (2 CPU processes, gloo)? | the same reference over all rows |
| P2 | Which tokens carry loss, and does that match the declared intent? | the trainer's own first batch against the declared intent |
| P3 | Are training tokens the same as inference tokens? | the tokenizer's chat template with the generation prompt |
| P4 | Does one packed document leak into another? | metamorphic: edit A, B's loss must not change |
| P5 | Does a resumed run match an uninterrupted run? | the uninterrupted run (deterministic CPU float32) |
| P6 | What precision does training actually use? | observation only (never fails) |
| P7 | After a short fine-tune, does the saved and reloaded model learn to stop? | behaviour: EOS probability and bounded generation |

**Around them:** a tiny proxy-model factory (real tokenizer, random weights, never real
weights); evidence labels seen / inferred / unknown with no overall pass badge; the
`conform` command (`check`, `diagnose`, `diff`, `bank run`, `list-probes`, `setup`; exit
codes 0 / 1 / 2, `--strict`); JSON, Markdown and JUnit reports; recipes with your own rows,
intent and trainer settings; a regression bank of pinned environments; controls; a mutation
suite; noise-floor calibration; an MCP server (stdio), a skill and agent setup with backups;
CI and release workflows; hash-pinned lockfiles; security notice.

## What the evidence shows

- **Historical bugs detected (bank):** the 2024 gradient-accumulation bug (transformers
  4.45.2); TRL's PAD = EOS masking bug (trl 0.16.0), seen by P2 and confirmed as behaviour
  by P7; and a third bug found by this tool, the position-0 label count (transformers 5.9.0),
  since confirmed by the maintainers' fix (PR 46204). **4 of 4 checks detected, 0 fails at
  the fixed versions.**
- **No false alarms on current releases:** 0 fails in every control run (24 v0.1 controls;
  120 M0 runs; 120 M4 runs), three tokenizer families.
- **Stable:** identical verdicts in every one of 10 repeats (220 + 120 runs), numeric spread 0.
- **Reproducible from the lockfile (same computer):** a fresh environment built only from
  `bank/envs/conform.lock` gave the same verdicts and the same numbers in all 36 control
  runs. With one CPU thread, 35 of 36 verdicts were the same; `p1-trl` on SmolLM2 moved
  from unknown to pass. Three P1-on-TRL runs sit near the 2× pass band (none near the 10×
  fail line), so a second computer may report them differently; this was recorded before
  any such check (`docs/m0_findings.md`).
- **GitHub CI (cloud virtual machines; not the second computer):** first run all green.
  122 tests pass on Ubuntu; 72 control runs (pinned stack and latest releases), **0 fails**,
  35 of 36 verdicts identical to this computer in each job; the bank on GitHub's Windows
  runner, **12 of 12 identical**. The one change is the borderline `p1-trl` TinyLlama run
  (1.96× → 3.55×: pass → unknown), the change recorded in advance; nothing came near the
  10× fail line.
- **Upstream main:** the weekly upstream workflow's first run (transformers 5.23.0.dev0,
  trl 1.18.0.dev0) passed: 36 controls, 0 fails, no verdict needing triage.
- **Injected defects:** 49 of 49 applicable caught (a first run's 2 misses were a fault in
  the injected defect, published in `docs/bank_results.md`).
- **P1's wider noise floor (2026-11-06):** the floor now also samples the batchings the
  trainers use and several CPU thread counts, with P1's own maths. Correct TRL runs peak at
  2.10× (was 3.82×); the six P1 controls give the same verdicts on 1 and 6 threads; bank,
  mutation suite and defect detection unchanged; 10 repeats stable; thresholds unchanged.
  On GitHub's Linux runners every verdict now matches this computer (36 of 36 controls,
  pinned and latest; the bank 12 of 12 on the Windows runner), where one P1-on-TRL run
  differed before (CI run 42267477781).
- **Inside Claude Code (M3, 2026-11-03):** in a new Code-tab session of the Claude desktop
  app (model Claude Opus 5.5) in the test folder, the agent called `diagnose` (no-stop,
  `sft.yaml`) on the `conform` MCP server without running Python, reported P7, P2 and P3
  (3 pass) with their evidence classes, and gave the report path
  (`conform-reports\diagnose-no-stop_2026-11-03_230813\report.md`); the report files
  match its message. In a follow-up `run_all` (5 pass, 2 info, 1 not tested) it read out
  both P1 unknown items word for word. All four pass conditions met.
- **Inside Codex (M3, 2026-11-03):** Codex CLI 0.166.0 in the test folder called
  `diagnose` (no-stop, `sft.yaml`) on the `conform` MCP server without running Python
  (a first call with a malformed symptom got a clear error, and Codex corrected it after
  reading the skill), reported P7, P2 and P3 (3 pass, all seen, "Unknown items: None") and
  gave the report path (`conform-reports\diagnose-no-stop_2026-11-03_232910\`); the report
  files match its message and the Claude Code run's report. All four pass conditions met.
  Codex's session context confirms the skill location `~/.agents/skills` and the server
  registration that `conform setup` writes.
- **Clean-install runs (M3, 2026-11-04):** a fresh environment with the built wheel
  (`~/.venvs/conform-clean`, not editable, hash-pinned dependencies), registered in both
  agents by its own `conform setup`, in a new test folder. Both agents called `diagnose`
  (no-stop, `sft.yaml`) on that server and gave the report path; both reports say 3 pass,
  every finding seen, identical to the earlier runs. Claude Code described the evidence as
  "directly observed, none inferred" and "no unknowns": all four conditions met. Codex
  gave the statuses and "Unknown items: None reported" but quoted no evidence class for P2
  and P3: three of four. Logs confirm each agent used only the clean environment.
- **After the fix (2026-11-04):** the `diagnose` tool's description and result now ask the
  agent to quote each finding's evidence class. A Codex rerun on the rebuilt clean install
  reported every probe as "seen" with counts, said there were no unknown items, and gave
  the report path; but its prompt also asked for the evidence classes. A further run with
  only the standard request quoted the classes unprompted ("Seen:" for each probe, "no
  inferred or unknown findings") but did not give the report path, which conform's skill
  and tool did not then ask for (the spec's Figure 9 expects "a path to the full report").
- **Closing runs (M3, 2026-11-04):** with the tool and the skill now also asking for the
  report path (clean install rebuilt), one run in each agent with only the standard
  request. Claude Code (2.1.309, Claude Opus 5.5) and Codex (0.166.0, `gpt-6-luna`) each
  called `diagnose` (no-stop, `sft.yaml`) on the clean install's server without running
  Python, reported P7, P2 and P3 (3 pass) with every finding labelled seen, said there
  were no unknown items, and gave the report path (`…_004230` and `…_004406`); the two
  reports are identical and match the replies. Codex did not open the skill, so the tool's
  own description and note were enough. All four pass conditions met in both: **M3
  complete.** (Recorded by the Claude Code session that made its run, checked against
  both session logs; **independently re-verified on 2026-11-06** by the main development
  session from both logs and both reports: exact standard prompts, one `diagnose` call
  each, no Python, only the clean server reachable, all four conditions met. The release
  workflow, an M3 scope item, is built but has not published anything, by your choice.)
- **Notable facts about current libraries:** TRL turns bf16 mixed precision on by default;
  the standard Transformers collator with a PAD = EOS tokenizer trains a model that never
  stops (measured by P7); Transformers isolates packed documents on CPU.

## Tests

126 automated tests pass (125 in the main environment, 1 MCP end-to-end test in
`conform-agent`), covering every probe, its injected defects, the reports, the CLI, the
bank runner, setup and the comparison tool. Run against the installed wheel instead of the
source tree, 115 pass; the other 5 test repository-only parts (bank cases, controls file,
scripts), which are not shipped in the package.

The clean-install check found and fixed one real defect: a second `conform setup --agent
codex` broke on Windows paths (regression test added; it fails on the old code).

## Limitations (stated in every report where they apply)

- Proxy models only: no real weights, so failures that need trained weights are not seen.
- CPU only: GPU kernels, mixed-precision GPU paths, DeepSpeed, FSDP and multi-GPU are not
  tested (P1M covers two CPU processes).
- P1 on TRL: the floor uses only P1's own maths, and TRL's float32 log-prob kernel rounds at
  up to about 2× of it (one of 30 correct runs at 2.10×), so a P1-on-TRL run near the 2×
  pass band can still read as unknown, and could read differently on another computer. The
  wider floor (2026-11-06) removed the thread-count flips seen on this computer.
- Case A's bank pair differs in 22 packages (cases B and C differ only in the library).
- DPO and reinforcement-learning trainers are out of scope (DPO until a confirmed DPO bug).
- Codex's configuration format and skill location: verified against a running Codex
  (0.166.0) on 2026-11-03.
- Lockfiles are specific to Windows and CPython 3.11.
- The bank evidence shows the probes behave as specified for the listed cases and versions;
  it does not prove the tool finds unknown bugs or that these defects matter to most users.

## Repository and publishing

- GitHub: **private** repository `shreyasraut0707/Training-Stack-Conformance-Layer`
  (created 2026-11-01), branch `main`; the spec PDF is not in it. Secret scan: no tokens,
  keys, passwords or binary files. Before it could ever become public: replace the 7
  example paths that contain your Windows user folder name, and decide whether your
  commits should show your e-mail address (GitHub offers a private "noreply" address).
- CI: `ci.yml` (tests, controls, bank; on every push, the bank also weekly), `upstream.yml`
  (controls against upstream `main`; weekly and on demand), `release.yml` (PyPI only on a
  published release; TestPyPI rehearsal on demand). The first run cost about USD 0.20 of
  the private repository's monthly allowance.
- PyPI: **nothing published.** The package builds and passes `twine check --strict`; the
  release workflow publishes with Trusted Publishing and attestations, with a TestPyPI
  rehearsal. Blocked on the final name (`conform` is taken) and on your accounts and
  approval (`docs/publishing_steps.md`, section 3).

## What you need to do personally

1. **Final name, PyPI and TestPyPI accounts, publishing** (only when you approve):
   `docs/publishing_steps.md`.
2. **Second-computer spot check** (closes M0's G4 and M4): pending while you have one
   computer; if you ever borrow another Windows computer for about two hours:
   `docs/g4_second_machine_steps.md`.
3. **Decisions:** (option A for the borderline P1-on-TRL runs was done on 2026-11-06; B or
   C only if a second computer still disagrees); contacting maintainers before public claims; the
   first public artifact; when the repository becomes public. (`bank-ga-fixed` is kept
   until all final verification is complete, by your decision.)

When you bring back the second-computer results, they are compared and recorded, and M0
and M4 are marked complete only if they pass.
