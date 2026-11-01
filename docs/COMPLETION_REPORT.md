# Training-Stack Conformance Layer: completion report

Date: 2026-11-01 · Version: 0.2.0.dev0 · Source of truth:
`Training_Stack_Conformance_Layer_Illustrated_Spec.pdf` · Detailed status: `docs/PROGRESS.md`

## Bottom line

**The project is not yet complete by the spec's acceptance criteria.** Everything that can
be built and verified on this computer is done and tested (123 tests pass). Three
acceptance checks remain open. The **second-computer spot check** (M0's G4 and M4's
stability threshold) is **pending**: the owner has only one computer, and it has not been
replaced by a virtual machine, by CI or by more runs on this computer. The **runs inside
Claude Code and Codex** (M3) need the owner at the keyboard.

| Milestone | Status |
|---|---|
| M0 Spike | Not complete: conditional go. G1, G2, G3 met; G4 met on this computer, with extra same-computer evidence; the second-computer spot check is pending (one computer only) |
| M1 v0.1 | **Complete** (0.1.0) |
| M2 Bank | **Complete**: results published with misses; CI now runs on GitHub (private repository; first run in `docs/PROGRESS.md`) |
| M3 Agents | Not complete: everything built; clean install and MCP `diagnose` verified; the run inside the two agents is missing |
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
- **Injected defects:** 49 of 49 applicable caught (a first run's 2 misses were a fault in
  the injected defect, published in `docs/bank_results.md`).
- **Notable facts about current libraries:** TRL turns bf16 mixed precision on by default;
  the standard Transformers collator with a PAD = EOS tokenizer trains a model that never
  stops (measured by P7); Transformers isolates packed documents on CPU.

## Tests

123 automated tests pass (122 in the main environment, 1 MCP end-to-end test in
`conform-agent`), covering every probe, its injected defects, the reports, the CLI, the
bank runner, setup and the comparison tool.

## Limitations (stated in every report where they apply)

- Proxy models only: no real weights, so failures that need trained weights are not seen.
- CPU only: GPU kernels, mixed-precision GPU paths, DeepSpeed, FSDP and multi-GPU are not
  tested (P1M covers two CPU processes).
- P1's pass band (2×) lets TRL's float32 kernel rounding read as unknown (by decision).
- Case A's bank pair differs in 22 packages (cases B and C differ only in the library).
- DPO and reinforcement-learning trainers are out of scope (DPO until a confirmed DPO bug).
- Codex's configuration format and skill location are not verified against a running Codex.
- Lockfiles are specific to Windows and CPython 3.11.
- The bank evidence shows the probes behave as specified for the listed cases and versions;
  it does not prove the tool finds unknown bugs or that these defects matter to most users.

## Repository and publishing

- GitHub: **private** repository `shreyasraut0707/Training-Stack-Conformance-Layer`
  (created 2026-11-01), branch `main`; the spec PDF is not in it.
- PyPI: **nothing published.** The package builds and passes `twine check --strict`; the
  release workflow publishes with Trusted Publishing and attestations, with a TestPyPI
  rehearsal. Blocked on the final name (`conform` is taken) and on your accounts and
  approval (`docs/publishing_steps.md`, section 3).

## What you need to do personally

1. **Agent check in Claude Code and Codex** (closes M3): `docs/m3_agent_check_steps.md`.
2. **Final name, PyPI and TestPyPI accounts, publishing** (only when you approve):
   `docs/publishing_steps.md`.
3. **Second-computer spot check** (closes M0's G4 and M4): pending while you have one
   computer; if you ever borrow another Windows computer for about two hours:
   `docs/g4_second_machine_steps.md`.
4. **Decisions:** contacting maintainers before public claims; the first public artifact;
   when the repository becomes public; how to count a pass ↔ unknown change on the three
   borderline P1-on-TRL runs at a second-computer check; keeping or deleting the unused
   `bank-ga-fixed` environment (1.4 GB).

When you bring back the agent outcomes or second-computer results, they are compared and
recorded, and the milestones are marked complete only if they pass.
