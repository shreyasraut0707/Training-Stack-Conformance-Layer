# Changelog

Versions follow the spec's milestones. Evidence for each entry is in `docs/m0_findings.md`
and `bank/results/`.

## 0.2.0.dev0 (M2 to M4 built; M3 and M4 not yet complete): 2026-11-01

Pre-release version: the v0.2 code is in place and tested (123 tests pass), but M4's
stability threshold still needs the second-machine check and M3 needs a run inside the agents.

- Clean install verified from the built wheel in a fresh environment; the skill now ships
  inside the package (`conform/skills/conform/SKILL.md`, kept identical to `skills/`).
- Probe availability now reflects the installed stack (spec Section 9): a probe is "not
  available" when PyTorch, Transformers, Accelerate (or TRL and datasets for P3 and P4) are
  missing, with the reason.
- Hash-pinned lockfiles (spec Section 12.2): `scripts/make_lockfile.py`,
  `bank/envs/conform.lock`, `bank/envs/conform-agent.lock` (verified with `--require-hashes`).
- P4 reads the attention kernel through an adapter and marks private access as fragile
  (spec Section 6).
- Case C confirmed upstream (transformers PR 46204); confidence raised to high.
- `scripts/compare_results.py` for the second-machine spot check; step-by-step guides for
  the agent check and publishing; P1's pass band kept at 2× by decision, fail multiplier 10×
  calibrated.
- Runner names Windows memory-exhaustion exit codes; P1M and P5 use less memory.
- Extra single-computer evidence for G4 (not a second-computer check): a fresh environment
  built from `conform.lock` reproduces every verdict and number; one CPU thread changes 1 of
  72 verdicts (`p1-trl` on SmolLM2, info → pass), with three P1-on-TRL runs recorded as
  borderline near the 2× pass band (`docs/m0_findings.md`). Thresholds unchanged.
- Codex skill location corrected to `~/.agents/skills` (Codex documentation); `diff` test
  uses the running interpreter so it works on any computer.
- Packaging: SPDX license expression (`license = "Apache-2.0"`, `license-files`;
  setuptools ≥ 77), so the build has no deprecation warnings; `twine check --strict` passes.
- Release workflow: `twine check` before upload, and a TestPyPI rehearsal on "Run workflow"
  (environment `testpypi`); real PyPI only on a published release (environment `pypi`).
- CI: time limits on every job, and a newer push cancels the older run (private
  repositories have a limited Actions allowance).

## Earlier work in the same pre-release (M2 to M4)

- **M2:** mutation suite (`conform.mutations`, `scripts/run_mutations.py`) for P1, P2, P3 and
  P7; noise-floor calibration (`conform.calibration`, `scripts/calibrate_noise_floor.py`,
  `docs/calibration.md`); new bank case `eos-masked-pad-eq-eos-2025-behaviour` (P7);
  `scripts/setup_bank_envs.py`; CI workflow (`.github/workflows/ci.yml`); results with misses
  in `docs/bank_results.md`; controls can be limited with `--only` and run with `--python`.
- **M3:** MCP server (`conform.mcp_server`; mcp 1.x and 2.x), skill (`skills/conform/SKILL.md`),
  `conform setup --agent claude-code|codex` (plan by default, `--apply` with backups), release
  workflow with Trusted Publishing and attestations, `SECURITY.md` official-sources notice,
  optional `[mcp]` extra.
- **M4:** P4 packing isolation, P5 resume equivalence, P1M multi-process loss averaging;
  DPO scope recommendation (`docs/dpo_scope_decision.md`).
- Fixes found by testing: P3 matches rows by similarity (a broken prefix defeated prefix
  matching); P5 identifies rows by their full token sequence (a shared system prompt made
  prefixes identical); P7 needs a clear 10× rise to call content "learned" and reloads with
  the proxy's attention kernel; the double-BOS mutant now always makes two leading BOS.

## 0.1.0 (M1, v0.1): 2026-11-01

Exit criterion met: all v0.1 probes pass their own unit tests and run on CPU against the
current stable release (controls: 24 of 24 match; bank rerun: 10 of 10 match).

- P3 (training versus inference tokens): compares the trainer's first real batch with the
  tokenizer's own chat template at inference time; fails on a prefix mismatch.
- P6 (precision): observation-only report of parameter, compute, gradient and optimizer
  dtypes after one real step, marking library defaults (shows TRL's bf16 default).
- P7 (EOS stopping): short real fine-tune, save, reload with the model's generation
  settings, EOS probability before and after, bounded greedy generation.
- `conform` command: `check`, `diagnose`, `diff`, `bank run`, `list-probes`; each probe in
  its own time-limited process; exit codes 0 / 1 / 2 and `--strict`.
- Reports: JSON, Markdown and JUnit XML from one result list.
- Recipes (`examples/sft.yaml`): model, trainer, data format, intent, rows, trainer settings.
- Shared trainer construction (`conform.stack`) used by P2, P3, P6 and P7.
- Bank runner saves each repeat as it finishes and stops at the first unexpected result.

## 0.0.1 (M0 spike)

- P1 (gradient-accumulation equivalence) with an independent float64 oracle and a measured
  noise floor; P2 (loss-bearing tokens) against a declared intent.
- Regression bank with three cases and pinned environments; G3 controls; G4 stability.
