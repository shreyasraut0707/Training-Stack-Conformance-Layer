# Training-Stack Conformance Layer (`conform`, working name)

Quick, executable checks ("probes") that test whether your fine-tuning libraries
(TRL `SFTTrainer`, Transformers `Trainer`) behave the way you assume. They run on a tiny
CPU proxy model built from your real tokenizer, **before** you spend GPU time.

Specification: `Training_Stack_Conformance_Layer_Illustrated_Spec.pdf`.
Progress and milestones: `docs/PROGRESS.md`. Evidence: `docs/m0_findings.md`.

## What a result means

A pass means "these listed behaviours held for these inputs on these versions", nothing
more. Every finding is labelled **seen** (observed directly), **inferred** (follows from
seen facts by a stated rule; never fails on its own) or **unknown** (could not be
determined; never a fail, always shown). There is no overall pass badge: the summary reads
like "4 pass, 1 fail, 2 not tested".

## Install (development)

Python 3.11, in a virtual environment:

```bash
python -m pip install -e ".[stack,dev]"
```

The core imports whatever PyTorch, Transformers and TRL you already have and records their
versions in every report instead of pinning its own.

## Use

Write a recipe (see `examples/sft.yaml`): the model or tokenizer id, the trainer, your
trainer settings, what you intend, and a few rows. The tool never needs the full dataset or
the model's weights.

```bash
conform check --recipe examples/sft.yaml --download
conform diagnose --symptom no-stop --recipe examples/sft.yaml
conform diff --from bank-eos-defective --to bank-eos-fixed --recipe examples/sft.yaml
conform bank run
conform list-probes
```

`--download` fetches only the model's config and tokenizer files (never weights); without
it everything runs offline. Each probe runs in its own time-limited process. Reports are
written as `report.json` (agents), `report.md` (people) and `report.xml` (JUnit, for CI).

**Exit codes:** 0 when no probe fails; 1 when any probe fails (with `--strict`, also when
any probe is not tested); 2 for a usage or internal error.

## Probes

| ID | Question | Fails when | Release |
|---|---|---|---|
| P1 | Does accumulating k micro-batches match one batch of the same samples? | loss or update deviates from an independent float64 reference beyond the measured noise floor | v0.1 |
| P2 | Which tokens carry loss, and does that match the declared intent? | only against a declared intent (`supervise_eos`, `mask_prompt`, `single_bos`) | v0.1 |
| P3 | Are the tokens used in training the same tokens inference would see? | the inference prompt is not an exact prefix of the training sequence | v0.1 |
| P6 | What precision does training actually use? | never (observation only; marks library defaults) | v0.1 |
| P7 | After a short fine-tune, does the saved and reloaded model learn to stop? | EOS probability does not rise, or generation does not stop | v0.1 |
| P4 | Does one packed document leak into another? | editing document A changes document B's loss (not tested without a kernel that isolates packed documents) | v0.2 |
| P5 | Does a resumed run match an uninterrupted run? | final parameters, learning-rate trajectory or data order differ | v0.2 |
| P1M | Is the loss averaged correctly across processes (2 CPU processes, gloo)? | the averaged update deviates from the float64 reference over all rows | v0.2 |

## Use with a coding agent

The MCP server (`python -m conform.mcp_server`, stdio) offers `list_probes`, `run_probe`,
`run_all`, `diagnose` and `diff_environments`, returning short summaries plus the report
path. It needs the optional `mcp` package (`pip install -e ".[mcp]"`). The skill in
`skills/conform/SKILL.md` tells the agent when to use it and how to report results
(quote evidence classes, read every unknown aloud, claim harm only when P7 measured it).

```bash
conform setup --agent claude-code
conform setup --agent codex
```

`setup` only prints the planned changes; add `--apply` to write them (existing files are
backed up). Claude Code: a project `.mcp.json` entry and a project skill. Codex: an
`[mcp_servers.conform]` table in `~/.codex/config.toml` and a skill (the Codex locations are
not yet verified against a running Codex).

## Regression bank

`bank/cases/` holds documented defects with a defective and a fixed pinned environment
(exact versions in `bank/envs/`). The tool is credible only if its probes flag the
defective pin and stay quiet on the fixed one; `bank/controls.yaml` checks that current
releases raise no false alarms. Run with `conform bank run` or `scripts/run_bank.py`.

## Limits

Proxy models (real weights are never used), single-process CPU, no GPU kernels or
distributed backends. Probes execute the installed training code: run them in a
throwaway environment when the recipe is untrusted. `trust_remote_code` stays off.

## Licence

Apache-2.0
