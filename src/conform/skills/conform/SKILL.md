---
name: conform
description: Check whether a fine-tuning stack (TRL SFTTrainer or the Transformers Trainer) does what the user assumes, before or after an expensive run. Use when a fine-tuned model never stops generating, does worse than the base model, or when the user wants to verify loss masking, gradient accumulation, chat-template consistency, precision or EOS learning, or compare library versions.
---

# conform: training-stack conformance probes

`conform` runs quick executable checks ("probes") of the user's installed training stack on
a tiny CPU proxy model built from their real tokenizer. It never uses their model's weights
or full dataset.

## When to use it

- "My fine-tuned model never stops generating" → `diagnose` with symptom `no-stop`.
- "My fine-tuned model is worse than the base model" → `diagnose` with symptom `worse-than-base`.
- Before a long training run → `run_all` on a recipe.
- After upgrading TRL, Transformers or PEFT → `diff_environments` between the two environments.

## How to run it

1. Write a recipe file (YAML) from the user's training script: `model` (model or tokenizer
   id), `trainer` (`trl` or `transformers`), `data_format`, `trainer_args` (their
   SFTConfig / TrainingArguments settings), `intent` (for example `supervise_eos: true`,
   `mask_prompt: true`, `single_bos: true`) and a few of their rows (`rows` or `rows_file`).
   Ask the user to confirm the intent; do not guess it.
2. Call the MCP tool (`diagnose`, `run_all`, `run_probe`, `list_probes`, `diff_environments`),
   or run the shown `command` with the CLI if MCP is not available. In a confirmation mode,
   show the user the exact `command` field before running it.
3. Open the full report (`report` path) only when the short summary is not enough.

## How to report the results (required)

- Quote every finding with its evidence class: **seen** (observed directly), **inferred**
  (follows from seen facts by a stated rule), **unknown** (could not be determined).
- Read every **unknown** item to the user. Never drop or soften one.
- Never claim harm (for example "this will make the model worse") unless a behavioural probe
  (P7) measured it. P2 facts about masking are mechanisms, not harm; P2's inferred findings
  say "confirm with P7".
- There is no overall pass. Say "N pass, M fail, K not tested" and list what was not tested.
- Give the user the path of the full report (the `report` field).
- A pass means only "these listed behaviours held for these inputs on these versions".
- Give the `reproducer` command for every fail.

## Limits to state when relevant

Proxy models only (no real weights), CPU only (no GPU kernels, DeepSpeed or multi-GPU;
P1M uses two CPU processes). P1 passes up to 2× its measured noise floor and fails above
10×; in between it reports unknown, which small TRL rounding differences can cause. A probe
that cannot run reports `not_tested` with a reason; that is not a pass.
