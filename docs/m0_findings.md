# M0 findings log

## 2026-10-31 · P1 on TRL SFTTrainer: accumulated update reads as "unknown"

**Environment.** Python 3.11.9 · torch 2.15.0+cpu · transformers 5.22.1 · trl 1.17.1 ·
CPU · proxy models Qwen/Qwen2.5-0.5B-Instruct (7ae5576) and
HuggingFaceTB/SmolLM2-135M-Instruct (12fd25f), seed 0 unless stated.

**Observation (seen).** P1's accumulated-update deviation, as a multiple of the
measured floor (pass ≤ 2×, unknown 2–10×, fail > 10×):

| Model   | Transformers Trainer | TRL SFTTrainer |
|---------|----------------------|----------------|
| Qwen    | 0.98×                | 3.68×          |
| SmolLM2 | 0.99×                | 2.27×          |

All loss checks and all one-batch checks were inside the pass band on both trainers.

**Diagnostics.**
1. *Float64 run.* The Transformers gap fell from 4.8e-7 to 5.5e-8; the TRL gap stayed
   at 1.8e-6. Cause: `trl/kernels/chunked_logprob.py` computes in float32 regardless of
   model dtype (the Transformers loss also upcasts to float32 via `logits.float()`).
2. *Trainer-style floor variant* (one backward per micro-batch, float32): 4.9e-7, the
   same as the existing floor, so accumulation order does not explain the gap.
3. *Direction check, 3 seeds × 2 models.* Cosine between (trainer − reference) and the
   reference gradient: TRL −0.09 to +0.10, Transformers −0.03 to +0.12; norm error
   about 1e-7. A scale or normalisation defect would give a cosine near ±1.
   Reproduce with `scripts/diagnose_p1_noise.py`.

**Conclusion (inferred).** Noise-like rounding in TRL's float32 kernel, about 2–4×
the Transformers path and about 1000× smaller than the injected mean-of-means defect
(about 1e-3). Rule: a difference that does not align with the gradient and has
no consistent sign across seeds is not a scaling error.

**Decision.** Thresholds unchanged (no tuning to fit a result, spec Section 14). P1
reports this as unknown with a stated limitation. Input for the M2 calibration: the
pass band must account for trainer-specific kernels; TRL's accumulated update noise
is about 1.0–1.8e-6 on these proxies.

## 2026-10-31 · P2: the standard Transformers collator masks EOS when PAD = EOS

**Environment.** As above. Tokenizer HuggingFaceTB/SmolLM2-135M-Instruct, where the
pad and EOS tokens are both id 2 (`<|im_end|>`). Four built-in prompt-completion
conversations. Intent declared: `supervise_eos=true`, `mask_prompt=true`.

**Observation (seen).**

| | Transformers Trainer + `DataCollatorForLanguageModeling(mlm=False)` | TRL SFTTrainer |
|---|---|---|
| Final EOS carries loss | **0 of 4 rows** | 4 of 4 rows |
| Prompt tokens carrying loss | 128 of 140 | 0 of 140 |
| Real tokens sharing the pad ID / of them without loss | 12 / 12 | 12 / 8 (the prompt's) |
| P2 status | fail (both intents) | pass |

Reproduce with
`python -m conform.probes.p2_loss_tokens HuggingFaceTB/SmolLM2-135M-Instruct --trainer transformers --intent supervise_eos=true --intent mask_prompt=true`.

**Mechanism (read from source).** transformers 5.22.1 `data_collator.py` line 794 sets
`labels[labels == pad_token_id] = -100`, which also masks every real EOS when the two
ids are equal. TRL builds labels from a completion mask and pads labels separately, so
it is not affected. On Qwen2.5 (pad ≠ EOS) the Transformers collator supervises EOS.

**Classification.** Documented behaviour that surprises users, not a confirmed bug
(spec Section 10.2, step 5): the collator does what its code says, and forum threads
report the "never stops" symptom from this setup. Harm is not claimed; P2 adds the
inferred finding "confirm with P7".

**Open question for the go/no-go memo.** Gate G3 requires zero fails on current
releases. Decide whether G3's control runs declare `supervise_eos` for the
Transformers collator path, since with a PAD = EOS tokenizer it fails by design.

## 2026-11-01 · Regression bank, first complete run: 10 of 10 runs match

**Run.** `python scripts/run_bank.py`, results in `bank/results/2026-11-01_003020/`,
exit code 0. Each probe ran in its own pinned environment (exact versions in
`bank/envs/`), offline, as a separate time-limited process.

| Case | Defective pin → status | Fixed pin → status | Reason given by the probe |
|---|---|---|---|
| A · ga-normalisation-2024 (P1, 2 models) | transformers 4.45.2 → **fail** | 5.10.1 → pass | "Accumulating micro-batches does not give the same result as one large batch" |
| B · eos-masked-pad-eq-eos-2025 (P2) | trl 0.16.0 → **fail** | trl 0.16.1 → pass | EOS carries loss in 0 of 4 rows → 4 of 4 rows |
| C · position-zero-label-count (P1, 2 models) | transformers 5.9.0 → **fail** | 5.10.1 → pass | "The loss normalisation differs from the reference in both runs, so the cause is not accumulation itself" |

**Case A detail (seen).** On 4.45.2 the accumulated loss is off by only 0.20–0.25%
while the accumulated update is off by 63–65%; the one-batch run is inside the floor.
Mechanism: each row's tokens are weighted 1/(4 × its token count) instead of 1/29, so
the 3-token row counts 2.4× and the 14-token row about 0.5×. A loss curve would look
normal, which is the spec's "silent" defect.

**New finding, case C (seen; source read).** From the 4.46 line (the October 2024 fix)
to 5.9.0, `num_items_in_batch` counts every label that is not -100, including position 0,
which a causal LM never predicts. P1's fourth row has no prompt, so the count is 30
instead of 29 and loss and update are both scaled by 29/30 (deviation exactly 0.0333 in
all four checks). 5.10.1 counts `labels[..., 1:]` "to avoid over-counting position 0".
5.10.0 is yanked on PyPI ("pushed from a week old main branch"). Not in the spec's list;
the pull request is not yet found, so confidence is medium. Harm is not claimed: the
effect is a uniform down-scaling of the update for rows whose first label is set.

**How the bank definition changed (for the go/no-go memo).** Case A's fixed pin was
first 4.46.3. P1 failed it, for case C's reason (its accumulated and one-batch runs
agreed with each other, both off by 1/30). The fixed pin was moved to 5.10.1, the first
release with both fixes, and case C was added. No threshold was changed. Case A's pair
now differs in more than one library (torch 2.4.1 → 2.15.0, transformers and their
dependencies); case B and case C pairs differ only in the library under test. The
4.46.3 environment (`bank-ga-fixed`) is no longer used by any case; its exact package
list stays in `bank/envs/bank-ga-fixed.txt`.

**Problems met on the way (all reported as `not_tested`, never as a wrong verdict).**
1. P1 and P2 set `data_seed`, which transformers 4.46.3 refuses with accelerate < 1.1.
   It was redundant with `seed`, so it was removed from both probes.
2. The 5.x environments lacked `accelerate`, which `transformers` does not install but
   the Trainer needs. Added (1.15.0, same in both), and every bank environment now passes
   a check that the Trainer's settings object can be created.

**Observation, case B (seen; source read).** TRL 0.16.x appends an EOS when a row does not
already end in one (`sft_trainer.py`, lines 459–462 in 0.16.0), and SmolLM2's template
ends each turn with `<|im_end|>` and a newline, so every row ends `<|im_end|>\n<|im_end|>`.
P2 therefore reports the prompt boundary as unknown in that case. Not claimed as a
defect; a question for P3 (M1).

**Gate status after this run.** G1 (at least 2 cases detected at pre-fix pins): met,
3 of 3. G2 (zero fails at fixed pins): met, 0 fails. G3 (controls) and G4 (stability):
not yet run.

## 2026-11-01 · G3 controls and G4 stability

**G3 controls (seen).** `bank/controls.yaml`: current releases (transformers 5.22.1,
trl 1.17.1, torch 2.15.0+cpu) in the main environment, three tokenizer families
(Qwen2.5 BPE, SmolLM2 GPT-2-style BPE, TinyLlama Llama-2 SentencePiece), four control
runs each. Intents are only those each stack documents (decision recorded in the file):
the Transformers collator path declares `single_bos` only, so its PAD = EOS behaviour
(entry above) stays an observation, not a control.

| Control | Qwen2.5 | SmolLM2 | TinyLlama |
|---|---|---|---|
| P1 · Transformers Trainer | pass | pass | pass |
| P1 · TRL SFTTrainer | info | info | pass |
| P2 · collator, `single_bos` | pass | pass | pass |
| P2 · TRL, `supervise_eos` + `mask_prompt` + `single_bos` | pass | pass | pass |

Zero fails in 12. The two `info` results are the TRL float32 kernel noise in the first
entry (unknown, never a fail). First single run: `bank/results/controls_2026-11-01_003908/`.

**G4 stability (seen).** Restarted from the beginning after the interrupted attempt
recorded in `docs/g4_stop_record.md`, with each repeat now saved as it finishes.
Started 09:15:56, finished 09:59:44 (43.8 minutes), both stages exit code 0.

| Stage | Results | Repeats | Runs per repeat | Runs matching | Distinct verdicts per run | Max deviation spread |
|---|---|---|---|---|---|---|
| Controls | `bank/results/controls_2026-11-01_091556/` | 10 of 10 | 12 | 120 of 120 | 1 for all 12 | 0 |
| Bank | `bank/results/2026-11-01_094057/` | 10 of 10 | 10 | 100 of 100 | 1 for all 10 | 0 |

Every status and every finding's evidence class and verdict were identical in all 10
repeats, and every P1 deviation was bit-for-bit identical (spread 0): on this machine,
CPU float32 runs are deterministic. Repeat folders were saved progressively (controls
09:18:12 to 09:40:57, bank 09:44:05 to 09:59:44), confirming per-repeat saving in the real run.

**Gate status.**

| Gate | Status |
|---|---|
| G1 detection | met: 3 of 3 cases |
| G2 silence on fixed pins | met: 0 fails |
| G3 controls | met: 0 fails in 12, three tokenizer families |
| G4 stability | 10-repeat requirement met on this machine (220 of 220 runs identical); **the second-machine spot check the spec requires has not been done** |

**Not tested.** Second machine (G4 spot check); GPU, mixed precision and distributed
paths; P3 to P7; real model weights; the noise-floor pass band (2×) and fail multiplier
(10×) remain provisional until the M2 calibration.

## 2026-11-01 · M1 (v0.1): P3, P6 and P7 on current releases

**Controls (seen).** `bank/controls.yaml` extended with P3 (TRL), P6 (both trainers) and
P7 (TRL): 24 of 24 runs match (`bank/results/controls_2026-11-01_104321/`). The bank
rerun with the shared trainer module (`conform.stack`): 10 of 10 match.

**P6: TRL turns bf16 mixed precision on by default (seen; source read).** On the same
proxy, the Transformers Trainer computes in float32 and TRL's SFTTrainer in bfloat16, with
parameters, gradients and optimizer state in float32 in both. TRL's config sets
`bf16 = not fp16` when bf16 is unset (`trl/trainer/base_config.py`, line 108). Reported as
a library default, not a defect.

**P7: the PAD = EOS collator behaviour is measured as harm (seen).** With the standard
`DataCollatorForLanguageModeling(mlm=False)` and a PAD = EOS tokenizer, the model learns
the completions (their token probability rises to about 0.98) but the EOS probability
falls (SmolLM2: 2×10⁻⁵ → 5×10⁻⁸; TinyLlama: 4×10⁻⁵ → 1×10⁻⁶) and greedy generation never
stops (0 of 4 prompts). With Qwen2.5 (PAD ≠ EOS) and with TRL on all three tokenizers,
EOS probability rises to about 0.99 and generation stops in 4 of 4. This confirms P2's
inferred finding for that setup; it stays outside the G3 controls (documented behaviour).

**P3 (seen).** On current TRL the inference prompt is an exact prefix of the training
sequence on all three tokenizers and both data formats; an injected extra BOS fails.
A first version of P3 matched rows to samples by shared prefix, which an extra BOS
defeats (it reported unknown); rows are now matched by overall similarity.

## 2026-11-01 · M4 stability: first attempt stopped by memory exhaustion

**Observation (seen).** The first 10-repeat run of the M4 controls
(`bank/results/controls_2026-11-01_111850/`) matched 12 of 12 in repeats 1 and 2. In
repeat 3, all six P5-on-TRL and P1M runs crashed with Windows exit codes 0xC000012D
("commit limit reached": out of virtual memory) and 0x40010004 (terminated by the
system); the runner recorded them as `not_tested`, flagged the changed verdicts and
stopped, keeping the three completed repeats. P4 and P5-on-Transformers stayed identical.

**Cause (inferred, rule: the exit codes name memory exhaustion).** This laptop has 7.3 GB
of RAM, much of it used by other applications; P1M runs three Python processes that each
load PyTorch and Transformers, and P5 trains four times in one process. Not a probe defect,
but it means stability was not demonstrated by that run.

**Changes.** P1M builds its reference proxy only after its two workers finish; P5 frees
each run's trainer before the next; the runner names Windows memory-exhaustion exit codes.
The 10-repeat run was then repeated with no other job of this project running: 120 of 120
runs matched, with identical verdicts in every repeat (`bank/results/controls_2026-11-01_171529/`).

## 2026-11-01 · Case C confirmed upstream; name availability

**Case C (seen; page read).** The fix is huggingface/transformers PR 46204, "Fix
`num_items_in_batch` over-counting for causal LM losses", merged 2026-05-27 and listed under
"Bugfixes and improvements" in the v5.10.1 release notes. A maintainer fixed it, so under spec
Section 10.2 (step 5) it is a confirmed defect, not "documented behaviour that surprises
users". Confidence raised from medium to high. v5.10.0 was yanked ("published on a corrupted
branch"), which is why the pair is 5.9.0 → 5.10.1.

**Name (seen; pages read, spec Section 14.1).** `conform` is taken on PyPI (a 2019 project,
"A simple config generator") and as a GitHub user account; npm could not be checked (HTTP
403). `training-stack-conform` and `conform-probes` are free on PyPI today (a free name is
not reserved). The package keeps the working name until the owner chooses.

## 2026-11-01 · Extra single-computer evidence for G4 (not a second-computer check)

**Why.** The owner has one computer, so the second-machine spot check (G4, and M4's
stability threshold) cannot be done. These runs test two of the things a second computer
would change (a freshly built environment, and a different CPU thread count), **on the
same computer**. They strengthen the evidence but do not replace the spot check, which
**stays pending**.

**Setup (seen).** This computer: AMD Ryzen 5 5600H, 6 cores, 12 threads, 7.3 GB RAM.
(a) `conform-verify`: a new environment built only from `bank/envs/conform.lock` with
`pip install --require-hashes`; its `pip freeze` matches `bank/envs/conform.txt` exactly
(58 of 58 packages). (b) The main environment with `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1` (one CPU thread instead of the default). Run one at a time, nothing
else running. Compared by verdict with `scripts/compare_results.py`.

| Run | Results | Compared with | Verdicts |
|---|---|---|---|
| A1 fresh lockfile environment, v0.1 controls | `controls_2026-11-01_212348/` | `controls_2026-11-01_104321/` | 24 of 24 identical |
| A2 fresh lockfile environment, M4 controls | `controls_2026-11-01_212926/` | `controls_2026-11-01_171529/` | 12 of 12 identical |
| B1 one CPU thread, v0.1 controls | `controls_2026-11-01_213238/` | `controls_2026-11-01_104321/` | 23 of 24 identical; `p1-trl` on SmolLM2 info → pass |
| B2 one CPU thread, M4 controls | `controls_2026-11-01_213842/` | `controls_2026-11-01_171529/` | 12 of 12 identical |

All 72 runs met their control expectation (`not_fail`): **0 fails**.

**The fresh environment reproduces the numbers bit for bit (seen).** Every P1 deviation in
A1 equals the primary run's, so the lockfile rebuilds the measured environment exactly.

**Thread count moves TRL's P1 numbers, and three runs sit near the 2× line (seen).**
Largest deviation of any P1 measurement, as a multiple of its noise floor (pass up to 2×,
unclear up to 10×, fail above 10×):

| Run | Default threads | One thread |
|---|---|---|
| `p1-transformers`, all three models | 0.99× to 1.00× (pass) | the same |
| `p1-trl`, SmolLM2 | 2.27× (unclear: info) | 1.69× (pass) |
| `p1-trl`, TinyLlama | 1.96× (pass) | 1.66× (pass) |
| `p1-trl`, Qwen2.5 | 3.68× (unclear: info) | 2.43× (unclear: info) |

The SmolLM2 change is the accumulated run's parameter update: 1.02×10⁻⁶ with the default
threads, 7.58×10⁻⁷ with one, against a floor of 4.49×10⁻⁷. Losses did not change.

**Interpretation (inferred, rule: only the thread count differed between B1 and the
primary run, and only the TRL runs moved).** The number of CPU threads changes the order
in which floating-point sums are added. TRL's chunked float32 log-prob kernel is rounding
the noise floor does not sample (P1's stated TRL limitation), so its share of the gap
moves with that order. This is the TRL kernel rounding already recorded under "Diagnosis"
above, not a defect. On one computer with a fixed thread count the results are
deterministic (10 repeats, spread 0); across thread counts they are not.

**What this means for the pending second-computer check (inferred).** A computer with a
different number of cores may move `p1-trl` on SmolLM2 or TinyLlama between pass and info
(unknown), and possibly Qwen2.5 from info to pass. That would show up as `DIFFERENT` in
`scripts/compare_results.py`, although none of these runs is near the fail line (largest
3.68×, fail above 10×). The check's criterion (identical verdicts) and the thresholds are
**not changed**; this is recorded now, before any second-computer run, so that such a
difference is not explained after the fact. Whether a pass ↔ info change on these three
runs counts against G4 is the owner's decision (`docs/PROGRESS.md`, open decisions).

**Gate status unchanged.** G4: 10 repeats identical on this computer; the second-machine
spot check **has not been done** (one computer only). M4's stability threshold: the same.

## 2026-11-01 · First GitHub CI run (cloud virtual machines, not a second computer)

**Run (seen).** `ci` run 41326929545 on the private repository, commit `e51c4a3`: all jobs
**success**. `tests (pinned)` 8 min 43 s, `tests (latest)` 11 min 5 s (both Ubuntu,
`Linux-7.0.0-1017-azure-x86_64`, Python 3.11.17), `bank` 9 min 56 s (Windows). The
`latest` job resolved to the same versions as the pinned stack (transformers 5.22.1,
trl 1.17.1, torch 2.15.0+cpu), so it is not yet evidence about newer releases. Artifacts
compared on this computer with `scripts/compare_results.py`.

| Job | Runs | Fails | Verdicts vs this computer |
|---|---|---|---|
| tests (pinned): pytest | 122 tests (MCP test skipped: no `mcp` in the pinned stack) | 0 | — |
| tests (pinned): G3 controls | 36 | **0** | 35 of 36 identical; `p1-trl` TinyLlama pass → info |
| tests (latest): G3 controls | 36 | **0** | 35 of 36 identical; the same run |
| bank (Windows) | 12 | 0 unexpected | **12 of 12 identical** (6 defective pins fail, 6 fixed pins do not) |

Baselines: controls `controls_2026-11-01_104321/` plus `controls_2026-11-01_171529/repeat-01/`;
bank `2026-11-01_104918/` plus `2026-11-01_111720/`.

**The predicted borderline run moved (seen).** Largest P1 deviation as a multiple of its
floor: `p1-trl` SmolLM2 2.27× here → 2.92× on CI (info both); Qwen2.5 3.68× → 3.32× (info
both); TinyLlama **1.96× → 3.55×** (pass → info). Transformers' P1 and P1M stayed between
0.98× and 1.05× (pass). No run came near the 10× fail line. This is the change recorded in
the section above **before** any other computer was used, and it is the TRL kernel rounding
described there.

**What this is and is not.** GitHub's runners are cloud virtual machines with a different
operating system and CPU. They show the verdicts that matter (detections, zero fails at fixed
pins, zero fails in controls) hold off this computer. They are **not** the physical second
computer the owner set as G4's requirement, so G4 and M4's stability threshold stay
**pending**. If a pass ↔ info change on the three borderline `p1-trl` runs were counted
against G4, this comparison would not meet "identical verdicts"; that decision is the
owner's (`docs/PROGRESS.md`, open decision 5).

**First run against upstream main (seen).** `upstream` run 41328069985 (its own workflow,
started by hand, commit `581d4a0`): **success**, 6 min 32 s, transformers 5.23.0.dev0 and
trl 1.18.0.dev0 from `main`. 36 control runs, **0 fails**; verdicts identical to the CI runs
on the current releases (35 of 36 identical to this computer, the same borderline `p1-trl`
TinyLlama run). No upstream change needs triage today.

## 2026-11-01 · G4 against the spec's text: the borderline P1-on-TRL runs

**What the spec says (read in the PDF).**
- Section 14 (go / no-go table), G4: "Identical verdicts across 10 repeated runs on one machine and a second
  spot check" ("Flaky verdicts destroy trust").
- Section 10.2, step 2: "Repeat every probe at least ten times on two machines in
  deterministic CPU float32 mode. Any verdict that changes counts as a defect in the probe."
- Figure 7: three outcomes. Inside the floor is pass, clearly above is fail, and in
  between is "unclear: report as unknown". So unknown is a verdict of its own.

**Reading.** Under that text, `p1-trl` moving between pass and unknown across machines (CI,
cloud VMs) or across thread counts (this computer) is a **verdict change, and so a defect in
the P1-on-TRL probe**, even though it never comes near a fail. Transformers' P1 and P1M did
not move (0.97× to 1.05× everywhere). Nothing changes on one machine with a fixed setup
(10 repeats, spread 0), which is why G4's one-machine part is met.

**Options (owner's decision; nothing has been changed).**

| Option | What it means | Effect |
|---|---|---|
| **A. Treat it as a probe defect and fix the probe (recommended)** | P1's own limitation says its noise floor "does not sample" the rounding of TRL's chunked float32 log-prob kernel. Measure the floor through the same TRL loss path, so the floor moves with the kernel on each machine. The 2× pass band and 10× fail multiplier stay as decided | follows the spec's rule; needs a code change, then calibration, mutation suite, controls and the 10 repeats rerun; result not known in advance |
| B. Count pass ↔ unknown as "not a change" | changes G4's comparison rule | contradicts "any verdict that changes"; needs your explicit approval of a criteria change |
| C. Leave P1-on-TRL out of the stability claim | report it as unstable across machines | honest but narrows M0's scope (M0 names P1 on TRL SFT) |

Until decided, the comparison rule and thresholds are unchanged, and every report on these
runs states the borderline.
