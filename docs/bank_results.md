# Regression bank: detection and false-alarm results (M2)

What the bank shows, and what it does not (spec Section 10). Every number links to a
results folder in `bank/results/`. "Defective → fixed" means the probe must fail at the
defective pin and not fail at the fixed pin; a fail on a current release is a false alarm.

## Historical cases (detection and silence)

| Case | Probe | Defective pin → status | Fixed pin → status | Repeats | Confidence |
|---|---|---|---|---|---|
| A · gradient-accumulation normalisation (Oct 2024) | P1, 2 models | transformers 4.45.2 → **fail** | 5.10.1 → pass | 10 (G4) | high |
| B · SFT masks EOS when PAD = EOS (TRL PR 3200) | P2 | trl 0.16.0 → **fail** | 0.16.1 → pass | 10 (G4) | high |
| B · the same, measured as behaviour | P7 | trl 0.16.0 → **fail** (EOS probability 2×10⁻⁵ → 5×10⁻⁸; stops 0 of 4) | 0.16.1 → pass (→ 0.992; stops 4 of 4) | 1 | high |
| C · position-0 label counted in `num_items_in_batch` (found by P1; fixed upstream by PR 46204) | P1, 2 models | transformers 5.9.0 → **fail** | 5.10.1 → pass | 10 (G4) | high |

Results: `2026-11-01_094057/` (G4, cases A, B, C), `2026-11-01_111720/` (B behaviour).
**Detected: 4 of 4 case checks. Fails at fixed pins: 0.**

Limitation: case A's two environments differ in 22 packages (torch 2.4.1 → 2.15.0 among
them); cases B and C differ only in the library under test. See `docs/m0_go_no_go.md`, 4.1.

## False alarms on current releases (controls)

`bank/controls.yaml`, transformers 5.22.1 and trl 1.17.1, three tokenizer families.

| Control set | Runs | Fails | Results |
|---|---|---|---|
| v0.1 (P1, P2, P3, P6, P7) | 24 | **0** (2 P1-on-TRL runs read as unknown) | `controls_2026-11-01_104321/` |
| M0 set (P1, P2) × 10 repeats | 120 | **0**; identical verdicts in every repeat | `controls_2026-11-01_091556/` |
| v0.2 set (P4, P5 on both trainers, P1M) × 10 repeats | 120 | **0**; identical verdicts in every repeat | `controls_2026-11-01_171529/` |
| v0.1 + v0.2 in a fresh environment built from `conform.lock` (same computer) | 36 | **0**; verdicts and numbers identical to the primary runs | `controls_2026-11-01_212348/`, `_212926/` |
| v0.1 + v0.2 on one CPU thread (same computer) | 36 | **0**; 35 of 36 verdicts identical; `p1-trl` on SmolLM2 info → pass | `controls_2026-11-01_213238/`, `_213842/` |

| v0.1 + v0.2 on GitHub CI (Ubuntu VM, pinned stack; and again with the latest releases) | 36 + 36 | **0**; 35 of 36 verdicts identical to this computer; `p1-trl` on TinyLlama pass → info | CI run 41326929545 (artifacts) |

The bank itself also ran on GitHub's Windows runner: **12 of 12 verdicts identical** to this
computer (CI run 41326929545).

The last three rows are extra evidence from the same computer or from cloud virtual
machines, **not** the second-computer spot check (still pending). They show that the thread count moves P1-on-TRL's numbers
across the 2× pass band for some models, never near the 10× fail line (details in
`docs/m0_findings.md`, "Extra single-computer evidence for G4").

By decision, controls declare only documented behaviour: the Transformers collator's
PAD = EOS masking is recorded as an observation (it fails P2's `supervise_eos` and P7 by
design), not counted as a false alarm.

## Misses and cases not covered (published, as the spec requires)

| Item | Status | Why |
|---|---|---|
| Transformers 4.46.3 as case A's fixed pin | **P1 fails it** | not a false alarm: it still has case C's defect (verified in its source), so case A's fixed pin moved to 5.10.1 |
| Ignored EOS label in the TRL 0.13 completion-only path (spec lead) | not pinned | seen only in a search snippet; affected versions unknown |
| Double BOS in a recent TRL path (spec lead) | not pinned | versions unknown; P3 and P2 detect the defect when it is injected (mutation suite) |
| Extra EOS in the DPO path (spec lead) | out of scope | see `docs/dpo_scope_decision.md`; guarded in TRL 1.17.1 |
| EOS masking in an on-policy helper, TRL 1.7–1.8 (spec lead) | out of scope | reinforcement-learning trainers are out of scope; low-confidence lead |
| P4, P5, P1M | no historical case | evidence is injected defects only (weaker), as the spec anticipated |

## Injected defects (mutation suite; weaker evidence)

`python scripts/run_mutations.py`, three tokenizer families.

- First run (`mutations_2026-11-01_110630/`): **49 of 51 caught**. The 2 misses were the
  double-BOS mutant on Qwen (no BOS token: the mutant prepended an EOS) and TinyLlama (its
  template has no BOS, so the mutant made a single BOS, which `single_bos` allows): a fault
  in the mutant, not in P2. P3 caught the same mutant on all three tokenizers.
- After correcting the mutant (always at least two leading BOS; not applicable without a BOS
  token), full rerun (`mutations_2026-11-01_112630/`): **49 of 49 applicable caught**, 2 not
  applicable (double BOS on Qwen, which defines no BOS token).
- Further mutants in the tests: P5 (optimizer and scheduler not restored; data restarted on
  resume), P4 (document boundaries withheld from the model), P1M (per-process normalisation).

## Calibration

`docs/calibration.md`: the fail multiplier (10×) is confirmed; the pass band (2×) is too
tight for TRL's correct runs (up to 3.82×) and awaits the owner's decision.

## What the bank does not prove

That the tool finds unknown bugs, or that these defects matter to most users. It shows only
that, for the listed cases and versions, the probes behave as specified (spec Section 10.2).
