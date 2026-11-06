# P1 noise-floor calibration (M2)

Spec Sections 7.2 and 10.2 (step 3): the multiplier is set on synthetic injected defects
and confirmed on held-out historical cases, so the real accumulation case is not used both
to tune and to judge. Data: `bank/results/calibration_2026-11-01_110130/calibration.json`
(`python scripts/calibrate_noise_floor.py`).

## What was measured

Each P1 check's deviation from the float64 reference, divided by its measured floor.
P1 fails when any check is clearly above the floor, so each run is summarised by its
largest ratio.

**Correct stacks** (transformers 5.22.1, trl 1.17.1; Qwen2.5, SmolLM2, TinyLlama; seeds 0–4;
30 runs):

| Trainer | Largest ratio seen |
|---|---|
| Transformers Trainer | 1.06× |
| TRL SFTTrainer | 3.82× (TRL's float32 log-prob kernel, see `docs/m0_findings.md`) |

**Injected defects** (the loss scaled by 1 + ε; SmolLM2; both trainers):

| ε | Weakest detection (ratio) |
|---|---|
| 10⁻³ | 8,388× |
| 10⁻⁴ | 839× |
| 10⁻⁵ | 84× |
| 10⁻⁶ | 7.7× |

**Held-out historical cases** (not used above): case A, the 2024 accumulation bug, sits at
about 1.4 × 10⁶× (update off by 63–65%); case C, the position-0 count, at about 2.8 × 10⁵×
(off by 1/30). Both are detected by any multiplier up to about 10⁵.

## Result

| Threshold | Provisional value | What the data says |
|---|---|---|
| Fail multiplier | 10× | **Confirmed.** 2.6× above the worst correct stack (3.82×); detects normalisation errors down to about 10⁻⁵ relative; both held-out historical cases detected by orders of magnitude |
| Pass band | 2× | **Too tight for TRL.** Correct TRL runs reach 3.82×, so they can read as "unknown" (never fail). Transformers runs stay below 1.06× |

## Decision needed (thresholds are not changed without the owner)

1. **Keep 2× / 10×** (no change): no false fails; TRL's correct accumulated update can read
   as unknown, as recorded today.
2. **Pass band 5×, fail multiplier 10×**: TRL's correct runs would pass (1.3× margin over
   the largest seen); detection is unchanged, because only the fail line decides a fail.

Either way the fail multiplier stays 10×. Until a choice is recorded, P1 keeps 2× / 10× and
its reports keep saying the values are provisional.

## What changing the pass band from 2× to 5× would and would not do

The pass band only decides between **pass** and **unknown** (status `info`). Only the fail
line (10×) decides a **fail**, and it does not change.

| | Keep 2× | Change to 5× |
|---|---|---|
| False alarms (a correct stack reported as fail) | none seen; decided by the 10× line | **unchanged** |
| Defects reported as fail | every deviation above 10× (normalisation errors down to about 10⁻⁵) | **unchanged** |
| Correct TRL runs (up to 3.82× seen) | often "unknown" (P1 on TRL with Qwen and SmolLM2, every repeat) | pass; margin over the largest seen is only 1.3×, so other seeds, tokenizers or versions may read as unknown again |
| A real defect between 2× and 5× of the floor (a normalisation error of roughly 3–6 × 10⁻⁷ relative) | reported as **unknown** (visible) | reported as **pass** (a false negative), but far below any historical case (10⁵× and above) |
| Honesty rule (spec: prefer "not tested" to a guess) | the floor does not model TRL's float32 kernel, and the report says so | treats TRL's unmodelled rounding as noise on the strength of a diagnosis, not a measurement in the floor |

**Advantages of 5×:** cleaner reports on TRL (pass instead of unknown); less alarm fatigue;
no change to what fails.
**Disadvantages of 5×:** a tiny loss of sensitivity in the pass/unknown boundary (a defect
of that size would read as pass rather than unknown); a narrow 1.3× margin; the threshold
moves because of one library's rounding rather than a better floor.

A third option would remove the trade-off: make the floor itself sample the trainer's own
rounding (for example a second float32 run through the same kernel), so TRL's noise is
measured rather than allowed for. That is a probe-design change for a later milestone.

**Recommendation:** keep 2× for now (no false alarms either way; unknowns stay visible and
explained), and revisit with the third option. The decision is the owner's.

## Decision (2026-11-01)

**The owner chose to keep the pass band at 2× and the fail multiplier at 10×.** The fail
multiplier is confirmed by this calibration; the pass band is kept deliberately, so TRL's
unmodelled float32 rounding continues to read as unknown rather than pass. Revisit when the
floor can sample the trainer's own kernel rounding.

## 2026-11-06: a wider floor (option A; thresholds unchanged)

The "third option" above was taken up, in a form that keeps the floor independent of the
library. Sampling the floor through TRL's own fused log-prob head was tried and **rejected**:
it is bit-for-bit the trainer's own computation (every ratio exactly 1.00×), so it could
never see anything wrong with it. Instead P1's floor now also samples, with the oracle's own
float32 maths, the batchings the trainers use (unpadded rows, all rows in one batch) and
every float32 sample at 1, 2, 4 and the default number of CPU threads (details:
`docs/m0_findings.md`, "Option A"). Data: `bank/results/calibration_2026-11-06_223241/`.

| | Previous floor | Wider floor |
|---|---|---|
| Correct Transformers runs, largest ratio | 1.06× | **0.82×** |
| Correct TRL runs, largest ratio | 3.82× | **2.10×** (one run; the rest below 2×) |
| Injected loss-scale defects | ε = 10⁻⁵ at 84×, 10⁻³ at 8388× | **the same** (caught through the loss check, whose floor is float32's resolution) |
| Mutation suite | 49 of 49 applicable | **49 of 49** |
| Historical cases A and C | fail at the defective pins | **fail**; fixed pins pass |

The pass band (2×) and fail multiplier (10×) are unchanged. Margin of the worst correct
stack under the fail line: 4.8× (was 2.6×). TRL's kernel still rounds at up to about 2× an
independent floor, so a P1-on-TRL run can still read as unknown near the line.
