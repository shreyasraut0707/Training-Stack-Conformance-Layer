# DPO scope decision (M4)

**Status: decided by the project owner on 2026-11-01: DPO stays outside the current scope
until there is a confirmed DPO bug** (option B below). The spec lists "a DPO scope decision"
as an M4 item and lists DPO as out of scope for v0.1 (Section 11).

## Evidence

| Source | What it shows | Confidence |
|---|---|---|
| Spec Section 3.2 / 10.1 (QRPO paper, arXiv 2507.08068) | "an extra EOS in the DPO path (still present at writing)" | LOW: the spec's copy was cut off before the appendix; no versions |
| TRL 1.17.1 source (`trl/trainer/dpo_trainer.py`, lines 1114–1121, read on 2026-11-01) | EOS is appended to `chosen` and `rejected` only when the text does not already end with it, and only for non-conversational data | HIGH for the current release |
| This project's bank | No pinned DPO case; no defective version identified | — |

So the one DPO lead is unconfirmed and, for the current release, guarded by an explicit
check. Nothing in the bank could show that a DPO probe detects a real defect.

## Options

| Option | Cost | Value now |
|---|---|---|
| A. Keep DPO out of scope for v0.2 | none | no unverified claims |
| B. Keep it out of v0.2, and add DPO variants of P2 and P3 (chosen and rejected sequences, prompt masking, EOS, template prefix) once the bank has one confirmed, pinned DPO case | a bank search first, then about a week | probes backed by evidence, as the spec requires for every claim |
| C. Implement DPO probes now | about a week | probes with no historical case: only injected-defect evidence, which the spec calls weaker |

## Recommendation

**Option B.** It follows the spec's rule that probes are credible only when the bank shows
they flag a documented defect and stay quiet on the fix. The first step is a search for a
pinned DPO defect (for example the QRPO paper's appendix and TRL's DPO issue history). If
none is found, DPO stays out of scope.
