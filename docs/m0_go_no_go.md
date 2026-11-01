# M0 go/no-go memo: Training-Stack Conformance Layer

**Date:** 2026-11-01 · **Milestone:** M0 spike · **Evidence:** `docs/m0_findings.md`, `bank/results/`

## 1. Decision

**Conditional go for M1 (option A, chosen on 2026-11-01).** G1, G2 and G3 are met. G4's
10-repeat requirement is met on this machine (220 of 220 runs identical), but the spec's
second-machine spot check has not been done. The spec fixes the thresholds in advance
(Section 14), so G4 counts as met only once that check gives identical verdicts
(Section 5). M1 starts only after that and after explicit approval.

## 2. Gate results (thresholds as fixed in spec Section 14)

| Gate | Threshold | Result | Evidence |
|---|---|---|---|
| G1 Detection | ≥ 2 bank cases flagged at pre-fix pins by the intended probe | **Met** (see 2.1) | `bank/results/2026-11-01_003020/`; every G4 repeat |
| G2 Silence on fixed pins | Zero fails at the matching fixed pins | **Met: 0 fails** | same |
| G3 Controls | Zero fails on current TRL, Transformers and PEFT with three tokenizer families | **Met: 0 fails in 12** | `bank/results/controls_2026-11-01_003908/` |
| G4 Stability | Identical verdicts in 10 repeats on one machine, plus a second spot check | **Partly met:** 10 repeats identical (220 of 220, numeric spread 0); second machine **not done** | `bank/results/controls_2026-11-01_091556/`, `bank/results/2026-11-01_094057/` |

### 2.1 G1 holds without case C

G1 needs at least two cases. The two **high-confidence** cases, A and B, meet it on their
own: each was flagged at its pre-fix pin by its intended probe in the first complete bank
run and in all 10 G4 repeats (P1 fails on transformers 4.45.2 for both models; P2 fails on
trl 0.16.0). Case C (medium confidence, a finding of this project whose pull request is not
yet found) is additional evidence. G1 does not depend on it: removing case C leaves 2 of 2
high-confidence cases detected.

## 3. What the spike built and showed

- **P1 (gradient accumulation)** runs the real Transformers Trainer and TRL SFTTrainer
  against an independent float64 oracle with a measured noise floor. It also compares the
  accumulated and one-batch runs directly, so it tells an accumulation defect (update off by
  63–65% while the loss is off by 0.2%) from a normalisation error that affects both runs.
- **P2 (which tokens carry loss)** reads the first real batch from the trainer's own
  dataloader and fails only against a declared intent.
- **Regression bank:** 3 cases, 5 pinned environments in use (exact versions in `bank/envs/`).

| Case | Probe | Defective → fixed | Confidence |
|---|---|---|---|
| A · gradient-accumulation normalisation | P1 | transformers 4.45.2 → 5.10.1 | high |
| B · SFT masks EOS when PAD = EOS | P2 | trl 0.16.0 → 0.16.1 | high (PR #3200 read) |
| C · position-0 label counted in `num_items_in_batch` | P1 | transformers 5.9.0 → 5.10.1 | medium (code read; PR not found) |

## 4. Things a reviewer should know

1. **Case C is a new finding by this tool**, not in the spec's list. Transformers 4.46 to
   5.9 counts a label a causal LM never predicts. It is already fixed upstream (5.10.1, with
   a comment explaining why), so no report is needed; finding its PR would raise confidence.
2. **Case A's fixed pin was moved** from 4.46.3 to 5.10.1, after 4.46.3 failed P1 for case
   C's reason. The move rests on source code read; no threshold was changed.
3. **Case A's pair is not isolated (limitation, see 4.1).**
4. **G3 counts only documented behaviour.** The Transformers collator masks EOS whenever
   PAD = EOS (SmolLM2, TinyLlama); that is recorded as documented behaviour that surprises
   users, not as a control failure.
5. **TRL's float32 kernel** makes P1 report "unknown" (never fail) for SFTTrainer's
   accumulated update on two tokenizers; the diagnosis shows rounding, not a scaling error.
6. **Problems met on the way** (`data_seed`, a missing `accelerate`, a yanked 5.10.0) all
   appeared as `not_tested`, never as a wrong verdict.

### 4.1 Limitation: case A's environments differ in more than one library

Cases B and C each compare two environments that differ **only** in the library under
test. Case A does not: its defective environment (`bank-ga-defective`, 26 packages) and its
fixed environment (`bank-p0-fixed`, 36 packages) differ in **22 packages**, including:

| Package | Defective | Fixed |
|---|---|---|
| transformers | 4.45.2 | 5.10.1 |
| torch | 2.4.1 | 2.15.0 |
| accelerate | 1.0.1 | 1.15.0 |
| huggingface_hub | 0.36.2 | 1.33.0 |
| tokenizers | 0.20.3 | 0.22.2 |
| numpy | 1.26.4 | 2.4.6 |

The rest are helper packages added or dropped between those releases (full list from
`bank/envs/bank-ga-defective.txt` and `bank/envs/bank-p0-fixed.txt`).

**What this means.** The pair alone cannot prove that the change in verdict comes from
Transformers: in principle, any of those 22 packages could contribute. Case A's G2 result
(no fail at the fixed pin) therefore rests on a multi-library comparison.

**What supports the attribution anyway (all seen, not new tests):**
- the mechanism is documented by Hugging Face (blog and fix PRs 34191, 34198) and visible
  in the source;
- P1's own pattern identifies it: on 4.45.2 the accumulated and one-batch runs disagree
  (accumulation defect), on 5.10.1 they agree and match the reference;
- an **isolated** comparison exists: `bank-ga-defective` (4.45.2) and `bank-ga-fixed`
  (4.46.3) differ only in transformers. On 4.46.3 (run `bank/results/2026-11-01_000416/`)
  the accumulated and one-batch runs moved by exactly the same amount (0.0333 in all four
  checks), so the accumulation defect was gone with only Transformers changed; the remaining
  deviation is case C.

**Not done:** an isolated environment for case A at 5.10.1 with torch 2.4.1 (it may not
install), and no claim is made about the 21 other packages.

## 5. The G4 second-machine spot check

**Purpose.** Show that the verdicts do not depend on this computer: one run of the controls
and one run of the bank on a second machine must give **the same verdict signature** (status
plus every finding's evidence class and verdict) as this machine's G4 runs. Numbers may
differ slightly between CPUs; verdicts must not.

**Second machine requirements**

| Requirement | Value |
|---|---|
| Operating system | Windows 10 or 11, 64-bit (same as this machine: the pinned PyPI PyTorch builds are CPU-only on Windows; Linux and macOS would install different builds and are out of scope for this check) |
| Python | 3.11, 64-bit, with the `py` launcher |
| Disk | about 8 GB free (main environment ≈ 1.0 GB, five bank environments ≈ 6.3 GB, tokenizer files ≈ 20 MB) |
| Network | needed only for installation (roughly 1.5 GB of downloads); the runs themselves are offline |
| GPU | not needed |
| Time | about 30–60 minutes to install; about 5 minutes for the two runs |

**Steps** are in Section 5.1. Step 0 runs on **this** machine; all others on the second one.

### 5.1 Commands

**0. On this machine, save the main environment's exact versions** (new file, no existing
file changes):

```bash
C:/Users/Shreyas/.venvs/conform/Scripts/python.exe -m pip freeze --exclude conform --disable-pip-version-check > bank/envs/conform.txt
```

**1. Get the project onto the second machine.** It lives in OneDrive, so signing in to the
same OneDrive syncs it; otherwise copy the whole folder. Open a terminal in the project
folder for every step below.

**2. Copy the three tokenizer snapshots** from this machine's
`%USERPROFILE%\.cache\huggingface\hub\` to the same folder on the second machine:
`models--Qwen--Qwen2.5-0.5B-Instruct`, `models--HuggingFaceTB--SmolLM2-135M-Instruct`,
`models--TinyLlama--TinyLlama-1.1B-Chat-v1.0`. This guarantees the same tokenizer files
(revisions `7ae5576…`, `12fd25f…`, `fe8a4ea…`). If copying is not possible, run
`python scripts/download_targets.py` in the main environment after step 3 and report the
revisions it prints; a different revision would make the check inconclusive.

**3. Create the main environment** (steps 3–6 are PowerShell commands):

```powershell
py -3.11 -m venv $HOME\.venvs\conform
```

```powershell
& $HOME\.venvs\conform\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
& $HOME\.venvs\conform\Scripts\python.exe -m pip install -r bank\envs\conform.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

```powershell
& $HOME\.venvs\conform\Scripts\python.exe -m pip install -e . --no-deps
```

**4. Create the five bank environments in use**, each from its saved exact list:

```powershell
foreach ($n in "bank-ga-defective","bank-eos-defective","bank-eos-fixed","bank-p0-defective","bank-p0-fixed") { py -3.11 -m venv "$HOME\.venvs\$n"; & "$HOME\.venvs\$n\Scripts\python.exe" -m pip install --upgrade pip; & "$HOME\.venvs\$n\Scripts\python.exe" -m pip install -r "bank\envs\$n.txt" }
```

**5. Check the setup** (expect 59 passed):

```powershell
& $HOME\.venvs\conform\Scripts\python.exe -m pytest -q
```

**6. Run the spot check**, once each, in this order:

```powershell
& $HOME\.venvs\conform\Scripts\python.exe scripts\run_controls.py
```

```powershell
& $HOME\.venvs\conform\Scripts\python.exe scripts\run_bank.py
```

Expected: both exit with code 0; the controls show 12 of 12 match and the bank 10 of 10.

**7. Bring the results back.** The two new folders in `bank/results/` (names start with
`controls_` and with the date) sync through OneDrive or can be copied back. Their verdict
signatures are then compared, run by run, with this machine's G4 results
(`controls_2026-11-01_091556/repeat-01/` and `2026-11-01_094057/repeat-01/`). G4 is met if
all 22 runs give identical signatures.

## 6. Not tested or not done

- G4's second-machine spot check (Section 5)
- P6 and P7 (listed in the spec's week-2 checklist; part of M1's scope)
- P3, P4 and P5
- GPU, mixed-precision, DeepSpeed and multi-device paths (spec Section 13's main blind spot)
- real model weights (proxy models only)
- PEFT (recorded in every environment, not exercised)
- the noise-floor thresholds (2× pass band, 10× fail line), provisional until M2 calibration
- P2 on TRL 0.16 could not place prompt boundaries, because of the extra EOS TRL appends; a
  question for P3

## 7. Decisions still open from spec Section 14

1. Licence: Apache-2.0 (decided)
2. Scope: SFT only for v0.1 (decided)
3. Contact TRL and Transformers maintainers before public claims (spec recommends yes): open
4. First public artifact, the tool or the bank write-up: open
