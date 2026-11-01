# G4 stop record

G4 (stability) was stopped at the user's request on **2026-11-01 at 00:51:53** (local time),
before completion. **G4 is not passed.**

## Where it stopped

| | |
|---|---|
| Planned run | controls × 10 (12 runs each), then bank × 10 (10 runs each): 220 probe runs |
| Started | 2026-11-01 00:41:23 |
| Stopped | 2026-11-01 00:51:53 (about 10.5 minutes in) |
| Phase reached | **controls × 10, in progress** |
| Last probe running | P2, TRL SFTTrainer, `Qwen/Qwen2.5-0.5B-Instruct` (control 10 of 12 within its repeat) |
| Repeat number reached | not recorded: the runner keeps finished repeats in memory |
| Bank × 10 | **not started** |

## What was saved

- **Nothing from this G4 attempt.** `scripts/run_controls.py` and `scripts/run_bank.py`
  write their results only after all repeats finish, so the repeats completed before the
  stop were held in memory and are lost. G4 must restart from the beginning.
- **All earlier results are preserved, unchanged**, in `bank/results/`:
  `2026-10-31_233804` (runner smoke test), `2026-10-31_235950` and `2026-11-01_000416`
  (earlier bank runs), `2026-11-01_003020` (first complete bank run, 10 of 10 match), and
  `controls_2026-11-01_003908` (the single G3 control run, 12 of 12 not fail).
- Checked at the stop: all 81 project files had the same name, size and timestamp before
  and after; no runner or probe process was left running.

## Gate status at the stop

| Gate | Status | Evidence |
|---|---|---|
| G1 detection | met (3 of 3 cases) | `bank/results/2026-11-01_003020/` |
| G2 silence on fixed pins | met (0 fails) | `bank/results/2026-11-01_003020/` |
| G3 controls | met on a single run (0 fails in 12) | `bank/results/controls_2026-11-01_003908/` |
| G4 stability | **not passed: not completed** | this file |

## To restart G4

Run from the project folder, in this order. Expect roughly 1–1.5 hours in total; keep the
laptop plugged in and awake, because both commands save results only when they finish.

```bash
cd C:/Users/Shreyas/OneDrive/Desktop/Training-Stack-Conformance-Layer
C:/Users/Shreyas/.venvs/conform/Scripts/python.exe scripts/run_controls.py --repeat 10
C:/Users/Shreyas/.venvs/conform/Scripts/python.exe scripts/run_bank.py --repeat 10
```

G4 passes only if both commands exit with code 0: every run matches its expectation and
every verdict is identical across all 10 repeats (`stability.md` in each results folder).

## Suggested before restarting (needs approval)

Change `run_and_report` in `src/conform/bank.py` to write each repeat to disk as soon as it
finishes, so a stop keeps the repeats already completed. Not done: it modifies a project file.
