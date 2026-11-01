# G4 second-machine spot check: step by step

Approved on 2026-11-01. Run these on the **second** computer only. Nothing here touches the
primary computer, changes a threshold, or starts M1. This refines Section 5 of
`docs/m0_go_no_go.md`: the project is copied to a local folder (not synced through
OneDrive) and the tokenizer files are downloaded and their versions checked.

Open **Windows PowerShell** (Start, type PowerShell) and run one block at a time.

## Step 1: Prerequisites

| Requirement | Why |
|---|---|
| Windows 10 or 11, 64-bit | PyTorch packages differ between operating systems; this matches the primary computer |
| Python 3.11, 64-bit, with the `py` launcher | the same Python as the primary computer |
| About 8 GB free disk space | main environment ≈ 1 GB, five bank environments ≈ 6.3 GB |
| Internet, for installing only | about 1.5 GB of downloads; the runs themselves are offline |
| About 1–1.5 hours | mostly installing; the runs take about 5 minutes |
| No GPU | everything runs on the CPU |

## Step 2: Install and check Python 3.11

Install Python 3.11 (Windows installer, 64-bit) from python.org if needed; keep "py launcher" ticked.

```powershell
py -3.11 --version
```
Expected: `Python 3.11.x`.

```powershell
py -3.11 -c "import struct; print(struct.calcsize('P') * 8)"
```
Expected: `64`.

## Step 3: Copy the project

Copy the whole `Training-Stack-Conformance-Layer` folder (USB stick or OneDrive download) to
`C:\Users\<your name>\conform-spotcheck`, outside OneDrive, so nothing syncs back.

```powershell
cd $HOME\conform-spotcheck
```
Run every later step from this folder.

```powershell
Get-ChildItem bank\envs
```
Expected: 7 `.txt` files, including `conform.txt`.

## Step 4: Create the main virtual environment

```powershell
py -3.11 -m venv $HOME\.venvs\conform
```

```powershell
& $HOME\.venvs\conform\Scripts\python.exe -m pip install --upgrade pip
```

## Step 5: Install the 58 pinned packages

```powershell
& $HOME\.venvs\conform\Scripts\python.exe -m pip install -r bank\envs\conform.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Check (expected: no output at all):

```powershell
Compare-Object (Get-Content bank\envs\conform.txt) (& $HOME\.venvs\conform\Scripts\python.exe -m pip freeze --exclude conform)
```

## Step 6: Install the project itself

```powershell
& $HOME\.venvs\conform\Scripts\python.exe -m pip install -e . --no-deps
```

## Step 7: Create the five bank environments

```powershell
foreach ($n in "bank-ga-defective","bank-eos-defective","bank-eos-fixed","bank-p0-defective","bank-p0-fixed") { py -3.11 -m venv "$HOME\.venvs\$n"; & "$HOME\.venvs\$n\Scripts\python.exe" -m pip install --upgrade pip; & "$HOME\.venvs\$n\Scripts\python.exe" -m pip install -r "bank\envs\$n.txt" }
```

Check (expected: five lines ending in `identical`):

```powershell
foreach ($n in "bank-ga-defective","bank-eos-defective","bank-eos-fixed","bank-p0-defective","bank-p0-fixed") { $d = Compare-Object (Get-Content "bank\envs\$n.txt") (& "$HOME\.venvs\$n\Scripts\python.exe" -m pip freeze); "$n : $(if ($d) {'DIFFERENT'} else {'identical'})" }
```

## Step 8: Download the tokenizer files

```powershell
& $HOME\.venvs\conform\Scripts\python.exe scripts\download_targets.py
```

The three printed paths must end with these versions; if any differs, stop and report it:

- Qwen: `7ae557604adf67be50417f59c2c2f167def9a775`
- SmolLM2: `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- TinyLlama: `fe8a4ea1ffedaf415f4da2f062534de366a451e6`

## Step 9: Check the setup

```powershell
& $HOME\.venvs\conform\Scripts\python.exe -m pytest -q
```
Expected final line: `59 passed`.

## Step 10: Run the spot check

```powershell
& $HOME\.venvs\conform\Scripts\python.exe scripts\run_controls.py --only p1-transformers p1-trl p2-transformers p2-trl p3-trl p6-transformers p6-trl p7-trl
```
(The v0.1 controls, 24 runs; the M4 controls come separately below.)

```powershell
$LASTEXITCODE
```
Expected: every row `match`, then `0`.

```powershell
& $HOME\.venvs\conform\Scripts\python.exe scripts\run_bank.py
```

```powershell
$LASTEXITCODE
```
Expected: every row `match`, then `0`. If either is not 0, do not re-run: go to Step 11.

**Added for M4 (v0.2 probes P4, P5, P1M; about 3 minutes):**

```powershell
& $HOME\.venvs\conform\Scripts\python.exe scripts\run_controls.py --only p4-trl p5-transformers p5-trl p1m-transformers
```

```powershell
$LASTEXITCODE
```
Expected: 12 rows `match`, then `0`. Close other large programs first: P1M runs three Python
processes at once, and the first attempt on the primary laptop (7.3 GB of memory) ran out of
memory.

## Step 11: Send the results back

```powershell
Get-ChildItem bank\results -Directory | Sort-Object LastWriteTime | Select-Object -Last 3 Name
```

```powershell
$new = Get-ChildItem bank\results -Directory | Sort-Object LastWriteTime | Select-Object -Last 3; Compress-Archive -Path $new.FullName -DestinationPath $HOME\Desktop\g4_second_machine.zip
```

(Three folders: the controls, the bank, and the M4 controls.)

Move `g4_second_machine.zip` to the primary computer (USB, email or other transfer) and
report where it was saved. It is then extracted into a new folder,
`bank\results\second_machine\`, and compared run by run with the primary G4 results.

## How the result is judged

G4 is met if every run gives identical verdicts on both computers: the same status and the
same evidence class and verdict for every finding. Small numeric differences are allowed; a
different verdict is not. The comparison is done on the primary computer, folder by folder:

```powershell
python scripts\compare_results.py bank\results\<primary folder> bank\results\second_machine\<second folder>
```

| Second-machine folder | Compare with (primary) | Runs | Closes |
|---|---|---|---|
| `controls_…` (Step 10, first command) | `controls_2026-11-01_104321` | 24 | G4 (and the v0.1 controls) |
| `<date>_<time>` (Step 10, bank) | the latest full bank run | 10 or more | G4 |
| `controls_…` (Step 10, M4 command) | `controls_2026-11-01_171529` | 12 | M4's stability threshold |

**Known borderline runs (recorded 2026-11-01, before any second-computer run).** On this
computer, changing only the CPU thread count moved `p1-trl` on SmolLM2 from info to pass;
`p1-trl` on TinyLlama (1.96×) and on Qwen2.5 (3.68×) also sit near the 2× pass band. A
computer with a different number of cores may therefore report `DIFFERENT` for these three
runs (pass ↔ info), never near the fail line. The criterion above is not changed; how such
a difference is counted is the owner's decision (`docs/m0_findings.md`, "Extra
single-computer evidence for G4").

Simpler alternative for Step 7: `& $HOME\.venvs\conform\Scripts\python.exe scripts\setup_bank_envs.py`
creates the bank environments and checks them against their lists in one command.
