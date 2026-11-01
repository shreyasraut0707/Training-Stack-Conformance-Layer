"""Run the regression bank (spec Section 10); results go to bank/results/<timestamp>/.

Usage: python scripts/run_bank.py [CASE_ID ...] [--repeat N] [--venv-root DIR] [--timeout SECONDS]
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

from conform.bank import DEFAULT_VENV_ROOT, PROJECT_ROOT, load_cases, run_and_report, run_case


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the regression bank")
    parser.add_argument("cases", nargs="*", help="case ids (default: all)")
    parser.add_argument("--repeat", type=int, default=1, help="run everything N times (G4 stability)")
    parser.add_argument("--venv-root", type=Path, default=DEFAULT_VENV_ROOT)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    cases = load_cases(set(args.cases) if args.cases else None)
    if not cases:
        parser.error("no matching cases")
    out_dir = PROJECT_ROOT / "bank" / "results" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return run_and_report(
        lambda: [r for case in cases for r in run_case(case, venv_root=args.venv_root, timeout=args.timeout)],
        repeat=args.repeat, out_dir=out_dir)


if __name__ == "__main__":
    sys.exit(main())
