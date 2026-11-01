"""Run the G3 controls (bank/controls.yaml); results go to bank/results/controls_<timestamp>/.

Usage: python scripts/run_controls.py [--repeat N] [--venv-root DIR] [--timeout SECONDS] [--python PATH]
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

from conform.bank import DEFAULT_VENV_ROOT, PROJECT_ROOT, load_controls, run_and_report, run_controls


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the G3 control matrix")
    parser.add_argument("--repeat", type=int, default=1, help="run everything N times (G4 stability)")
    parser.add_argument("--venv-root", type=Path, default=DEFAULT_VENV_ROOT)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--python", help="interpreter to run the controls with (default: the "
                                         "environment named in bank/controls.yaml)")
    parser.add_argument("--only", nargs="+", metavar="ID", help="run only these control ids")
    args = parser.parse_args(argv)
    controls = load_controls()
    out_dir = PROJECT_ROOT / "bank" / "results" / datetime.now().strftime("controls_%Y-%m-%d_%H%M%S")
    return run_and_report(lambda: run_controls(controls, venv_root=args.venv_root, timeout=args.timeout,
                                               python=args.python, only=set(args.only or ())),
                          repeat=args.repeat, out_dir=out_dir)


if __name__ == "__main__":
    sys.exit(main())
