"""Run every injected defect through its intended probe (spec Section 10.2, step 4).

Results go to bank/results/mutations_<timestamp>/ (mutations.json and summary.md). Mutants
are built by this project, so this evidence is weaker than the historical bank cases.

Usage: python scripts/run_mutations.py [--models MODEL ...] [--seed N]
"""
import argparse
import json
import logging
import sys
import warnings
from datetime import datetime

from conform.bank import PROJECT_ROOT
from conform.mutations import run_all
from conform.proxy_model import DEFAULT_TARGETS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the mutation suite")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    rows = run_all(args.models, seed=args.seed)
    out = PROJECT_ROOT / "bank" / "results" / datetime.now().strftime("mutations_%Y-%m-%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "mutations.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    applicable = [r for r in rows if r["caught"] is not None]
    caught = sum(r["caught"] for r in applicable)
    lines = [f"Mutants caught: {caught} of {len(applicable)} applicable "
             f"({len(rows) - len(applicable)} not applicable; injected defects are weaker evidence "
             "than the historical bank cases)", "",
             "| Mutant | Probe | Trainer | Model | Expected | Status | Caught |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['mutant']} | {r['probe']} | {r['trainer']} | {r['model']} | {r['expect']} | "
              f"{r['status']} | {'n/a' if r['caught'] is None else 'yes' if r['caught'] else 'NO'} |" for r in rows]
    missed = [r for r in applicable if not r["caught"]]
    if missed:
        lines += ["", "Missed:"] + [f"- {r['mutant']} / {r['probe']} / {r['trainer']} / {r['model']}: "
                                    f"{r['status']} {r['reason'] or ''}" for r in missed]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nResults: {out}")
    return 0 if not missed else 1


if __name__ == "__main__":
    sys.exit(main())
