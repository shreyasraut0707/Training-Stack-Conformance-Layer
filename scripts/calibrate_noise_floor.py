"""Measure P1's noise-floor calibration data (spec Section 10.2, step 3).

Correct stacks: every default tokenizer family, both trainers, several seeds. Injected
defects: the loss scaled by (1 + epsilon) for small epsilons. Results go to
bank/results/calibration_<timestamp>/calibration.json. Choosing thresholds from them is a
separate, recorded decision.

Usage: python scripts/calibrate_noise_floor.py [--seeds 0 1 2 3 4]
"""
import argparse
import json
import logging
import sys
import warnings
from datetime import datetime

from conform.bank import PROJECT_ROOT
from conform.calibration import correct_stacks, injected_defects, summarise
from conform.noise_floor import PROVISIONAL_MULTIPLIER, PROVISIONAL_PASS_BAND
from conform.proxy_model import DEFAULT_TARGETS, LLAMA_TYPE


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure P1 noise-floor calibration data")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    correct = correct_stacks(DEFAULT_TARGETS, args.seeds)
    defects = injected_defects(LLAMA_TYPE)
    summary = summarise(correct, defects)
    out = PROJECT_ROOT / "bank" / "results" / datetime.now().strftime("calibration_%Y-%m-%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(
        {"current": {"pass_band": PROVISIONAL_PASS_BAND, "fail_multiplier": PROVISIONAL_MULTIPLIER},
         "summary": summary, "correct": correct, "defects": defects}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results: {out / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
