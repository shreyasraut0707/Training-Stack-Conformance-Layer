"""Noise-floor calibration data for P1 (spec Sections 7.2 and 10.2, step 3).

The spec: set the noise-floor multiplier on synthetic injected defects, then confirm on
held-out historical cases, so the real accumulation case is not used both to tune and to
judge. This module only measures: for correct stacks (several tokenizer families, both
trainers, several seeds) and for small injected loss-scale defects, it records each check's
deviation as a multiple of its measured floor. Choosing thresholds from these numbers is a
separate, recorded decision (docs/calibration.md).
"""
from __future__ import annotations

from conform.mutations import loss_scale, _hf, _sft
from conform.probes.p1_grad_accum import run as run_p1

EPSILONS = (1e-3, 1e-4, 1e-5, 1e-6)


def ratios(result) -> dict:
    """Each reference check's deviation / floor, keyed 'run / metric'."""
    return {f"{f.evidence['run']} / {f.evidence['metric']}": f.evidence["deviation"] / f.evidence["floor"]
            for f in result.findings if "floor" in f.evidence}


def correct_stacks(model_ids, seeds, trainers=("transformers", "trl")) -> list[dict]:
    rows = []
    for model_id in model_ids:
        for trainer in trainers:
            for seed in seeds:
                result = run_p1(model_id, seed=seed, trainer=trainer)
                rows.append({"model": model_id, "trainer": trainer, "seed": seed,
                             "status": result.status.value, "ratios": ratios(result)})
    return rows


def injected_defects(model_id, epsilons=EPSILONS, trainers=("transformers", "trl"), seed=0) -> list[dict]:
    rows = []
    for eps in epsilons:
        for trainer in trainers:
            base = _hf() if trainer == "transformers" else _sft()
            result = run_p1(model_id, seed=seed, trainer=trainer, trainer_cls=loss_scale(eps)(base))
            rows.append({"model": model_id, "trainer": trainer, "epsilon": eps,
                         "status": result.status.value, "ratios": ratios(result)})
    return rows


def summarise(correct: list[dict], defects: list[dict]) -> dict:
    """P1 fails when any check is clearly above the floor, so both sides use each run's
    largest ratio: the largest seen on correct stacks (false-alarm risk, per trainer) and the
    smallest such value across runs of each injected defect (detection strength, per epsilon)."""
    out = {"correct_max": {}, "defect_detection": {}}
    for row in correct:
        top = max(row["ratios"].values())
        out["correct_max"][row["trainer"]] = max(out["correct_max"].get(row["trainer"], 0.0), top)
    for row in defects:
        key = f"{row['epsilon']:g}"
        top = max(row["ratios"].values())
        out["defect_detection"][key] = min(out["defect_detection"].get(key, float("inf")), top)
    return out
