"""P6: what precision does training actually use? (observation only; never fails)

Spec Section 7. One real training step with the library's own precision defaults (unless
the recipe sets them), then the dtypes P6 sees: parameters, the outputs of the model's
linear layers (the compute dtype), gradients, and the optimizer state. Each precision
setting is marked as the library default or as set by the recipe, so silent defaults
are visible (for example, TRL turns bf16 mixed precision on when it is not set).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile

import torch

from conform.environment import capture_environment
from conform.proxy_model import build_proxy
from conform.result import EvidenceClass, Finding, ProbeResult, Status, exit_code
from conform.stack import (
    DATA_FORMATS, SAMPLES, TRAINERS, add_common_arguments, build_trainer, common_kwargs,
    conversations, reproducer as make_reproducer,
)

PROBE_ID = "P6-precision"
ORACLE = "observation only: dtypes seen during and after one real training step"
PRECISION_SETTINGS = ("bf16", "fp16", "tf32")
LIMITATIONS = [
    "proxy model built in float32 and passed as an object, so a trainer's own model-loading "
    "dtype default (when given a model name) is not observed",
    "CPU only; GPU kernels and their precision behaviour are not tested",
    "observation only: P6 never fails",
]


def _names(dtypes) -> list[str]:
    return sorted(str(d).replace("torch.", "") for d in dtypes)


def _settings(built, trainer, trainer_args) -> dict:
    """Each precision setting's resolved value, and whether the recipe set it."""
    args = built.args
    return {name: {"value": getattr(args, name, None),
                   "source": "recipe" if name in (trainer_args or {}) else "library default"}
            for name in PRECISION_SETTINGS if hasattr(args, name)}


def observe(built) -> dict:
    """Run one training step and record the dtypes seen."""
    compute, grads = set(), set()
    model = built.model
    handles = [m.register_forward_hook(lambda _m, _i, out: compute.add(out.dtype))
               for m in model.modules() if isinstance(m, torch.nn.Linear)]
    handles += [p.register_hook(lambda g: grads.add(g.dtype) or g)
                for p in model.parameters() if p.requires_grad]
    try:
        built.train()
    finally:
        for h in handles:
            h.remove()
    optimizer = getattr(built.optimizer, "optimizer", built.optimizer)
    state = {t.dtype for s in optimizer.state.values() for t in s.values() if torch.is_tensor(t) and t.dim() > 0}
    return {
        "parameters": _names({p.dtype for p in model.parameters()}),
        "compute": _names(compute),
        "gradients": _names(grads),
        "optimizer": type(optimizer).__name__,
        "optimizer_state": _names(state),
    }


def run(model_id: str, *, trainer: str = "transformers", seed: int = 0,
        data_format: str = "prompt_completion", samples=None, rows_path=None,
        trainer_args: dict | None = None) -> ProbeResult:
    if trainer not in TRAINERS:
        raise ValueError(f"unknown trainer {trainer!r}; choose from {sorted(TRAINERS)}")
    if data_format not in DATA_FORMATS:
        raise ValueError(f"unknown data format {data_format!r}; choose from {DATA_FORMATS}")
    environment = capture_environment(device="cpu", dtype="library default",
                                      seeds={"proxy": seed, "trainer": seed})
    reproducer = make_reproducer("p6_precision", model_id, trainer=trainer, data_format=data_format,
                                 seed=seed, rows_path=rows_path, trainer_args=trainer_args)
    limitations = list(LIMITATIONS)
    common = dict(oracle=ORACLE, environment=environment, reproducer=reproducer, limitations=limitations)
    try:
        model, tokenizer, info = build_proxy(model_id, seed=seed)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as out:
            built, dropped = build_trainer(
                trainer, model, tokenizer, conversations(samples or SAMPLES), output_dir=out,
                seed=seed, data_format=data_format, force_fp32=False,
                trainer_args={"learning_rate": 1e-4, "logging_steps": 1, **(trainer_args or {}),
                              "max_steps": 1})
            settings = _settings(built, trainer, trainer_args)
            seen = observe(built)
    except Exception as exc:  # an API change must become not_tested, never a wrong verdict
        return ProbeResult(PROBE_ID, Status.NOT_TESTED, reason=f"{type(exc).__name__}: {exc}", **common)
    if dropped:
        limitations.append(f"this library version has no option(s) {dropped}; "
                           "they were left at the library default")

    stated = ", ".join(f"{k}={v['value']} ({v['source']})" for k, v in settings.items())
    findings = [
        Finding(EvidenceClass.SEEN, f"Precision settings: {stated}", evidence={"settings": settings}),
        Finding(EvidenceClass.SEEN, f"Parameters after one step are {', '.join(seen['parameters'])}",
                evidence={"parameters": seen["parameters"]}),
        Finding(EvidenceClass.SEEN, f"Linear layers computed in {', '.join(seen['compute']) or 'nothing seen'}",
                evidence={"compute": seen["compute"]}),
        Finding(EvidenceClass.SEEN, f"Gradients are {', '.join(seen['gradients'])}",
                evidence={"gradients": seen["gradients"]}),
        Finding(EvidenceClass.SEEN,
                f"Optimizer {seen['optimizer']} keeps its state in {', '.join(seen['optimizer_state']) or 'no tensors'}",
                evidence={"optimizer": seen["optimizer"], "optimizer_state": seen["optimizer_state"]}),
    ]
    if seen["compute"] and seen["compute"] != seen["parameters"]:
        findings.append(Finding(
            EvidenceClass.SEEN,
            f"Mixed precision: computation runs in {', '.join(seen['compute'])} while parameters "
            f"are stored in {', '.join(seen['parameters'])}",
            evidence={"compute": seen["compute"], "parameters": seen["parameters"]}))
    silent = [k for k, v in settings.items() if v["source"] == "library default" and v["value"]]
    if silent:
        findings.append(Finding(
            EvidenceClass.SEEN,
            f"Turned on by the library default, not by the recipe: {', '.join(silent)}",
            evidence={"silent_defaults": silent}))
    observed = {"trainer": TRAINERS[trainer], "proxy": info.to_dict(), "settings": settings, **seen,
                "unsupported_trainer_options": dropped}
    return ProbeResult(PROBE_ID, Status.INFO, findings=findings, observed=observed, **common)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P6: what precision does training use")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    result = run(args.model_id, **common_kwargs(args))
    print(json.dumps(result.to_dict(), indent=2))
    return exit_code([result])


if __name__ == "__main__":
    sys.exit(main())
