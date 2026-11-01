"""P5: does a resumed run match an uninterrupted run?

Spec Section 7 (release v0.2). The oracle is the uninterrupted run in deterministic CPU
float32. The same proxy, rows and seed are trained N steps straight through, and
separately k steps, checkpointed, then resumed in a fresh trainer up to N. P5 fails if the
final parameters differ beyond the measured floor, if the learning-rate trajectory differs,
or if the order in which rows were seen differs. The default AdamW optimizer and a warmup
schedule are used on purpose: their saved state is what a resume must restore.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import TrainerCallback

from conform.environment import capture_environment
from conform.noise_floor import FLOAT32_RESOLUTION, ABOVE, UNCLEAR, classify, relative_gap
from conform.proxy_model import build_proxy
from conform.result import EvidenceClass, Finding, ProbeResult, Status, exit_code
from conform.stack import (
    DATA_FORMATS, SAMPLES, TRAINERS, add_common_arguments, build_trainer, common_kwargs, conversations,
    reproducer as make_reproducer,
)

PROBE_ID = "P5-resume-equivalence"
ORACLE = "the uninterrupted run, in deterministic CPU float32"
TOTAL_STEPS = 6
STOP_AT = 3
EXTRA_SAMPLES = (("Name a fruit.", "An apple."), ("What colour is snow?", "Snow is white."),
                 ("Say no.", "No."), ("What is 3+3?", "It is 6."))
LIMITATIONS = [
    "proxy model; real weights not used",
    f"{TOTAL_STEPS} steps, checkpoint after step {STOP_AT}, batch size 1, single process on CPU",
    "multi-process and GPU resume paths not tested",
]


class _Recorder(TrainerCallback):
    """Learning rate and the rows of each optimizer step, through the public callback API."""

    def __init__(self, stop_at=None):
        self.lrs, self.stop_at = {}, stop_at

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        if optimizer is not None:
            self.lrs[state.global_step] = optimizer.param_groups[0]["lr"]
        if self.stop_at is not None and state.global_step >= self.stop_at:
            control.should_training_stop = True
            control.should_save = True
        return control


class _OrderCollator:
    """Wraps the trainer's collator and records which rows each micro-batch contains."""

    def __init__(self, inner, tokenizer):
        self.inner, self.seen = inner, []
        self.pad = tokenizer.pad_token_id

    def __call__(self, features):
        # The full token sequence identifies a row: chat templates often start every row with
        # the same system prompt, so a prefix would make all rows look alike.
        self.seen.append(tuple(tuple(f["input_ids"]) for f in features))
        return self.inner(features)


def _train(trainer, model, tokenizer, convs, out, seed, data_format, trainer_args, trainer_cls,
           stop_at=None, resume=None):
    args = {"max_steps": TOTAL_STEPS, "learning_rate": 1e-3, "lr_scheduler_type": "linear",
            "warmup_steps": 2, "per_device_train_batch_size": 1, "logging_steps": 1,
            "save_strategy": "steps", "save_steps": STOP_AT, "save_only_model": False,
            **(trainer_args or {})}
    built, dropped = build_trainer(trainer, model, tokenizer, convs, output_dir=out, seed=seed,
                                   data_format=data_format, trainer_args=args, batch_size=1,
                                   trainer_cls=trainer_cls)
    recorder = _Recorder(stop_at)
    built.add_callback(recorder)
    built.data_collator = _OrderCollator(built.data_collator, tokenizer)
    built.train(resume_from_checkpoint=resume)
    params = torch.cat([p.detach().reshape(-1).double() for p in built.model.parameters()])
    result = {"params": params, "lrs": recorder.lrs, "seen": built.data_collator.seen, "dropped": dropped}
    del built, model  # free this run's trainer and model before the next run (small machines)
    gc.collect()
    return result


def run(model_id: str, *, trainer: str = "transformers", seed: int = 0, data_format: str = "prompt_completion",
        samples=None, rows_path=None, trainer_args: dict | None = None, trainer_cls=None) -> ProbeResult:
    if trainer not in TRAINERS:
        raise ValueError(f"unknown trainer {trainer!r}; choose from {sorted(TRAINERS)}")
    if data_format not in DATA_FORMATS:
        raise ValueError(f"unknown data format {data_format!r}; choose from {DATA_FORMATS}")
    environment = capture_environment(device="cpu", dtype="float32", seeds={"proxy": seed, "trainer": seed})
    reproducer = make_reproducer("p5_resume", model_id, trainer=trainer, data_format=data_format, seed=seed,
                                 rows_path=rows_path, trainer_args=trainer_args)
    limitations = list(LIMITATIONS)
    common = dict(oracle=ORACLE, environment=environment, reproducer=reproducer, limitations=limitations)
    convs = conversations(samples or (SAMPLES + EXTRA_SAMPLES))
    try:
        model, tokenizer, info = build_proxy(model_id, seed=seed)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
            work = Path(work)
            straight = _train(trainer, copy.deepcopy(model), tokenizer, convs, work / "a", seed, data_format,
                              trainer_args, None)
            again = _train(trainer, copy.deepcopy(model), tokenizer, convs, work / "b", seed, data_format,
                           trainer_args, None)
            first = _train(trainer, copy.deepcopy(model), tokenizer, convs, work / "c", seed, data_format,
                           trainer_args, trainer_cls, stop_at=STOP_AT)
            checkpoint = work / "c" / f"checkpoint-{STOP_AT}"
            if not checkpoint.exists():
                raise RuntimeError(f"no checkpoint written at step {STOP_AT}")
            resumed = _train(trainer, copy.deepcopy(model), tokenizer, convs, work / "c", seed, data_format,
                             trainer_args, trainer_cls, resume=str(checkpoint))
    except Exception as exc:  # an API change must become not_tested, never a wrong verdict
        return ProbeResult(PROBE_ID, Status.NOT_TESTED, reason=f"{type(exc).__name__}: {exc}", **common)
    if straight["dropped"]:
        limitations.append(f"this library version has no option(s) {straight['dropped']}; "
                           "they were left at the library default")

    floor = max(relative_gap(again["params"], straight["params"]), FLOAT32_RESOLUTION)
    param_gap = relative_gap(resumed["params"], straight["params"])
    param_verdict = classify(param_gap, floor)
    lr_after = {s: lr for s, lr in straight["lrs"].items() if s > STOP_AT}
    lr_resumed = {s: lr for s, lr in resumed["lrs"].items() if s > STOP_AT}
    lr_same = lr_after.keys() == lr_resumed.keys() and all(abs(lr_after[s] - lr_resumed[s]) <= 1e-12 * max(1.0, abs(lr_after[s]))
                                                         for s in lr_after)
    order_straight = straight["seen"]
    order_resumed = first["seen"][:STOP_AT] + resumed["seen"]
    order_same = order_resumed == order_straight

    findings = [
        Finding(EvidenceClass.UNKNOWN if param_verdict == UNCLEAR else EvidenceClass.SEEN,
                f"Final parameters of the resumed run differ from the uninterrupted run by {param_gap:.3g} "
                f"(floor {floor:.3g}, measured between two uninterrupted runs)",
                evidence={"metric": "parameters", "deviation": param_gap, "floor": floor, "verdict": param_verdict}),
        Finding(EvidenceClass.SEEN,
                "Learning rate after the resume " + ("matches" if lr_same else "differs from")
                + f" the uninterrupted run (steps {STOP_AT + 1}-{TOTAL_STEPS})",
                evidence={"metric": "learning_rate", "uninterrupted": lr_after, "resumed": lr_resumed,
                          "verdict": "inside" if lr_same else "above"}),
        Finding(EvidenceClass.SEEN,
                "Rows were seen in " + ("the same order" if order_same else "a different order")
                + " in the resumed run",
                evidence={"metric": "data_order", "same": order_same, "micro_batches": len(order_straight),
                          "verdict": "inside" if order_same else "above"}),
    ]
    if param_verdict == ABOVE or not lr_same or not order_same:
        status = Status.FAIL
        findings.append(Finding(EvidenceClass.INFERRED,
                                "Interrupting and resuming training changes what the model learns",
                                rule="a resumed run that differs from the uninterrupted run in parameters, "
                                     "learning rate or data order is not equivalent"))
    elif param_verdict == UNCLEAR:
        status = Status.INFO
    else:
        status = Status.PASS
    observed = {"trainer": TRAINERS[trainer], "proxy": info.to_dict(), "steps": TOTAL_STEPS, "stop_at": STOP_AT,
                "param_gap": param_gap, "floor": floor, "lr_same": lr_same, "order_same": order_same}
    return ProbeResult(PROBE_ID, status, findings=findings, observed=observed, noise_floor=floor, **common)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P5: does a resumed run match an uninterrupted run")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    result = run(args.model_id, **common_kwargs(args))
    print(json.dumps(result.to_dict(), indent=2))
    return exit_code([result])


if __name__ == "__main__":
    sys.exit(main())
