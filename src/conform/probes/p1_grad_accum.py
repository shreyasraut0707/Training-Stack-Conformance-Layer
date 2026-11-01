"""P1: does accumulating k micro-batches match one batch of the same samples?

Spec Section 7. The trainer runs the same four rows twice: as four accumulated
micro-batches of one row, and as one batch of four rows. Both are compared with the
independent float64 oracle. The rows have deliberately unequal numbers of supervised
tokens; with equal counts the classic defect cannot show (spec Figure 3).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile

import torch
from transformers import Trainer, TrainingArguments, default_data_collator

from conform.environment import capture_environment
from conform.noise_floor import (
    ABOVE, INSIDE, PROVISIONAL_MULTIPLIER, PROVISIONAL_PASS_BAND, UNCLEAR, classify, flat_gradient,
    measure_floor, relative_gap,
)
from conform.oracle import IGNORE_INDEX
from conform.proxy_model import build_proxy
from conform.result import EvidenceClass, Finding, ProbeResult, Status, exit_code

PROBE_ID = "P1-grad-accum-equivalence"
TRAINERS = {"transformers": "Transformers Trainer", "trl": "TRL SFTTrainer"}
ORACLE = ("float64 reference: summed token loss over non-ignored labels / total "
          "non-ignored tokens of the whole batch (conform.oracle)")
SEQ_LEN = 16
# (real tokens, prompt tokens) per row. Supervised targets after the causal shift are
# 14, 8, 4 and 3. The last row has no prompt, so its position-0 label is set but can
# never be a target: counting it by mistake is also caught.
ROW_SHAPES = ((16, 2), (12, 4), (7, 3), (4, 0))
LEARNING_RATE = 1e-3
LIMITATIONS = [
    "proxy model; real weights not used",
    "single-process CPU float32; GPU, mixed precision and distributed paths not tested",
    f"noise-floor fail multiplier ({PROVISIONAL_MULTIPLIER:g}x) calibrated on injected defects and "
    f"held-out historical cases; pass band ({PROVISIONAL_PASS_BAND:g}x) kept by decision, so TRL's "
    "unmodelled kernel rounding reads as unknown (docs/calibration.md)",
    "update checked as the accumulated gradient handed to the optimizer "
    "(plain SGD, no clipping); optimizer maths itself not tested",
]
TRL_LIMITATION = (
    "TRL's chunked log-prob kernel computes in float32 whatever the model dtype; the noise "
    "floor does not sample its rounding, so its accumulated update can read as unknown "
    "(docs/m0_findings.md)"
)


def build_rows(vocab_limit: int, pad_id: int, seed: int) -> list[dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    rows = []
    for real, prompt in ROW_SHAPES:
        input_ids = torch.full((SEQ_LEN,), pad_id, dtype=torch.long)
        input_ids[:real] = torch.randint(0, vocab_limit, (real,), generator=g)
        attention_mask = torch.zeros(SEQ_LEN, dtype=torch.long)
        attention_mask[:real] = 1
        labels = torch.full((SEQ_LEN,), IGNORE_INDEX, dtype=torch.long)
        labels[prompt:real] = input_ids[prompt:real]
        rows.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
    return rows


def supervised_targets(rows) -> list[int]:
    return [int((row["labels"][1:] != IGNORE_INDEX).sum()) for row in rows]


class _CapturingSGD(torch.optim.SGD):
    """Plain SGD that copies the accumulated gradient just before applying it, so P1 reads
    the update without relying on a trainer callback (works across library versions)."""

    def __init__(self, params, lr):
        self._params = list(params)
        super().__init__(self._params, lr=lr)
        self.gradient = None

    def step(self, closure=None):
        self.gradient = flat_gradient(self._params)
        return super().step(closure)


def _training_kwargs(output_dir, *, batch_size, accumulation, seed) -> dict:
    return dict(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        max_steps=1,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=0.0,
        bf16=False,
        fp16=False,
        use_cpu=True,
        # No data_seed: older Transformers need accelerate >= 1.1 for it, and seed already fixes the data order
        seed=seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )


def _unpadded(rows) -> list[dict[str, list[int]]]:
    """Rows without padding, for trainers whose own collator pads each batch."""
    out = []
    for row in rows:
        keep = row["attention_mask"].bool()
        out.append({"input_ids": row["input_ids"][keep].tolist(), "labels": row["labels"][keep].tolist()})
    return out


def _build_trainer(trainer, model, tokenizer, rows, kwargs, optimizer, trainer_cls):
    if trainer == "transformers":
        return (trainer_cls or Trainer)(
            model=model, args=TrainingArguments(**kwargs), train_dataset=rows,
            data_collator=default_data_collator, optimizers=(optimizer, None),
        )
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    # Already tokenized, with our own labels: SFTTrainer skips tokenization, keeps the
    # labels, and its own collator pads each batch. Packing is P4's job, so it is off.
    args = SFTConfig(**kwargs, packing=False, padding_free=False)
    return (trainer_cls or SFTTrainer)(
        model=model, args=args, train_dataset=Dataset.from_list(_unpadded(rows)),
        processing_class=tokenizer, optimizers=(optimizer, None),
    )


def run_trainer(model, tokenizer, rows, *, trainer, batch_size, accumulation, seed, trainer_cls):
    """One optimizer step with the real trainer; returns (logged loss, applied gradient)."""
    optimizer = _CapturingSGD(model.parameters(), lr=LEARNING_RATE)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as out:
        kwargs = _training_kwargs(out, batch_size=batch_size, accumulation=accumulation, seed=seed)
        output = _build_trainer(trainer, model, tokenizer, rows, kwargs, optimizer, trainer_cls).train()
    if optimizer.gradient is None:
        raise RuntimeError("the trainer never reached an optimizer step")
    return output.training_loss, optimizer.gradient


def run(
    model_id: str,
    *,
    seed: int = 0,
    trainer: str = "transformers",
    trainer_cls: type[Trainer] | None = None,
) -> ProbeResult:
    if trainer not in TRAINERS:
        raise ValueError(f"unknown trainer {trainer!r}; choose from {sorted(TRAINERS)}")
    environment = capture_environment(
        device="cpu", dtype="float32", seeds={"proxy": seed, "trainer": seed}
    )
    reproducer = f"python -m conform.probes.p1_grad_accum {model_id} --trainer {trainer} --seed {seed}"
    limitations = LIMITATIONS + ([TRL_LIMITATION] if trainer == "trl" else [])
    common = dict(oracle=ORACLE, environment=environment, reproducer=reproducer,
                  limitations=limitations)
    try:
        model, tokenizer, info = build_proxy(model_id, seed=seed)
        pad_id = next(i for i in (tokenizer.pad_token_id, tokenizer.eos_token_id, 0) if i is not None)
        rows = build_rows(len(tokenizer), pad_id, seed)
        micro_batches = [{k: v.unsqueeze(0) for k, v in row.items()} for row in rows]
        ref_loss, ref_grad, floor = measure_floor(model, micro_batches)
        runs = {
            "accumulated (4 micro-batches x 1 row)": run_trainer(
                copy.deepcopy(model), tokenizer, rows, trainer=trainer, batch_size=1,
                accumulation=len(rows), seed=seed, trainer_cls=trainer_cls),
            "one batch (1 x 4 rows)": run_trainer(
                copy.deepcopy(model), tokenizer, rows, trainer=trainer, batch_size=len(rows),
                accumulation=1, seed=seed, trainer_cls=trainer_cls),
        }
    except Exception as exc:  # an API change must become not_tested, never a wrong verdict
        return ProbeResult(PROBE_ID, Status.NOT_TESTED, reason=f"{type(exc).__name__}: {exc}", **common)

    findings, verdicts = [], []
    run_verdicts = {}
    observed = {
        "trainer": TRAINERS[trainer],
        "proxy": info.to_dict(),
        "supervised_targets_per_row": supervised_targets(rows),
        "reference_loss": ref_loss,
        "floor_loss": floor.loss,
        "floor_update": floor.gradient,
        "pass_band": PROVISIONAL_PASS_BAND,
        "multiplier": PROVISIONAL_MULTIPLIER,
    }
    for name, (loss, grad) in runs.items():
        observed[f"{name}: loss"] = loss
        for metric, deviation, metric_floor in (
            ("loss", relative_gap(loss, ref_loss), floor.loss),
            ("update", relative_gap(grad, ref_grad), floor.gradient),
        ):
            verdict = classify(deviation, metric_floor)
            verdicts.append(verdict)
            run_verdicts.setdefault(name, []).append(verdict)
            observed[f"{name}: {metric} deviation"] = deviation
            findings.append(Finding(
                EvidenceClass.UNKNOWN if verdict == UNCLEAR else EvidenceClass.SEEN,
                f"{name}: {metric} differs from the float64 reference by {deviation:.3g} "
                f"(floor {metric_floor:.3g}, pass up to {metric_floor * PROVISIONAL_PASS_BAND:.3g}, "
                f"fail above {metric_floor * PROVISIONAL_MULTIPLIER:.3g})",
                evidence={"run": name, "metric": metric, "deviation": deviation,
                          "floor": metric_floor, "verdict": verdict},
            ))

    (acc_name, (acc_loss, acc_grad)), (one_name, (one_loss, one_grad)) = runs.items()
    pair = {"loss": relative_gap(acc_loss, one_loss), "update": relative_gap(acc_grad, one_grad)}
    pair_verdicts = {classify(pair["loss"], floor.loss), classify(pair["update"], floor.gradient)}
    pair_verdict = ABOVE if ABOVE in pair_verdicts else UNCLEAR if UNCLEAR in pair_verdicts else INSIDE
    observed["accumulated vs one batch: loss gap"] = pair["loss"]
    observed["accumulated vs one batch: update gap"] = pair["update"]
    findings.append(Finding(
        EvidenceClass.UNKNOWN if pair_verdict == UNCLEAR else EvidenceClass.SEEN,
        f"The accumulated and one-batch runs differ from each other by {pair['loss']:.3g} in loss "
        f"and {pair['update']:.3g} in update",
        evidence={"run": "accumulated vs one batch", "metric": "loss and update",
                  "loss_gap": pair["loss"], "update_gap": pair["update"], "verdict": pair_verdict},
    ))

    if ABOVE in verdicts:
        status = Status.FAIL
        accumulated_off = ABOVE in run_verdicts[acc_name]
        one_batch_off = ABOVE in run_verdicts[one_name]
        if accumulated_off and pair_verdict == ABOVE:
            claim = "Accumulating micro-batches does not give the same result as one large batch"
            rule = "the accumulated run is clearly away from both the reference and the one-batch run"
        elif accumulated_off and one_batch_off:
            claim = ("The loss normalisation differs from the reference in both runs, so the cause "
                     "is not accumulation itself")
            rule = "equal deviations in the accumulated and one-batch runs point to a shared normalisation"
        else:
            claim = "Training on this stack does not optimise the reference objective"
            rule = "a loss or update clearly above the noise floor differs from the reference"
        findings.append(Finding(EvidenceClass.INFERRED, claim, rule=rule))
    elif UNCLEAR in verdicts:
        status = Status.INFO
    else:
        status = Status.PASS
    return ProbeResult(PROBE_ID, status, findings=findings, observed=observed,
                       noise_floor=floor.gradient, **common)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P1: gradient-accumulation equivalence")
    parser.add_argument("model_id")
    parser.add_argument("--trainer", choices=sorted(TRAINERS), default="transformers")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = run(args.model_id, seed=args.seed, trainer=args.trainer)
    print(json.dumps(result.to_dict(), indent=2))
    return exit_code([result])


if __name__ == "__main__":
    sys.exit(main())
