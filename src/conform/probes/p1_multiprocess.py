"""P1M: is the loss averaged correctly across processes? (multi-process CPU probe)

Spec Section 11 (M4: "a multi-process CPU probe for loss averaging"). Two real training
processes (PyTorch's CPU "gloo" backend, through the Trainer's own distributed path) each
take rows with a different number of supervised tokens. After the processes average their
gradients, the update must match the independent float64 reference over all the rows of
that step: summed token loss / total supervised tokens across both processes. A per-process
normalisation (mean of the processes' means) deviates when the processes' token counts differ.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from conform.environment import capture_environment
from conform.noise_floor import ABOVE, UNCLEAR, classify, measure_floor, relative_gap
from conform.probes.p1_grad_accum import (
    LEARNING_RATE, _CapturingSGD, _training_kwargs, build_rows, supervised_targets,
)
from conform.proxy_model import build_proxy
from conform.result import EvidenceClass, Finding, ProbeResult, Status, exit_code

PROBE_ID = "P1M-multiprocess-loss-averaging"
ORACLE = ("float64 reference over the rows of both processes: summed token loss / total supervised "
          "tokens of the step (conform.oracle)")
WORLD_SIZE = 2
LIMITATIONS = [
    "proxy model; real weights not used",
    "two processes on one CPU with the gloo backend; multi-GPU, NCCL, DeepSpeed and FSDP are not tested",
    "Transformers Trainer only; the update is the averaged gradient handed to the optimizer (plain SGD)",
]


class _RowCollator:
    """Records which rows this process trains on, then collates without the row id."""

    def __init__(self, inner):
        self.inner, self.rows = inner, []

    def __call__(self, features):
        self.rows += [int(f["row_id"]) for f in features]
        return self.inner([{k: v for k, v in f.items() if k != "row_id"} for f in features])


def worker(model_id, seed, out, trainer_args) -> None:
    """One training process; WORLD_SIZE, RANK and the rendezvous come from the environment."""
    from transformers import Trainer, TrainingArguments, default_data_collator

    rank = int(os.environ["RANK"])
    model, tokenizer, _ = build_proxy(model_id, seed=seed)
    pad = next(i for i in (tokenizer.pad_token_id, tokenizer.eos_token_id, 0) if i is not None)
    rows = [{**row, "row_id": i} for i, row in enumerate(build_rows(len(tokenizer), pad, seed))]
    optimizer = _CapturingSGD(model.parameters(), lr=LEARNING_RATE)
    kwargs = {**_training_kwargs(out, batch_size=1, accumulation=1, seed=seed), "ddp_backend": "gloo",
              **(trainer_args or {})}
    built = Trainer(model=model, args=TrainingArguments(**kwargs), train_dataset=rows,
                    data_collator=default_data_collator, optimizers=(optimizer, None))
    built.data_collator = _RowCollator(default_data_collator)
    built.train()
    Path(out, f"rank{rank}.json").write_text(json.dumps({"rows": built.data_collator.rows[:1]}), encoding="utf-8")
    if rank == 0:
        torch.save(optimizer.gradient, Path(out, "gradient.pt"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(model_id: str, *, seed: int = 0, trainer: str = "transformers", trainer_args: dict | None = None,
        timeout: int = 600, **_ignored) -> ProbeResult:
    environment = capture_environment(device="cpu (2 processes, gloo)", dtype="float32",
                                      seeds={"proxy": seed, "trainer": seed})
    flags = f" --trainer-args '{json.dumps(trainer_args, sort_keys=True)}'" if trainer_args else ""
    reproducer = f"python -m conform.probes.p1_multiprocess {model_id} --seed {seed}{flags}"
    limitations = list(LIMITATIONS)
    common = dict(oracle=ORACLE, environment=environment, reproducer=reproducer, limitations=limitations)
    if trainer != "transformers":
        return ProbeResult(PROBE_ID, Status.NOT_TESTED, reason="the multi-process probe covers the "
                           "Transformers Trainer only in v0.2", **common)
    try:
        # The proxy for the reference is built only after the workers finish, so this process
        # holds no model while the two training processes run (memory matters on small machines).
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as out:
            port = _free_port()
            procs = []
            for rank in range(WORLD_SIZE):
                env = {**os.environ, "WORLD_SIZE": str(WORLD_SIZE), "RANK": str(rank), "LOCAL_RANK": str(rank),
                       "LOCAL_WORLD_SIZE": str(WORLD_SIZE), "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                       "USE_LIBUV": "0", "PYTHONIOENCODING": "utf-8"}
                command = [sys.executable, "-m", "conform.probes.p1_multiprocess", model_id, "--seed", str(seed),
                           "--worker", out]
                if trainer_args:
                    command += ["--trainer-args", json.dumps(trainer_args)]
                procs.append(subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                              text=True, encoding="utf-8", errors="replace"))
            errors = []
            for proc in procs:
                _, err = proc.communicate(timeout=timeout)
                if proc.returncode != 0:
                    errors.append((err.strip().splitlines() or ["no error output"])[-1])
            if errors:
                raise RuntimeError(f"a training process failed: {errors[0]}")
            used = [json.loads(Path(out, f"rank{r}.json").read_text(encoding="utf-8"))["rows"][0]
                    for r in range(WORLD_SIZE)]
            gradient = torch.load(Path(out, "gradient.pt"))
        model, tokenizer, info = build_proxy(model_id, seed=seed)
        pad = next(i for i in (tokenizer.pad_token_id, tokenizer.eos_token_id, 0) if i is not None)
        rows = build_rows(len(tokenizer), pad, seed)
        step_rows = [rows[i] for i in used]
        micro_batches = [{k: v.unsqueeze(0) for k, v in row.items()} for row in step_rows]
        _, ref_grad, floor = measure_floor(copy.deepcopy(model), micro_batches)
    except Exception as exc:  # a platform or API problem must become not_tested, never a wrong verdict
        return ProbeResult(PROBE_ID, Status.NOT_TESTED, reason=f"{type(exc).__name__}: {exc}", **common)

    deviation = relative_gap(gradient, ref_grad)
    verdict = classify(deviation, floor.gradient)
    counts = supervised_targets(step_rows)
    findings = [Finding(
        EvidenceClass.UNKNOWN if verdict == UNCLEAR else EvidenceClass.SEEN,
        f"The update averaged across {WORLD_SIZE} processes differs from the float64 reference by {deviation:.3g} "
        f"(floor {floor.gradient:.3g}; supervised tokens per process {counts})",
        evidence={"metric": "update", "deviation": deviation, "floor": floor.gradient, "verdict": verdict,
                  "tokens_per_process": counts})]
    if verdict == ABOVE:
        status = Status.FAIL
        findings.append(Finding(EvidenceClass.INFERRED,
                                "Multi-process training does not optimise the same objective as one process",
                                rule="an averaged update clearly above the noise floor means the processes' "
                                     "losses are not normalised by the total token count"))
    else:
        status = Status.PASS if verdict != UNCLEAR else Status.INFO
    observed = {"proxy": info.to_dict(), "rows_per_process": used, "tokens_per_process": counts,
                "deviation": deviation, "floor": floor.gradient, "trainer_args": trainer_args or {}}
    return ProbeResult(PROBE_ID, status, findings=findings, observed=observed, noise_floor=floor.gradient, **common)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P1M: loss averaging across processes")
    parser.add_argument("model_id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trainer", default="transformers")
    parser.add_argument("--trainer-args", help="JSON object of trainer settings from the recipe")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    trainer_args = json.loads(args.trainer_args) if args.trainer_args else None
    if args.worker:
        worker(args.model_id, args.seed, args.worker, trainer_args)
        return 0
    result = run(args.model_id, seed=args.seed, trainer=args.trainer, trainer_args=trainer_args)
    print(json.dumps(result.to_dict(), indent=2))
    return exit_code([result])


if __name__ == "__main__":
    sys.exit(main())
