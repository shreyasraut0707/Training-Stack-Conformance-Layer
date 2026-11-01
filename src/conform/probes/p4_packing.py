"""P4: does one packed document leak into another? (metamorphic)

Spec Section 7 (release v0.2). The trainer packs several short documents into one
sequence. P4 takes that real packed batch, measures document B's per-token loss, edits
document A's tokens and measures again: B's loss must not change. Boundaries come from
the batch's own position_ids (padding-free packing restarts them at each document).

Verdict (spec): fail if B's loss changes beyond the floor; not tested without the required
attention kernel. If B is isolated, the result is a pass on the kernel in use. If B is not
isolated and the kernel is not a varlen (flash) kernel, the result is not tested, with the
measured leak reported as a seen fact. A leak with a flash kernel is a fail.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile

import torch

from conform.environment import capture_environment
from conform.noise_floor import ABOVE, FLOAT32_RESOLUTION, INSIDE, classify, relative_gap
from conform.oracle import IGNORE_INDEX
from conform.proxy_model import build_proxy
from conform.result import EvidenceClass, Finding, ProbeResult, Status, exit_code
from conform.stack import (
    DATA_FORMATS, SAMPLES, TRAINERS, add_common_arguments, attention_kernel, build_trainer, common_kwargs,
    conversations, reproducer as make_reproducer,
)

PROBE_ID = "P4-packing-isolation"
ORACLE = "metamorphic: editing document A in a packed sequence must not change document B's per-token loss"
PACK_LENGTH = 512
BOUNDARY_KEYS = ("position_ids", "cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "seq_idx")
LIMITATIONS = [
    "proxy model; real weights not used",
    "CPU only: flash-attention varlen kernels are GPU-only, so isolation is checked on the kernel in use",
    f"packing as the trainer does it for this recipe (default: TRL packing, max_length {PACK_LENGTH})",
]


def documents(position_ids) -> list[tuple[int, int]]:
    """[start, end) spans of the documents in a packed row: position_ids restart at 0."""
    pos = position_ids.tolist()
    starts = [i for i, p in enumerate(pos) if p == 0] + [len(pos)]
    return [(a, b) for a, b in zip(starts, starts[1:]) if b - a > 1]


@torch.no_grad()
def doc_losses(model, inputs, labels, span) -> torch.Tensor:
    logits = model(**inputs).logits[0].double()
    start, end = span
    targets = [(t, labels[0, t].item()) for t in range(start + 1, end) if labels[0, t].item() != IGNORE_INDEX]
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.tensor([-log_probs[t - 1, y].item() for t, y in targets], dtype=torch.float64)


def measure_leak(model, batch, vocab: int, drop_boundaries: bool = False) -> dict:
    """Edit document A, measure document B. `drop_boundaries` withholds the document
    boundaries from the model (a deliberate break, for testing the measurement)."""
    model.eval()
    docs = documents(batch["position_ids"][0])
    if len(docs) < 2:
        raise LookupError("the packed batch holds fewer than two documents")
    a, b = docs[0], docs[1]
    inputs = {k: v for k, v in batch.items() if k not in ("labels", "seq_lengths")}
    if drop_boundaries:
        inputs = {k: v for k, v in inputs.items() if k not in BOUNDARY_KEYS}
    labels = batch["labels"]
    before = doc_losses(model, inputs, labels, b)
    repeat = doc_losses(model, inputs, labels, b)
    edited = batch["input_ids"].clone()
    edited[0, a[0] + 1:a[1]] = (edited[0, a[0] + 1:a[1]] + 1) % vocab
    after = doc_losses(model, {**inputs, "input_ids": edited}, labels, b)
    floor = max(relative_gap(repeat, before), FLOAT32_RESOLUTION)
    gap = relative_gap(after, before)
    return {"documents": len(docs), "doc_a": a, "doc_b": b, "b_tokens": len(before), "gap": gap,
            "floor": floor, "verdict": classify(gap, floor)}


def run(model_id: str, *, trainer: str = "trl", seed: int = 0, data_format: str = "prompt_completion",
        samples=None, rows_path=None, trainer_args: dict | None = None, trainer_cls=None) -> ProbeResult:
    if trainer not in TRAINERS:
        raise ValueError(f"unknown trainer {trainer!r}; choose from {sorted(TRAINERS)}")
    if data_format not in DATA_FORMATS:
        raise ValueError(f"unknown data format {data_format!r}; choose from {DATA_FORMATS}")
    environment = capture_environment(device="cpu", dtype="float32", seeds={"proxy": seed, "trainer": seed})
    reproducer = make_reproducer("p4_packing", model_id, trainer=trainer, data_format=data_format, seed=seed,
                                 rows_path=rows_path, trainer_args=trainer_args)
    limitations = list(LIMITATIONS)
    common = dict(oracle=ORACLE, environment=environment, reproducer=reproducer, limitations=limitations)
    if trainer == "transformers":
        return ProbeResult(PROBE_ID, Status.NOT_TESTED, reason="the Transformers Trainer has no packing of its own",
                           **common)
    try:
        model, tokenizer, info = build_proxy(model_id, seed=seed)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as out:
            built, dropped = build_trainer(
                trainer, model, tokenizer, conversations(samples or SAMPLES), output_dir=out, seed=seed,
                data_format=data_format, trainer_cls=trainer_cls,
                trainer_args={"packing": True, "max_length": PACK_LENGTH, **(trainer_args or {})})
            batch = next(iter(built.get_train_dataloader()))
            kernel, fragile = attention_kernel(built.model)
            kernel = kernel or "unknown"
            if fragile:
                limitations.append("fragile: the attention kernel's name is read from a private "
                                   "attribute (config._attn_implementation) through an adapter")
            if "position_ids" not in batch:
                return ProbeResult(PROBE_ID, Status.NOT_TESTED, reason=(
                    "this packing mode passes no document boundaries (no position_ids), so packed "
                    "documents share attention by design"), **common)
            leak = measure_leak(built.model, batch, len(tokenizer))
    except LookupError as exc:
        return ProbeResult(PROBE_ID, Status.NOT_TESTED, reason=str(exc), **common)
    except Exception as exc:  # an API change must become not_tested, never a wrong verdict
        return ProbeResult(PROBE_ID, Status.NOT_TESTED, reason=f"{type(exc).__name__}: {exc}", **common)
    if dropped:
        limitations.append(f"this library version has no option(s) {dropped}; "
                           "they were left at the library default")

    fact = Finding(
        EvidenceClass.SEEN if leak["verdict"] in (INSIDE, ABOVE) else EvidenceClass.UNKNOWN,
        f"Editing document A changed document B's per-token loss by {leak['gap']:.3g} (floor {leak['floor']:.3g}; "
        f"{leak['documents']} documents packed, attention kernel '{kernel}')",
        evidence={"metric": "document_b_loss", "deviation": leak["gap"], "floor": leak["floor"],
                  "verdict": leak["verdict"], "kernel": kernel})
    findings = [fact]
    observed = {"trainer": TRAINERS[trainer], "proxy": info.to_dict(), "kernel": kernel, **leak}
    if leak["verdict"] == INSIDE:
        return ProbeResult(PROBE_ID, Status.PASS, findings=findings, observed=observed, **common)
    if leak["verdict"] == ABOVE and "flash" not in kernel:
        return ProbeResult(PROBE_ID, Status.NOT_TESTED, findings=findings, observed=observed, reason=(
            f"packed documents are not isolated with the '{kernel}' attention kernel; the required varlen "
            "(flash) kernel is not available here"), **common)
    if leak["verdict"] == ABOVE:
        findings.append(Finding(EvidenceClass.INFERRED, "Packed documents influence each other during training",
                                rule="a document's loss that changes when another document is edited means "
                                     "attention crosses document boundaries"))
        return ProbeResult(PROBE_ID, Status.FAIL, findings=findings, observed=observed, **common)
    return ProbeResult(PROBE_ID, Status.INFO, findings=findings, observed=observed, **common)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P4: packed-document isolation")
    add_common_arguments(parser)
    parser.set_defaults(trainer="trl")
    args = parser.parse_args(argv)
    result = run(args.model_id, **common_kwargs(args))
    print(json.dumps(result.to_dict(), indent=2))
    return exit_code([result])


if __name__ == "__main__":
    sys.exit(main())
