"""Noise floor: how much two mathematically identical computations differ (spec Section 7.2).

The floor is measured, not assumed: the oracle's reference step is re-run with the
micro-batches in reverse order and in float32, and the largest gap to the float64
reference is the floor. A stack fails only when it is clearly above that floor.

The loss floor is never below float32's resolution, because trainers report their
loss as a float32 number and cannot be more precise than that. The measured floor is
the largest of only a few samples of rounding noise, and the stack under test is one
more sample, so a stack within a small band above the floor still counts as inside.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from conform.oracle import reference_loss

# Calibrated in M2 (docs/calibration.md): the 10x fail multiplier is confirmed; the 2x pass
# band is kept by the owner's decision (2026-11-01). Names kept for compatibility.
PROVISIONAL_PASS_BAND = 2.0
PROVISIONAL_MULTIPLIER = 10.0

# Relative precision of a float32 number (about 1.2e-7).
FLOAT32_RESOLUTION = float(torch.finfo(torch.float32).eps)

INSIDE, UNCLEAR, ABOVE = "inside", "unclear", "above"


@dataclass
class Floor:
    loss: float
    gradient: float


def relative_gap(value, reference) -> float:
    """|value - reference| / |reference| for scalars or flattened tensors."""
    value = torch.as_tensor(value, dtype=torch.float64)
    reference = torch.as_tensor(reference, dtype=torch.float64)
    gap = torch.linalg.vector_norm(value - reference)
    scale = torch.linalg.vector_norm(reference)
    return float(gap / scale) if scale > 0 else float(gap)


def classify(
    deviation: float,
    floor: float,
    multiplier: float = PROVISIONAL_MULTIPLIER,
    pass_band: float = PROVISIONAL_PASS_BAND,
) -> str:
    """Spec Figure 7: inside the floor, unclear (report unknown), or clearly above."""
    if deviation <= floor * pass_band:
        return INSIDE
    if deviation <= floor * multiplier:
        return UNCLEAR
    return ABOVE


def flat_gradient(params) -> torch.Tensor:
    """All gradients of `params` as one float64 vector (zeros where a gradient is missing)."""
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).to(torch.float64)
        for p in params
    ])


def reference_step(model, micro_batches, *, dtype=torch.float64, reverse=False):
    """Oracle loss and gradient for one optimizer step over `micro_batches`, at `dtype`.

    The model is only used for its forward pass (no labels are passed, so none of the
    library's loss code runs); the loss itself comes from the independent oracle.
    """
    ref = copy.deepcopy(model).to(dtype)
    ref.train()
    ref.zero_grad(set_to_none=True)
    batches = list(reversed(micro_batches)) if reverse else list(micro_batches)
    pairs = [
        (ref(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits, b["labels"])
        for b in batches
    ]
    loss = reference_loss(pairs, dtype=dtype)
    loss.backward()
    return loss.item(), flat_gradient(ref.parameters())


def measure_floor(model, micro_batches):
    """Return (reference loss, reference gradient, Floor) at float64."""
    ref_loss, ref_grad = reference_step(model, micro_batches)
    variants = [
        reference_step(model, micro_batches, reverse=True),
        reference_step(model, micro_batches, dtype=torch.float32),
        reference_step(model, micro_batches, dtype=torch.float32, reverse=True),
    ]
    floor = Floor(
        loss=max(FLOAT32_RESOLUTION, *(relative_gap(loss, ref_loss) for loss, _ in variants)),
        gradient=max(relative_gap(grad, ref_grad) for _, grad in variants),
    )
    return ref_loss, ref_grad, floor
