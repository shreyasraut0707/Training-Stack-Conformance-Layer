"""Independent oracle for loss maths (spec Section 6, Figure 6).

The referee: computes the correct causal-LM loss itself, in float64. It does not
call any Transformers or TRL loss or masking code (not even
torch.nn.functional.cross_entropy, which the Transformers loss uses), so a defect
in the library cannot cancel out. It does use basic PyTorch tensor maths, which
keeps it differentiable for comparing parameter updates in P1. It works in float64
by default; float32 is used only to measure the noise floor.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

IGNORE_INDEX = -100


def token_losses(
    logits: torch.Tensor, labels: torch.Tensor, dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position negative log-likelihood and the mask of supervised positions.

    The causal shift happens here: logits at position t are scored against the label
    at t+1. logits: (batch, seq, vocab); labels: (batch, seq). Both outputs are
    (batch, seq - 1).
    """
    logits = logits[:, :-1, :].to(dtype)
    targets = labels[:, 1:]
    mask = targets != IGNORE_INDEX
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    nll = -log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return torch.where(mask, nll, torch.zeros_like(nll)), mask


def loss_sum_and_count(
    logits: torch.Tensor, labels: torch.Tensor, dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, int]:
    """Summed loss and number of supervised tokens for one micro-batch."""
    nll, mask = token_losses(logits, labels, dtype)
    return nll.sum(), int(mask.sum())


def accumulated_loss(sums_and_counts: Sequence[tuple[torch.Tensor | float, int]]) -> torch.Tensor:
    """Correct: summed loss over all micro-batches / total supervised tokens."""
    total_tokens = sum(n for _, n in sums_and_counts)
    if total_tokens == 0:
        raise ValueError("no supervised tokens: every label is -100")
    total_loss = sum(torch.as_tensor(s, dtype=torch.float64) for s, _ in sums_and_counts)
    return total_loss / total_tokens


def mean_of_means(sums_and_counts: Sequence[tuple[torch.Tensor | float, int]]) -> torch.Tensor:
    """The known defective formula: the average of per-micro-batch averages.

    Kept only to build injected defects for calibration (spec Section 10.2, step 3).
    Never used as a reference.
    """
    if any(n == 0 for _, n in sums_and_counts):
        raise ValueError("a micro-batch has no supervised tokens")
    means = [torch.as_tensor(s, dtype=torch.float64) / n for s, n in sums_and_counts]
    return sum(means) / len(means)


def reference_loss(
    micro_batches: Iterable[tuple[torch.Tensor, torch.Tensor]], dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Reference loss for one optimizer step made of (logits, labels) micro-batches."""
    return accumulated_loss([loss_sum_and_count(lg, lb, dtype) for lg, lb in micro_batches])
