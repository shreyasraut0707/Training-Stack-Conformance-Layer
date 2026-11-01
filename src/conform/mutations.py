"""Injected defects ("mutants") to measure what the probes catch (spec Section 10.2, step 4).

Each mutant is a small subclass of the real trainer that breaks one thing. This evidence is
weaker than the historical bank cases, because this project constructs the defects; reports
say so. A mutant is "caught" when its intended probe fails (or, for an expected-report
mutant, when the probe reports the change as a fact without failing).
"""
from __future__ import annotations

import importlib

import torch

IGNORE = -100


def _sft():
    from trl import SFTTrainer
    return SFTTrainer


def _hf():
    from transformers import Trainer
    return Trainer


def mean_of_means(base):
    class MeanOfMeans(base):
        """Withhold num_items_in_batch: each micro-batch is averaged on its own (the 2024 bug)."""

        def training_step(self, model, inputs, num_items_in_batch=None):
            return super().training_step(model, inputs, num_items_in_batch=None)
    return MeanOfMeans


def position_zero_count(base):
    class PositionZeroCount(base):
        """Count every non-ignored label, including position 0 (transformers 4.46 to 5.9)."""

        def _get_num_items_in_batch(self, batch_samples, device):
            return sum(b["labels"].ne(IGNORE).sum() for b in batch_samples).to(device)
    return PositionZeroCount


def loss_scale(eps):
    def factory(base):
        class LossScale(base):
            __doc__ = f"Multiply the loss by (1 + {eps}): a small normalisation error."

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                out = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                           num_items_in_batch=num_items_in_batch)
                if return_outputs:
                    return out[0] * (1 + eps), out[1]
                return out * (1 + eps)
        return LossScale
    return factory


def ignore_one_microbatch(base):
    class IgnoreOneMicrobatch(base):
        """The second micro-batch's labels are all ignored, but it is still counted."""

        def training_step(self, model, inputs, num_items_in_batch=None):
            self._mutant_calls = getattr(self, "_mutant_calls", 0) + 1
            if self._mutant_calls == 2:
                inputs = {**inputs, "labels": torch.full_like(inputs["labels"], IGNORE)}
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
    return IgnoreOneMicrobatch


def _prepared(edit, doc):
    """An SFTTrainer whose prepared dataset rows are edited by `edit(example, tokenizer)`."""
    def factory(base):
        class Prepared(base):
            __doc__ = doc

            def _prepare_dataset(self, dataset, processing_class, *args, **kwargs):
                dataset = super()._prepare_dataset(dataset, processing_class, *args, **kwargs)
                return dataset.map(lambda ex: edit(dict(ex), processing_class))
        return Prepared
    return factory


def _drop_eos(ex, tok):
    ex["labels"] = [IGNORE if t == tok.eos_token_id else lab for t, lab in zip(ex["input_ids"], ex["labels"])]
    return ex


def _unmask_prompt(ex, tok):
    ex["labels"] = list(ex["input_ids"])
    return ex


def _double_bos(ex, tok):
    """Make the row start with at least two BOS tokens (one more than it had, and never fewer
    than two: a single BOS is allowed). Needs a tokenizer that defines a BOS token."""
    bos = tok.bos_token_id
    lead = 0
    while lead < len(ex["input_ids"]) and ex["input_ids"][lead] == bos:
        lead += 1
    add = max(1, 2 - lead)
    ex["input_ids"], ex["labels"] = [bos] * add + ex["input_ids"], [IGNORE] * add + ex["labels"]
    return ex


def _change_boundary(ex, tok):
    """Replace the last prompt token (the end of the generation prompt) with another token."""
    start = next((i for i, lab in enumerate(ex["labels"]) if lab != IGNORE), None)
    if start:
        ids = list(ex["input_ids"])
        ids[start - 1] = tok.eos_token_id if ids[start - 1] != tok.eos_token_id else 0
        ex["input_ids"] = ids
    return ex


def _extra_eos(ex, tok):
    ex["input_ids"], ex["labels"] = ex["input_ids"] + [tok.eos_token_id], ex["labels"] + [tok.eos_token_id]
    return ex


def wrong_generation_eos(base):
    class WrongGenerationEOS(base):
        """The saved generation settings name a stop token the model never emits."""

        def save_model(self, output_dir=None, _internal_call=False):
            tok = self.processing_class
            wrong = tok.bos_token_id if tok.bos_token_id not in (None, tok.eos_token_id) else 0
            self.model.generation_config.eos_token_id = [wrong]
            return super().save_model(output_dir, _internal_call)
    return WrongGenerationEOS


# id, intended probe, trainers, factory, probe keyword arguments, expectation.
MUTANTS = [
    ("mean-of-means", "P1", ("transformers", "trl"), mean_of_means, {}, "fail"),
    ("position-zero-count", "P1", ("transformers",), position_zero_count, {}, "fail"),
    ("loss-scale-1e-3", "P1", ("transformers", "trl"), loss_scale(1e-3), {}, "fail"),
    ("loss-scale-1e-5", "P1", ("transformers", "trl"), loss_scale(1e-5), {}, "fail"),
    ("ignore-one-microbatch", "P1", ("transformers", "trl"), ignore_one_microbatch, {}, "fail"),
    ("drop-eos-labels", "P2", ("trl",), _prepared(_drop_eos, "EOS labels set to -100"),
     {"intent": {"supervise_eos": True}}, "fail"),
    ("drop-eos-labels", "P7", ("trl",), _prepared(_drop_eos, "EOS labels set to -100"), {}, "fail"),
    ("unmask-prompt", "P2", ("trl",), _prepared(_unmask_prompt, "prompt tokens carry loss"),
     {"intent": {"mask_prompt": True}}, "fail"),
    ("double-bos", "P3", ("trl",), _prepared(_double_bos, "an extra BOS is prepended"), {}, "fail"),
    ("double-bos", "P2", ("trl",), _prepared(_double_bos, "an extra BOS is prepended"),
     {"intent": {"single_bos": True}}, "fail"),
    ("change-turn-boundary", "P3", ("trl",), _prepared(_change_boundary, "the generation prompt's last "
                                                       "token is replaced"), {}, "fail"),
    ("extra-trailing-eos", "P3", ("trl",), _prepared(_extra_eos, "an EOS is appended"), {}, "report"),
    ("wrong-generation-eos", "P7", ("trl",), wrong_generation_eos, {}, "fail"),
]
PROBE_MODULES = {"P1": "p1_grad_accum", "P2": "p2_loss_tokens", "P3": "p3_template", "P7": "p7_eos_stop"}
# Mutants that only exist for some tokenizers: they are reported as not applicable elsewhere.
REQUIRES_BOS = {"double-bos"}


def _has_bos(model_id: str) -> bool:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, local_files_only=True).bos_token_id is not None


def run_mutant(mutant, model_id: str, trainer: str, seed: int = 0) -> dict:
    """Run one mutant through its intended probe and judge whether it was caught."""
    mid, probe, _trainers, factory, kwargs, expect = mutant
    row = {"mutant": mid, "probe": probe, "trainer": trainer, "model": model_id, "expect": expect}
    if mid in REQUIRES_BOS and not _has_bos(model_id):
        return {**row, "status": "not applicable", "caught": None,
                "reason": "the tokenizer defines no BOS token, so a double BOS cannot occur"}
    base = _hf() if trainer == "transformers" else _sft()
    run = importlib.import_module(f"conform.probes.{PROBE_MODULES[probe]}").run
    result = run(model_id, trainer=trainer, seed=seed, trainer_cls=factory(base), **kwargs)
    if expect == "fail":
        caught = result.status.value == "fail"
    else:  # reported as a seen fact without failing
        caught = result.status.value != "fail" and any(
            f.evidence_class.value == "seen" and "after the chat template" in f.claim for f in result.findings)
    return {**row, "status": result.status.value, "caught": caught, "reason": result.reason}


def run_all(model_ids, seed: int = 0) -> list[dict]:
    return [run_mutant(m, model_id, trainer, seed) for m in MUTANTS for trainer in m[2] for model_id in model_ids]
