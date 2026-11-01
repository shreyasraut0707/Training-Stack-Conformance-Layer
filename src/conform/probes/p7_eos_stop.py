"""P7: after a short fine-tune, does the saved and reloaded model learn to emit EOS and stop?

Spec Sections 7 and 7.1. Mask inspection can mislead (an ignored EOS barely moved accuracy;
a suspected pad/EOS layout was benign), so P7 measures behaviour. It fine-tunes the proxy
briefly with the real trainer on short completions that end in EOS, saves and reloads the
model (with the model's real generation settings attached, so the saved settings are part
of the chain), then compares the probability of EOS at the end of each completion before
and after, and runs a bounded greedy generation. It fails if EOS probability does not rise
or generation does not stop. If training did not even learn the completions' content, the
result is unknown, never a fail: a too-short run must not pass for a stopping defect.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

from conform.environment import capture_environment
from conform.proxy_model import build_proxy
from conform.result import EvidenceClass, Finding, ProbeResult, Status, exit_code
from conform.stack import (
    DATA_FORMATS, SAMPLES, TRAINERS, TRANSFORMERS_RECIPE_LIMITATION, add_common_arguments,
    build_trainer, common_kwargs, conversations, reproducer as make_reproducer, template_ids,
)

PROBE_ID = "P7-eos-stopping"
ORACLE = ("behaviour: EOS probability at the end of each completion before and after a short "
          "fine-tune, on the saved and reloaded model, and a bounded greedy generate call")
STEPS = 60
LEARNING_RATE = 5e-3
EXTRA_NEW_TOKENS = 8
# "Training learned the completions" needs a clear margin, not a rounding-level rise; real
# training here raises the completions' token probability about 10,000-fold.
LEARNED_FACTOR = 10.0
LIMITATIONS = [
    "proxy model; real weights not used, so a short fine-tune stands in for a real one",
    f"training budget fixed by the probe ({STEPS} steps at learning rate {LEARNING_RATE}, constant); "
    "the recipe's other trainer settings apply",
    "generation is greedy (do_sample=False) for determinism; the saved generation settings' stop "
    "tokens and other options apply",
]


def _stop_ids(generation_config, tokenizer) -> list[int]:
    eos = generation_config.eos_token_id
    if eos is None:
        return []
    return sorted(set(eos if isinstance(eos, (list, tuple)) else [eos]))


def targets(tokenizer, convs):
    """Per sample: inference prompt ids, full template ids, and the position of the final EOS."""
    out = []
    eos = tokenizer.eos_token_id
    for prompt, completion in convs:
        full = template_ids(tokenizer, prompt + completion)
        prompt_ids = template_ids(tokenizer, prompt, add_generation_prompt=True)
        positions = [i for i, t in enumerate(full) if t == eos and i >= len(prompt_ids)]
        out.append({"prompt": prompt_ids, "full": full, "eos_at": positions[-1] if positions else None})
    return out


@torch.no_grad()
def measure(model, items, eos_id) -> dict:
    """Mean EOS probability at the end of each completion, and mean probability of the
    completion's own content tokens (did training learn anything at all?)."""
    model.eval()
    eos_p, content_p = [], []
    for item in items:
        if item["eos_at"] is None:
            continue
        ids = torch.tensor([item["full"][:item["eos_at"] + 1]])
        probs = torch.softmax(model(input_ids=ids).logits[0].float(), dim=-1)
        eos_p.append(probs[item["eos_at"] - 1, eos_id].item())
        start = len(item["prompt"])
        content = [probs[i - 1, item["full"][i]].item() for i in range(start, item["eos_at"])]
        if content:
            content_p.append(sum(content) / len(content))
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {"eos_probability": mean(eos_p), "content_probability": mean(content_p), "samples": len(eos_p)}


@torch.no_grad()
def generate(model, items, stop_ids, pad_id) -> list[dict]:
    model.eval()
    results = []
    for item in items:
        if item["eos_at"] is None:
            continue
        budget = item["eos_at"] + 1 - len(item["prompt"]) + EXTRA_NEW_TOKENS
        ids = torch.tensor([item["prompt"]])
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=budget,
                             do_sample=False, pad_token_id=pad_id)
        new = out[0, ids.shape[1]:].tolist()
        stopped_at = next((i for i, t in enumerate(new) if t in stop_ids), None)
        results.append({"budget": budget, "new_tokens": len(new), "stopped": stopped_at is not None,
                        "stopped_after": None if stopped_at is None else stopped_at + 1})
    return results


def run(model_id: str, *, trainer: str = "trl", seed: int = 0, data_format: str = "prompt_completion",
        samples=None, rows_path=None, trainer_args: dict | None = None, trainer_cls=None) -> ProbeResult:
    if trainer not in TRAINERS:
        raise ValueError(f"unknown trainer {trainer!r}; choose from {sorted(TRAINERS)}")
    if data_format not in DATA_FORMATS:
        raise ValueError(f"unknown data format {data_format!r}; choose from {DATA_FORMATS}")
    environment = capture_environment(device="cpu", dtype="float32", seeds={"proxy": seed, "trainer": seed})
    reproducer = make_reproducer("p7_eos_stop", model_id, trainer=trainer, data_format=data_format,
                                 seed=seed, rows_path=rows_path, trainer_args=trainer_args)
    limitations = LIMITATIONS + ([TRANSFORMERS_RECIPE_LIMITATION] if trainer == "transformers" else [])
    common = dict(oracle=ORACLE, environment=environment, reproducer=reproducer, limitations=limitations)
    convs = conversations(samples or SAMPLES)
    try:
        from transformers import AutoModelForCausalLM, GenerationConfig

        model, tokenizer, info = build_proxy(model_id, seed=seed)
        try:  # the model's real generation settings, so the saved settings are tested too
            model.generation_config = GenerationConfig.from_pretrained(model_id, local_files_only=True)
            generation_source = "the model's generation_config.json"
        except OSError:
            generation_source = "library default (no generation_config.json found)"
        items = targets(tokenizer, convs)
        eos_id = tokenizer.eos_token_id
        before = measure(model, items, eos_id)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as out:
            built, dropped = build_trainer(
                trainer, model, tokenizer, convs, output_dir=out, seed=seed, data_format=data_format,
                trainer_cls=trainer_cls,
                trainer_args={**(trainer_args or {}), "max_steps": STEPS, "learning_rate": LEARNING_RATE,
                              "lr_scheduler_type": "constant", "warmup_steps": 0, "weight_decay": 0.0,
                              "logging_steps": STEPS})
            built.train()
            saved = Path(out) / "saved"
            built.save_model(str(saved))
            reloaded = AutoModelForCausalLM.from_pretrained(
                str(saved), attn_implementation="eager").to(torch.float32)  # same kernel as the proxy
            stop_ids = _stop_ids(reloaded.generation_config, tokenizer)
            after = measure(reloaded, items, eos_id)
            generations = generate(reloaded, items, stop_ids, tokenizer.pad_token_id or eos_id)
    except Exception as exc:  # an API change must become not_tested, never a wrong verdict
        return ProbeResult(PROBE_ID, Status.NOT_TESTED, reason=f"{type(exc).__name__}: {exc}", **common)
    if dropped:
        limitations.append(f"this library version has no option(s) {dropped}; "
                           "they were left at the library default")
    if not before["samples"]:
        return ProbeResult(PROBE_ID, Status.NOT_TESTED, reason="no sample completion ends in the EOS token",
                           **common)

    stopped = sum(g["stopped"] for g in generations)
    rose = after["eos_probability"] > before["eos_probability"]
    learned = after["content_probability"] >= LEARNED_FACTOR * before["content_probability"]
    findings = [
        Finding(EvidenceClass.SEEN,
                f"EOS probability at the end of the completion: {before['eos_probability']:.3g} before, "
                f"{after['eos_probability']:.3g} after the fine-tune (mean of {before['samples']} samples)",
                evidence={"before": before["eos_probability"], "after": after["eos_probability"]}),
        Finding(EvidenceClass.SEEN,
                f"Greedy generation from the reloaded model stopped in {stopped} of {len(generations)} prompts "
                f"(stop tokens {stop_ids} from {generation_source})",
                evidence={"stopped": stopped, "prompts": len(generations), "stop_ids": stop_ids}),
        Finding(EvidenceClass.SEEN,
                f"Probability of the completions' own tokens: {before['content_probability']:.3g} before, "
                f"{after['content_probability']:.3g} after",
                evidence={"before": before["content_probability"], "after": after["content_probability"]}),
    ]
    if not learned:
        findings.append(Finding(EvidenceClass.UNKNOWN,
                                "The fine-tune did not learn the completions, so stopping cannot be judged"))
        status = Status.INFO
    elif not rose or stopped < len(generations):
        status = Status.FAIL
        findings.append(Finding(
            EvidenceClass.SEEN,
            "The saved and reloaded model did not learn to stop: "
            + ("EOS probability did not rise" if not rose else
               f"generation ran past the end in {len(generations) - stopped} of {len(generations)} prompts"),
            evidence={"eos_rose": rose, "all_stopped": stopped == len(generations)}))
    else:
        status = Status.PASS
    observed = {"trainer": TRAINERS[trainer], "data_format": data_format, "proxy": info.to_dict(),
                "before": before, "after": after, "generations": generations, "stop_ids": stop_ids,
                "generation_settings": generation_source, "steps": STEPS, "learning_rate": LEARNING_RATE,
                "unsupported_trainer_options": dropped}
    return ProbeResult(PROBE_ID, status, findings=findings, observed=observed, **common)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P7: does the fine-tuned model learn to stop")
    add_common_arguments(parser)
    parser.set_defaults(trainer="trl")
    args = parser.parse_args(argv)
    result = run(args.model_id, **common_kwargs(args))
    print(json.dumps(result.to_dict(), indent=2))
    return exit_code([result])


if __name__ == "__main__":
    sys.exit(main())
