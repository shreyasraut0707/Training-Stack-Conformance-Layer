"""P3: are the tokens used in training the same tokens inference would see?

Spec Section 7. The oracle is the tokenizer's own chat template at inference time, with
the generation prompt: that is what a deployed model is fed. For every row of the
trainer's first real batch, the inference prompt must be an exact prefix of the training
sequence. A prefix mismatch (for example an extra BOS, or a turn boundary rendered
differently) fails. Tokens the trainer adds after the template's end are reported as facts
but never fail on their own: the spec forbids failing on a static heuristic alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher

from conform.environment import capture_environment
from conform.proxy_model import build_proxy
from conform.result import EvidenceClass, Finding, ProbeResult, Status, exit_code
from conform.stack import (
    DATA_FORMATS, SAMPLES, TRAINERS, add_common_arguments, common_kwargs, conversations,
    first_batch, real_ids, reproducer as make_reproducer, template_ids,
)

PROBE_ID = "P3-train-inference-tokens"
ORACLE = "the tokenizer's own chat template at inference time, with the generation prompt"
LIMITATIONS = [
    "proxy model; real weights not used",
    "inference is taken to be apply_chat_template(..., add_generation_prompt=True); a serving "
    "stack that renders prompts differently is not tested",
]
NO_TEMPLATE_REASON = ("the Transformers Trainer applies no chat template of its own: the training "
                      "tokens come from the user's preprocessing, so there is no trainer path to compare")


def _common_prefix(a, b) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def compare_row(train, prompt, full) -> dict:
    """Facts for one training row against its inference prompt and full template rendering."""
    prefix = _common_prefix(train, prompt)
    fact = {"training_tokens": len(train), "prompt_tokens": len(prompt), "prefix_match": prefix == len(prompt),
            "first_difference": None if prefix == len(prompt) else prefix}
    if train == full:
        fact["after_template"] = "identical to the template"
    elif train[:len(full)] == full:
        fact["after_template"] = "extra tokens"
        fact["extra_tokens"] = train[len(full):]
    else:
        fact["after_template"] = "differs from the template"
    return fact


def match_rows(batch_rows, fulls):
    """Pair each training row with the sample whose full template rendering it is most similar
    to; None when two samples tie (ambiguous). Similarity, not a shared prefix, so a broken
    prefix (the very thing P3 checks) cannot prevent the match."""
    matches = []
    for train in batch_rows:
        scores = [SequenceMatcher(None, train, full, autojunk=False).ratio() for full in fulls]
        best = max(scores)
        matches.append(scores.index(best) if scores.count(best) == 1 else None)
    return matches


def run(model_id: str, *, trainer: str = "trl", seed: int = 0, data_format: str = "prompt_completion",
        samples=None, rows_path=None, trainer_args: dict | None = None, trainer_cls=None) -> ProbeResult:
    if trainer not in TRAINERS:
        raise ValueError(f"unknown trainer {trainer!r}; choose from {sorted(TRAINERS)}")
    if data_format not in DATA_FORMATS:
        raise ValueError(f"unknown data format {data_format!r}; choose from {DATA_FORMATS}")
    environment = capture_environment(device="cpu", dtype="float32", seeds={"proxy": seed, "trainer": seed})
    reproducer = make_reproducer("p3_template", model_id, trainer=trainer, data_format=data_format,
                                 seed=seed, rows_path=rows_path, trainer_args=trainer_args)
    limitations = list(LIMITATIONS)
    common = dict(oracle=ORACLE, environment=environment, reproducer=reproducer, limitations=limitations)
    if trainer == "transformers":
        return ProbeResult(PROBE_ID, Status.NOT_TESTED, reason=NO_TEMPLATE_REASON, **common)
    convs = conversations(samples or SAMPLES)
    try:
        model, tokenizer, info = build_proxy(model_id, seed=seed)
        batch, dropped = first_batch(trainer, model, tokenizer, convs, seed, data_format, trainer_args,
                                     trainer_cls)
        prompts = [template_ids(tokenizer, p, add_generation_prompt=True) for p, _ in convs]
        fulls = [template_ids(tokenizer, p + c) for p, c in convs]
        rows = [real_ids(batch, i) for i in range(batch["input_ids"].shape[0])]
    except Exception as exc:  # an API change must become not_tested, never a wrong verdict
        return ProbeResult(PROBE_ID, Status.NOT_TESTED, reason=f"{type(exc).__name__}: {exc}", **common)
    if dropped:
        limitations.append(f"this library version has no option(s) {dropped}; "
                           "they were left at the library default")

    facts, findings = [], []
    for train, sample in zip(rows, match_rows(rows, fulls)):
        if sample is None:
            facts.append({"sample": None})
            continue
        fact = {"sample": sample, **compare_row(train, prompts[sample], fulls[sample])}
        if not fact["prefix_match"]:
            at = fact["first_difference"]
            fact["training_token_there"] = tokenizer.convert_ids_to_tokens(train[at]) if at < len(train) else None
            fact["inference_token_there"] = tokenizer.convert_ids_to_tokens(prompts[sample][at])
        facts.append(fact)

    matched = [f for f in facts if f["sample"] is not None]
    mismatched = [f for f in matched if not f["prefix_match"]]
    findings.append(Finding(
        EvidenceClass.SEEN,
        f"The inference prompt is an exact prefix of the training sequence in "
        f"{len(matched) - len(mismatched)} of {len(matched)} rows",
        evidence={"rows_checked": len(matched), "rows_mismatched": len(mismatched)}))
    for f in mismatched:
        findings.append(Finding(
            EvidenceClass.SEEN,
            f"Sample {f['sample']}: training and inference tokens first differ at position "
            f"{f['first_difference']} (training {f['training_token_there']!r}, inference "
            f"{f['inference_token_there']!r})",
            evidence={k: f[k] for k in ("sample", "first_difference", "training_token_there",
                                        "inference_token_there")}))
    extra = [f for f in matched if f["after_template"] == "extra tokens"]
    if extra:
        added = sorted({tok for f in extra for tok in tokenizer.convert_ids_to_tokens(f["extra_tokens"])})
        findings.append(Finding(
            EvidenceClass.SEEN,
            f"The trainer adds tokens after the chat template's end in {len(extra)} of {len(matched)} "
            f"rows: {added}",
            evidence={"rows": len(extra), "tokens": added}))
    differs = [f for f in matched if f["after_template"] == "differs from the template"]
    if differs:
        findings.append(Finding(
            EvidenceClass.SEEN,
            f"After the prompt, the training sequence differs from the template's rendering in "
            f"{len(differs)} of {len(matched)} rows", evidence={"rows": len(differs)}))
    if len(matched) < len(facts):
        findings.append(Finding(EvidenceClass.UNKNOWN,
                                f"{len(facts) - len(matched)} of {len(facts)} rows could not be matched "
                                "to a sample"))

    if mismatched:
        status = Status.FAIL
        findings.append(Finding(
            EvidenceClass.INFERRED,
            "The model is trained on a context that differs from the one it sees at inference",
            rule="an inference prompt that is not a prefix of the training sequence means the "
                 "training and inference contexts differ"))
    elif len(matched) < len(facts) or not matched:
        status = Status.INFO
    else:
        status = Status.PASS
    observed = {"trainer": TRAINERS[trainer], "data_format": data_format, "proxy": info.to_dict(),
                "rows": facts, "unsupported_trainer_options": dropped}
    return ProbeResult(PROBE_ID, status, findings=findings, observed=observed, **common)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P3: training tokens versus inference tokens")
    add_common_arguments(parser)
    parser.set_defaults(trainer="trl")
    args = parser.parse_args(argv)
    result = run(args.model_id, **common_kwargs(args))
    print(json.dumps(result.to_dict(), indent=2))
    return exit_code([result])


if __name__ == "__main__":
    sys.exit(main())
