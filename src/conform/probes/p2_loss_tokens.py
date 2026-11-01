"""P2: which tokens carry loss, and does that match the declared intent?

Spec Section 7. P2 takes the first real batch from the trainer's own dataloader, so
the installed version's preprocessing, chat template, masking and collation are all
exercised, and reports what it sees in the labels: supervised spans, whether the final
EOS carries loss, how many BOS tokens start each row, prompt tokens that carry loss,
and real tokens that share the pad ID. These are facts. P2 fails only against an
intent the user declares; with no intent it reports facts and an unknown.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

from conform.environment import capture_environment
from conform.oracle import IGNORE_INDEX
from conform.proxy_model import build_proxy
from conform.result import EvidenceClass, Finding, ProbeResult, Status, exit_code
from conform.stack import (
    DATA_FORMATS, SAMPLES, TRAINERS, TRANSFORMERS_RECIPE_LIMITATION, add_common_arguments,
    common_kwargs, conversations as make_conversations, first_batch, reproducer as make_reproducer,
    supported as _supported, template_ids as _template_ids,
)

PROBE_ID = "P2-loss-bearing-tokens"
ORACLE = ("the labels tensor of the trainer's first real batch, compared with the declared "
          "intent; prompt boundaries from the tokenizer's own chat template")
INTENT_KEYS = ("supervise_eos", "mask_prompt", "single_bos")
LIMITATIONS = [
    "proxy model; real weights not used",
    "packing and padding-free batches not tested unless the recipe enables them (P4)",
]
BUILT_IN_ROWS_LIMITATION = "built-in sample conversations, not the user's rows"


@dataclass
class RowFacts:
    real_tokens: int
    prompt_tokens: int | None  # None when the row could not be matched to a sample
    supervised_spans: list[tuple[int, int]]  # [start, end) positions that carry loss
    supervised_prompt_tokens: int | None
    leading_bos: int | None  # None when the tokenizer defines no BOS
    final_eos_supervised: bool | None  # None when the row has no real EOS
    pad_id_real_tokens: int  # real (attended) tokens whose id equals the pad id
    pad_id_real_tokens_masked: int


def analyse_row(input_ids, labels, attention_mask, *, prompt_tokens, bos_id, eos_id, pad_id) -> RowFacts:
    """Facts about one row. Position 0 never carries loss: it is never a prediction target."""
    ids, lab = input_ids.tolist(), labels.tolist()
    real = attention_mask.bool().tolist() if attention_mask is not None else [True] * len(ids)
    carries = [i > 0 and label != IGNORE_INDEX for i, label in enumerate(lab)]
    real_positions = [i for i, r in enumerate(real) if r]

    spans, start = [], None
    for i, c in enumerate(carries + [False]):
        if c and start is None:
            start = i
        elif not c and start is not None:
            spans.append((start, i))
            start = None

    leading_bos = None
    if bos_id is not None:
        leading_bos = 0
        for i in real_positions:
            if ids[i] != bos_id:
                break
            leading_bos += 1

    eos_positions = [i for i in real_positions if ids[i] == eos_id]
    pad_positions = [i for i in real_positions if pad_id is not None and ids[i] == pad_id]
    prompt_positions = real_positions[:prompt_tokens] if prompt_tokens is not None else None
    return RowFacts(
        real_tokens=len(real_positions),
        prompt_tokens=prompt_tokens,
        supervised_spans=spans,
        supervised_prompt_tokens=(
            None if prompt_positions is None else sum(carries[i] for i in prompt_positions)
        ),
        leading_bos=leading_bos,
        final_eos_supervised=carries[eos_positions[-1]] if eos_positions else None,
        pad_id_real_tokens=len(pad_positions),
        pad_id_real_tokens_masked=sum(not carries[i] for i in pad_positions if i > 0),
    )


def _prompt_lengths(tokenizer, conversations) -> dict[tuple[int, ...], int]:
    """Map each full tokenized conversation to its prompt length, from the chat template."""
    lengths = {}
    for prompt, completion in conversations:
        prompt_ids = _template_ids(tokenizer, prompt, add_generation_prompt=True)
        full_ids = _template_ids(tokenizer, prompt + completion)
        common = 0
        while common < min(len(prompt_ids), len(full_ids)) and prompt_ids[common] == full_ids[common]:
            common += 1
        lengths[tuple(full_ids)] = common
    return lengths


def _findings(rows, tokenizer, intent):
    """Seen facts, unknowns, and intent checks. Returns (findings, violated, uncheckable)."""
    findings, violated, uncheckable = [], [], []
    n = len(rows)
    eos, bos, pad = tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id

    with_eos = [r for r in rows if r.final_eos_supervised is not None]
    if with_eos:
        eos_supervised = sum(r.final_eos_supervised for r in with_eos)
        findings.append(Finding(
            EvidenceClass.SEEN,
            f"The final EOS token (id {eos}) carries loss in {eos_supervised} of {len(with_eos)} rows",
            evidence={"rows_checked": len(with_eos), "eos_supervised": eos_supervised},
        ))
    if len(with_eos) < n:
        findings.append(Finding(EvidenceClass.UNKNOWN, f"{n - len(with_eos)} of {n} rows contain no real EOS token"))

    if bos is None:
        findings.append(Finding(EvidenceClass.SEEN, "The tokenizer defines no BOS token"))
    else:
        counts = [r.leading_bos for r in rows]
        findings.append(Finding(EvidenceClass.SEEN, f"Leading BOS tokens (id {bos}) per row: {counts}",
                                evidence={"leading_bos": counts}))

    known = [r for r in rows if r.prompt_tokens is not None]
    if known:
        total = sum(r.prompt_tokens for r in known)
        supervised = sum(r.supervised_prompt_tokens for r in known)
        findings.append(Finding(
            EvidenceClass.SEEN,
            f"{supervised} of {total} prompt tokens carry loss ({len(known)} rows)",
            evidence={"prompt_tokens": total, "supervised_prompt_tokens": supervised},
        ))
    if len(known) < n:
        findings.append(Finding(
            EvidenceClass.UNKNOWN,
            f"Prompt boundary unknown for {n - len(known)} of {n} rows "
            "(row does not match the chat template's tokenization)",
        ))

    real_pad = sum(r.pad_id_real_tokens for r in rows)
    masked_pad = sum(r.pad_id_real_tokens_masked for r in rows)
    findings.append(Finding(
        EvidenceClass.SEEN,
        f"{real_pad} real tokens share the pad ID {pad}; {masked_pad} of them carry no loss"
        if real_pad else f"No real token shares the pad ID {pad}",
        evidence={"pad_id_real_tokens": real_pad, "pad_id_real_tokens_masked": masked_pad},
    ))

    # Each intent key: (rows it can be checked on, the per-row fact it compares).
    checks = {
        "supervise_eos": (with_eos, lambda r: r.final_eos_supervised),
        "mask_prompt": (known, lambda r: r.supervised_prompt_tokens == 0),
        "single_bos": (rows, lambda r: (r.leading_bos or 0) <= 1),  # no BOS defined counts as 0
    }
    for key, expected in intent.items():
        checked, actual = checks[key]
        stated = f"{key}={str(expected).lower()}"
        if not checked:
            uncheckable.append(key)
            findings.append(Finding(EvidenceClass.UNKNOWN,
                                    f"Declared intent {stated} cannot be checked on these rows",
                                    evidence={"intent": key}))
            continue
        not_met = sum(actual(r) != expected for r in checked)
        if not_met:
            violated.append(key)
            findings.append(Finding(
                EvidenceClass.SEEN,
                f"Declared intent {stated} is not met in {not_met} of {len(checked)} rows",
                evidence={"intent": key, "rows_checked": len(checked), "rows_not_met": not_met},
            ))
            if key == "supervise_eos" and expected:
                findings.append(Finding(EvidenceClass.INFERRED,
                                        "The model may not learn to stop; confirm with P7",
                                        rule="unsupervised EOS implies no gradient on stopping"))
    if not intent:
        findings.append(Finding(EvidenceClass.UNKNOWN,
                                "No intent declared: P2 reports facts only and cannot pass or fail"))
    return findings, violated, uncheckable


def run(model_id: str, *, trainer: str = "transformers", intent: dict | None = None,
        seed: int = 0, data_format: str = "prompt_completion", samples=None, rows_path=None,
        trainer_args: dict | None = None, trainer_cls=None) -> ProbeResult:
    if trainer not in TRAINERS:
        raise ValueError(f"unknown trainer {trainer!r}; choose from {sorted(TRAINERS)}")
    if data_format not in DATA_FORMATS:
        raise ValueError(f"unknown data format {data_format!r}; choose from {DATA_FORMATS}")
    unknown = set(intent or {}) - set(INTENT_KEYS)
    if unknown:
        raise ValueError(f"unknown intent keys {sorted(unknown)}; choose from {INTENT_KEYS}")
    environment = capture_environment(device="cpu", dtype="float32", seeds={"proxy": seed, "trainer": seed})
    flags = "".join(f" --intent {k}={str(v).lower()}" for k, v in (intent or {}).items())
    reproducer = make_reproducer("p2_loss_tokens", model_id, trainer=trainer, data_format=data_format,
                                 seed=seed, rows_path=rows_path, trainer_args=trainer_args, extra=flags)
    limitations = (LIMITATIONS + ([] if samples else [BUILT_IN_ROWS_LIMITATION])
                   + ([TRANSFORMERS_RECIPE_LIMITATION] if trainer == "transformers" else []))
    common = dict(oracle=ORACLE, environment=environment, reproducer=reproducer,
                  limitations=limitations, declared_intent=intent)
    conversations = make_conversations(samples or SAMPLES)
    try:
        model, tokenizer, info = build_proxy(model_id, seed=seed)
        batch, dropped = first_batch(trainer, model, tokenizer, conversations, seed, data_format,
                                     trainer_args, trainer_cls)
        if dropped:
            limitations.append(f"this library version has no option(s) {dropped}; "
                               "they were left at the library default")
        lengths = _prompt_lengths(tokenizer, conversations)
        masks = batch.get("attention_mask")
        rows = []
        for i in range(batch["input_ids"].shape[0]):
            mask = masks[i] if masks is not None else None
            ids = batch["input_ids"][i]
            real_ids = tuple((ids[mask.bool()] if mask is not None else ids).tolist())
            rows.append(analyse_row(
                ids, batch["labels"][i], mask, prompt_tokens=lengths.get(real_ids),
                bos_id=tokenizer.bos_token_id, eos_id=tokenizer.eos_token_id,
                pad_id=tokenizer.pad_token_id))
    except Exception as exc:  # an API change must become not_tested, never a wrong verdict
        return ProbeResult(PROBE_ID, Status.NOT_TESTED, reason=f"{type(exc).__name__}: {exc}", **common)

    findings, violated, uncheckable = _findings(rows, tokenizer, intent or {})
    if violated:
        status = Status.FAIL
    elif not intent or uncheckable:
        status = Status.INFO
    else:
        status = Status.PASS
    observed = {
        "trainer": TRAINERS[trainer],
        "data_format": data_format,
        "unsupported_trainer_options": dropped,
        "proxy": info.to_dict(),
        "token_ids": {"bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id,
                      "pad": tokenizer.pad_token_id},
        "rows": [asdict(r) for r in rows],
    }
    return ProbeResult(PROBE_ID, status, findings=findings, observed=observed, **common)


def _parse_intent(items, parser) -> dict | None:
    intent = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if value.lower() not in ("true", "false"):
            parser.error(f"--intent expects key=true or key=false, got {item!r}")
        intent[key] = value.lower() == "true"
    return intent or None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P2: which tokens carry loss")
    add_common_arguments(parser)
    parser.add_argument("--intent", action="append", metavar="KEY=true|false",
                        help=f"declared intent; keys: {', '.join(INTENT_KEYS)}")
    args = parser.parse_args(argv)
    result = run(args.model_id, intent=_parse_intent(args.intent, parser), **common_kwargs(args))
    print(json.dumps(result.to_dict(), indent=2))
    return exit_code([result])


if __name__ == "__main__":
    sys.exit(main())
