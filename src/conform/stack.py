"""Build the real trainer under test from a few sample conversations (shared by P2, P3, P6, P7).

Only public APIs are used: the trainer is constructed as a user would, and probes read
its own dataloader, model and optimizer. Trainer settings from a recipe (`trainer_args`)
override the probe defaults; options this library version does not have are dropped and
reported, never silently passed.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

TRAINERS = {
    "transformers": "Transformers Trainer + DataCollatorForLanguageModeling(mlm=False)",
    "trl": "TRL SFTTrainer",
}
DATA_FORMATS = ("prompt_completion", "messages")

# Prompt-completion conversations of different lengths, so the batch is padded.
SAMPLES = (
    ("What is 2+2?", "It is 4."),
    ("Name a primary colour.", "Red is a primary colour."),
    ("Say hi.", "Hi."),
    ("Spell cat backwards.", "The word cat backwards is tac."),
)

TRANSFORMERS_RECIPE_LIMITATION = (
    "Transformers path uses the common recipe: chat-template tokenization plus "
    "DataCollatorForLanguageModeling(mlm=False); a custom collator may behave differently"
)


def load_samples(path) -> tuple[tuple[str, str], ...]:
    """Read user rows from a JSON list of {"prompt": ..., "completion": ...} objects."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not rows or not all(isinstance(r, dict) and {"prompt", "completion"} <= set(r) for r in rows):
        raise ValueError("rows must be a non-empty JSON list of {'prompt': ..., 'completion': ...}")
    return tuple((str(r["prompt"]), str(r["completion"])) for r in rows)


def conversations(samples=SAMPLES):
    """(prompt messages, completion messages) for each (prompt, completion) pair."""
    return [([{"role": "user", "content": p}], [{"role": "assistant", "content": c}]) for p, c in samples]


def template_ids(tokenizer, messages, **kwargs) -> list[int]:
    out = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    return list(out["input_ids"] if hasattr(out, "keys") else out)


def supported(config_cls, options: dict) -> tuple[dict, list[str]]:
    """Split options into those this version's config accepts and those it lacks."""
    fields = {f.name for f in dataclasses.fields(config_cls)}
    return ({k: v for k, v in options.items() if k in fields},
            sorted(k for k in options if k not in fields))


def build_trainer(trainer, model, tokenizer, convs, *, output_dir, seed,
                  data_format="prompt_completion", trainer_args=None, batch_size=None,
                  force_fp32=True, optimizers=(None, None), trainer_cls=None):
    """Return (trainer instance, dropped option names).

    `force_fp32` sets bf16 and fp16 off unless the recipe sets them; P6 passes False so the
    library's own precision defaults apply. `trainer_cls` lets tests inject a defect.
    """
    base = dict(output_dir=str(output_dir), per_device_train_batch_size=batch_size or len(convs),
                use_cpu=True, seed=seed, dataloader_num_workers=0, report_to="none",
                disable_tqdm=True, save_strategy="no")
    if force_fp32:
        base.update(bf16=False, fp16=False)
    if trainer == "transformers":
        kept, dropped = supported(TrainingArguments, {**base, "remove_unused_columns": False,
                                                      **(trainer_args or {})})
        rows = [{"input_ids": template_ids(tokenizer, p + c)} for p, c in convs]
        built = (trainer_cls or Trainer)(
            model=model, args=TrainingArguments(**kept), train_dataset=rows,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            optimizers=optimizers)
        return built, dropped

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    if data_format == "messages":
        records = [{"messages": p + c} for p, c in convs]
    else:
        records = [{"prompt": p, "completion": c} for p, c in convs]
    # Packing is P4's job, so it is off unless the recipe turns it on.
    kept, dropped = supported(SFTConfig, {**base, "packing": False, "padding_free": False,
                                          **(trainer_args or {})})
    built = (trainer_cls or SFTTrainer)(
        model=model, args=SFTConfig(**kept), train_dataset=Dataset.from_list(records),
        processing_class=tokenizer, optimizers=optimizers)
    return built, dropped


def first_batch(trainer, model, tokenizer, convs, seed, data_format="prompt_completion",
                trainer_args=None, trainer_cls=None):
    """The first batch from the trainer's own dataloader (public API), plus any trainer
    options this library version does not support (they were left out)."""
    import tempfile

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as out:
        built, dropped = build_trainer(trainer, model, tokenizer, convs, output_dir=out, seed=seed,
                                       data_format=data_format, trainer_args=trainer_args,
                                       trainer_cls=trainer_cls)
        return next(iter(built.get_train_dataloader())), dropped


def attention_kernel(model) -> tuple[str | None, bool]:
    """Adapter (spec Section 6): the attention implementation the model runs, and whether it
    had to be read from a private attribute (then reports mark it fragile)."""
    public = getattr(model.config, "attn_implementation", None)
    if isinstance(public, str):
        return public, False
    return getattr(model.config, "_attn_implementation", None), True


def real_ids(batch, i) -> list[int]:
    """Row i of a batch without padding (attended tokens only)."""
    ids = batch["input_ids"][i]
    mask = batch.get("attention_mask")
    return (ids[mask[i].bool()] if mask is not None else ids).tolist()


def add_common_arguments(parser) -> None:
    """Command-line options shared by the recipe-driven probes (P2, P3, P6, P7)."""
    parser.add_argument("model_id")
    parser.add_argument("--trainer", choices=sorted(TRAINERS), default="transformers")
    parser.add_argument("--data-format", choices=DATA_FORMATS, default="prompt_completion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", help="JSON file with a list of {prompt, completion} rows")
    parser.add_argument("--trainer-args", help="JSON object of trainer settings from the recipe")


def common_kwargs(args) -> dict:
    return dict(trainer=args.trainer, data_format=args.data_format, seed=args.seed,
                samples=load_samples(args.rows) if args.rows else None, rows_path=args.rows,
                trainer_args=json.loads(args.trainer_args) if args.trainer_args else None)


def reproducer(module, model_id, *, trainer, data_format, seed, rows_path=None,
               trainer_args=None, extra="") -> str:
    command = (f"python -m conform.probes.{module} {model_id} --trainer {trainer} "
               f"--data-format {data_format} --seed {seed}")
    if rows_path:
        command += f" --rows {rows_path}"
    if trainer_args:
        command += f" --trainer-args '{json.dumps(trainer_args, sort_keys=True)}'"
    return command + extra
