"""Which probes exist, what they need, and how a recipe becomes each probe's arguments."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

PROBES = {
    "P1": {"module": "p1_grad_accum", "name": "Gradient-accumulation equivalence", "release": "v0.1",
           "question": "Does accumulating k micro-batches match one batch of the same samples?",
           "uses_rows": False},
    "P1M": {"module": "p1_multiprocess", "name": "Multi-process loss averaging", "release": "v0.2",
            "question": "Is the loss averaged correctly across processes (CPU, gloo)?",
            "uses_rows": False},
    "P2": {"module": "p2_loss_tokens", "name": "Loss-bearing tokens", "release": "v0.1",
           "question": "Which tokens carry loss, and does that match the declared intent?",
           "uses_rows": True},
    "P3": {"module": "p3_template", "name": "Training versus inference tokens", "release": "v0.1",
           "question": "Are the tokens used in training the same tokens inference would see?",
           "uses_rows": True},
    "P4": {"module": "p4_packing", "name": "Packing isolation", "release": "v0.2",
           "question": "Does one packed document leak into another?", "uses_rows": True},
    "P5": {"module": "p5_resume", "name": "Resume equivalence", "release": "v0.2",
           "question": "Does a resumed run match an uninterrupted run?", "uses_rows": True},
    "P6": {"module": "p6_precision", "name": "Precision formats", "release": "v0.1",
           "question": "What precision does training actually use?", "uses_rows": True},
    "P7": {"module": "p7_eos_stop", "name": "EOS stopping", "release": "v0.1",
           "question": "After a short fine-tune, does the saved and reloaded model learn to emit EOS and stop?",
           "uses_rows": True},
}

# Symptom -> the probes most related to it, most direct first (spec Sections 5 and 9).
SYMPTOMS = {
    "no-stop": ["P7", "P2", "P3"],
    "worse-than-base": ["P1", "P2", "P3", "P6"],
}

RECIPE_KEYS = {"model", "trainer", "data_format", "seed", "intent", "rows", "rows_file",
               "trainer_args", "probes", "timeout"}


class RecipeError(ValueError):
    """A recipe the CLI cannot use (exit code 2)."""


# Libraries each probe needs from the installed stack (spec Section 9: probes available for
# the installed stack, and why any are unavailable). Checked without importing them.
STACK = ("torch", "transformers", "accelerate")
EXTRA_REQUIREMENTS = {"P3": ("trl", "datasets"), "P4": ("trl", "datasets")}


def _installed(package: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(package) is not None


def missing(probe_id: str) -> list[str]:
    """Libraries this probe needs that are not installed."""
    return [p for p in STACK + EXTRA_REQUIREMENTS.get(probe_id, ()) if not _installed(p)]


def unavailable_reason(probe_id: str) -> str | None:
    info = PROBES[probe_id]
    if info["module"] is None:
        return info["unavailable"]
    gone = missing(probe_id)
    return f"the installed stack lacks {', '.join(gone)}" if gone else None


def available(probe_id: str) -> bool:
    return unavailable_reason(probe_id) is None


def load_recipe(path) -> dict:
    """Read and validate a recipe file (YAML or JSON)."""
    path = Path(path)
    try:
        recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RecipeError(f"cannot read recipe {path}: {exc}") from exc
    if not isinstance(recipe, dict) or "model" not in recipe:
        raise RecipeError("a recipe needs at least a 'model' (a Hugging Face model or tokenizer id)")
    unknown = set(recipe) - RECIPE_KEYS
    if unknown:
        raise RecipeError(f"unknown recipe keys {sorted(unknown)}; allowed: {sorted(RECIPE_KEYS)}")
    if recipe.get("trainer", "trl") not in ("trl", "transformers"):
        raise RecipeError("trainer must be 'trl' or 'transformers'")
    for pid in recipe.get("probes", []):
        if pid not in PROBES:
            raise RecipeError(f"unknown probe {pid!r}; choose from {sorted(PROBES)}")
    if "rows" in recipe and "rows_file" in recipe:
        raise RecipeError("give either 'rows' or 'rows_file', not both")
    if "rows_file" in recipe:
        recipe["rows_file"] = str((path.parent / recipe["rows_file"]).resolve())
    return recipe


def probe_args(probe_id: str, recipe: dict, rows_path: str | None) -> list[str]:
    """Command-line arguments for one probe, built from the recipe."""
    trainer = recipe.get("trainer", "trl")
    args = ["--trainer", trainer, "--seed", str(recipe.get("seed", 0))]
    if PROBES[probe_id]["uses_rows"]:
        args += ["--data-format", recipe.get("data_format", "prompt_completion")]
        if rows_path:
            args += ["--rows", rows_path]
        if recipe.get("trainer_args"):
            args += ["--trainer-args", json.dumps(recipe["trainer_args"], sort_keys=True)]
    if probe_id == "P2":
        for key, value in (recipe.get("intent") or {}).items():
            args += ["--intent", f"{key}={str(bool(value)).lower()}"]
    return args


def selected_probes(recipe: dict, requested=None) -> list[str]:
    ids = list(requested or recipe.get("probes") or [p for p in PROBES if available(p)])
    for pid in ids:
        if pid not in PROBES:
            raise RecipeError(f"unknown probe {pid!r}; choose from {sorted(PROBES)}")
    return ids
