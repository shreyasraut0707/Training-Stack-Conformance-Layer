"""Proxy model factory (spec Section 7.2).

Builds a tiny, randomly initialised copy of a target architecture that keeps the
real tokenizer and vocabulary size, so token IDs stay valid. It tests how the
trainer behaves, not what a real model knows. Pretrained weights are never downloaded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Ungated targets: no Hugging Face account or token needed.
QWEN_TYPE = "Qwen/Qwen2.5-0.5B-Instruct"
LLAMA_TYPE = "HuggingFaceTB/SmolLM2-135M-Instruct"
# Llama-2 SentencePiece tokenizer family: the third family for the G3 controls.
TINYLLAMA_TYPE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_TARGETS = (QWEN_TYPE, LLAMA_TYPE, TINYLLAMA_TYPE)

SUPPORTED_MODEL_TYPES = ("llama", "qwen2")

# About two layers and a small hidden size (spec design choice).
PROXY_SHAPE = {
    "num_hidden_layers": 2,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
}

# Dropped from the real config so the config class recomputes them for the proxy
# shape (layer_types and head_dim depend on layer count and hidden size), and so
# the original bf16 dtype does not leak into the proxy.
_DROP_KEYS = (
    "layer_types", "head_dim", "model_type", "architectures",
    "transformers_version", "dtype", "torch_dtype",
)

# Only these files are fetched: never model weights.
TOKENIZER_AND_CONFIG_FILES = [
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json",
]


@dataclass
class ProxyInfo:
    model_id: str
    model_type: str
    revision: str | None  # Hub commit the config was read from, when known
    vocab_size: int
    tokenizer_len: int
    num_hidden_layers: int
    hidden_size: int
    seed: int
    pad_token_id: int | None
    eos_token_id: int | None
    bos_token_id: int | None
    pad_equals_eos: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def download_target(model_id: str) -> str:
    """Fetch only the config and tokenizer files into the Hugging Face cache."""
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, allow_patterns=TOKENIZER_AND_CONFIG_FILES)


def _cached_revision(model_id: str) -> str | None:
    """Hub commit of the cached config.json (its snapshot folder name), if known."""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(model_id, "config.json")
    return Path(path).parent.name if isinstance(path, str) else None


def build_proxy(
    model_id: str,
    *,
    seed: int = 0,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
):
    """Return (model, tokenizer, ProxyInfo) for a tiny random copy of `model_id`."""
    config = AutoConfig.from_pretrained(
        model_id, local_files_only=local_files_only, trust_remote_code=trust_remote_code
    )
    if config.model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"unsupported model_type {config.model_type!r}; supported: {SUPPORTED_MODEL_TYPES}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=local_files_only, trust_remote_code=trust_remote_code
    )

    settings = config.to_dict()
    for key in _DROP_KEYS:
        settings.pop(key, None)
    settings.update(PROXY_SHAPE)
    if "max_window_layers" in settings:  # Qwen2: keep every proxy layer full attention
        settings["max_window_layers"] = PROXY_SHAPE["num_hidden_layers"]
    proxy_config = AutoConfig.for_model(config.model_type, **settings)

    if len(tokenizer) > proxy_config.vocab_size:
        raise ValueError(
            f"tokenizer has {len(tokenizer)} tokens but vocab_size is {proxy_config.vocab_size}"
        )

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(proxy_config, attn_implementation="eager")
    model = model.to(torch.float32)

    info = ProxyInfo(
        model_id=model_id,
        model_type=config.model_type,
        revision=_cached_revision(model_id),
        vocab_size=proxy_config.vocab_size,
        tokenizer_len=len(tokenizer),
        num_hidden_layers=proxy_config.num_hidden_layers,
        hidden_size=proxy_config.hidden_size,
        seed=seed,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_equals_eos=(
            tokenizer.pad_token_id is not None
            and tokenizer.pad_token_id == tokenizer.eos_token_id
        ),
    )
    return model, tokenizer, info
