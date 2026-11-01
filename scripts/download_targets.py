"""Download config + tokenizer files (never weights) for the default proxy targets."""
from conform.proxy_model import DEFAULT_TARGETS, download_target

for model_id in DEFAULT_TARGETS:
    path = download_target(model_id)
    print(f"{model_id} -> {path}")
