"""Record exactly what a probe ran against (spec design rule 3)."""
from __future__ import annotations

import platform
from importlib import metadata
from typing import Any

TRACKED_PACKAGES = ("torch", "transformers", "trl", "peft", "accelerate", "datasets")


def package_versions() -> dict[str, str | None]:
    """Installed version of each tracked package, or None if it is missing."""
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def capture_environment(*, device: str, dtype: str, seeds: dict[str, int]) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **package_versions(),
        "device": device,
        "dtype": dtype,
        "seeds": dict(seeds),
    }
