"""Write a hash-pinned lockfile from a pinned requirements list (spec Section 12.2).

Runs `pip install --dry-run` (nothing is installed) and reads each chosen file's SHA-256
from pip's report, so the lockfile pins exact files, not only versions. The result is
specific to the platform and Python version it was made on (here: Windows, CPython 3.11).
Install from it with `pip install --require-hashes -r <lockfile>`.

Usage: python scripts/make_lockfile.py bank/envs/conform.txt [--extra-index-url URL]
"""
import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


def _sha256_of(url: str) -> str:
    import hashlib
    import urllib.request

    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Make a hash-pinned lockfile")
    parser.add_argument("requirements", type=Path)
    parser.add_argument("--extra-index-url", default="https://download.pytorch.org/whl/cpu")
    parser.add_argument("--out", type=Path, help="default: the requirements file with .lock")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as work:
        report = Path(work) / "report.json"
        subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--quiet",
                        "--disable-pip-version-check", "--report", str(report), "-r", str(args.requirements),
                        "--extra-index-url", args.extra_index_url], check=True)
        items = json.loads(report.read_text(encoding="utf-8"))["install"]
    lines = [f"# Hash-pinned lockfile generated from {args.requirements.as_posix()} by scripts/make_lockfile.py",
             f"# Platform: {platform.system()} {platform.machine()}, CPython {platform.python_version()}",
             f"# Install: pip install --require-hashes -r <this file> --extra-index-url {args.extra_index_url}"]
    missing = []
    for item in sorted(items, key=lambda i: i["metadata"]["name"].lower()):
        name, version = item["metadata"]["name"], item["metadata"]["version"]
        digest = item["download_info"].get("archive_info", {}).get("hashes", {}).get("sha256")
        if digest is None:  # some indexes publish no hash: hash the exact file pip chose
            try:
                digest = _sha256_of(item["download_info"]["url"])
            except OSError:
                missing.append(name)
                continue
        lines.append(f"{name}=={version} \\\n    --hash=sha256:{digest}")
    out = args.out or args.requirements.with_suffix(".lock")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(items) - len(missing)} packages)")
    if missing:
        print(f"no sha256 for: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
