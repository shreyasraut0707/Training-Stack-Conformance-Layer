"""Create the regression bank's pinned environments from bank/envs/<name>.txt.

Creates only the environments that bank cases use, each with the interpreter running this
script (Python 3.11), installs its exact package list, and checks the installed versions
against the list. Existing environments are left untouched unless --recreate is given.

Usage: python scripts/setup_bank_envs.py [--venv-root DIR] [--recreate] [--only NAME ...]
"""
import argparse
import subprocess
import sys
from pathlib import Path

from conform.bank import DEFAULT_VENV_ROOT, PROJECT_ROOT, load_cases, venv_python


def used_envs() -> list[str]:
    names = {env["venv"] for case in load_cases() for env in case["environments"].values()}
    return sorted(names)


def freeze(python: Path) -> list[str]:
    out = subprocess.run([str(python), "-m", "pip", "freeze", "--disable-pip-version-check"],
                         capture_output=True, text=True, check=True).stdout
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create the bank's pinned environments")
    parser.add_argument("--venv-root", type=Path, default=DEFAULT_VENV_ROOT)
    parser.add_argument("--recreate", action="store_true", help="delete and recreate existing environments")
    parser.add_argument("--only", nargs="+", help="environment names (default: all used by bank cases)")
    args = parser.parse_args(argv)
    problems = 0
    for name in args.only or used_envs():
        wanted = PROJECT_ROOT / "bank" / "envs" / f"{name}.txt"
        if not wanted.exists():
            print(f"{name}: no package list at {wanted}")
            problems += 1
            continue
        python = venv_python(args.venv_root, name)
        if python.exists() and not args.recreate:
            print(f"{name}: exists, checking only")
        else:
            args.venv_root.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, "-m", "venv", "--clear", str(args.venv_root / name)], check=True)
            subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"], check=True)
            subprocess.run([str(python), "-m", "pip", "install", "--quiet", "-r", str(wanted)], check=True)
        expected = sorted(line.strip() for line in wanted.read_text(encoding="utf-8").splitlines() if line.strip())
        got = freeze(python)
        if got == expected:
            print(f"{name}: identical to {wanted.name}")
        else:
            problems += 1
            print(f"{name}: DIFFERENT from {wanted.name}: missing {sorted(set(expected) - set(got))}, "
                  f"extra {sorted(set(got) - set(expected))}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
