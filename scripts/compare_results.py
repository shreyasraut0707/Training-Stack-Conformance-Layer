"""Compare two results folders run by run by verdict (the G4 second-machine spot check).

A verdict is a run's status plus each finding's evidence class and verdict; numbers may
differ slightly between computers, verdicts may not. A folder may be a results folder or
one of its repeat-NN subfolders (repeat-01 is used when the folder holds repeats).

Usage: python scripts/compare_results.py PRIMARY_FOLDER SECOND_FOLDER
Exit code 0 when every run present in both folders has the same verdict and the two
folders contain the same runs; 1 otherwise.
"""
import json
import sys
from pathlib import Path

from conform.cli import verdict_signature


def load(folder: Path) -> dict:
    folder = Path(folder)
    if not list(folder.glob("*.json")) and (folder / "repeat-01").is_dir():
        folder = folder / "repeat-01"
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))}


def compare(primary: Path, second: Path) -> tuple[list[str], bool]:
    a, b = load(primary), load(second)
    lines, ok = [], bool(a) and a.keys() == b.keys()
    for name in sorted(a.keys() | b.keys()):
        if name not in a or name not in b:
            lines.append(f"MISSING {'in second' if name in a else 'in primary'}: {name}")
            continue
        same = verdict_signature(a[name]) == verdict_signature(b[name])
        ok &= same
        lines.append(f"{'same     ' if same else 'DIFFERENT'} {name}: {a[name]['status']} / {b[name]['status']}")
    lines.append(f"{len([n for n in a if n in b])} runs compared: "
                 + ("all verdicts identical" if ok else "NOT identical (see above)"))
    return lines, ok


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    lines, ok = compare(Path(argv[0]), Path(argv[1]))
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
