from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "pegasus-instances"
MANIFEST = ROOT / "manifests" / "workflow_sha256.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    failed = False
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = DATA_ROOT / relative
        if not path.is_file():
            print(f"MISSING  {relative}")
            failed = True
            continue
        actual = sha256(path)
        status = "OK" if actual == expected else "MISMATCH"
        print(f"{status:8} {relative}")
        failed = failed or actual != expected
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
