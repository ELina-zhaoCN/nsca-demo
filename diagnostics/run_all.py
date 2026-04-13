#!/usr/bin/env python3
"""
Run all four pre-training diagnostics (SPEC Issue #5).

Usage (from repository root):
    python diagnostics/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TESTS = [
    ("Noisy TV", "scripts/noisy_tv_test.py"),
    ("Forgetting", "scripts/forgetting_test.py"),
    ("Balloon", "scripts/balloon_test.py"),
    ("Slot Discovery", "scripts/slot_discovery_test.py"),
]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    failed: list[str] = []
    for name, rel in TESTS:
        script = root / rel
        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)
        r = subprocess.run([sys.executable, str(script)], cwd=str(root))
        if r.returncode != 0:
            failed.append(name)
    print("\n" + "=" * 60)
    if failed:
        print("Failed:", ", ".join(failed))
        return 1
    print("All diagnostics finished with exit code 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
