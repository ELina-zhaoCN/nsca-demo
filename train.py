#!/usr/bin/env python3
"""
NSCA training launcher (Layer 0 world model via `scripts/train_world_model.py`).

Usage:
    python train.py
    python train.py --config configs/default.yaml
    python train.py --config configs/training_config_local.yaml --phase vision
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def _maybe_resolve_wrapper_config(argv: list[str], root: Path) -> list[str]:
    """If --config points to a wrapper YAML with `training_config`, use that file."""
    if yaml is None or "--config" not in argv:
        return argv
    i = argv.index("--config")
    if i + 1 >= len(argv):
        return argv
    selected = Path(argv[i + 1])
    if not selected.is_absolute():
        selected = root / selected
    if not selected.exists():
        return argv
    try:
        with open(selected, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except OSError:
        return argv
    inner = data.get("training_config")
    if not inner:
        return argv
    inner_path = Path(inner)
    if not inner_path.is_absolute():
        inner_path = root / inner_path
    if not inner_path.exists():
        return argv
    new_argv = argv.copy()
    new_argv[i + 1] = str(inner_path)
    return new_argv


def main() -> int:
    root = Path(__file__).resolve().parent
    script = root / "scripts" / "train_world_model.py"
    local_config = root / "configs" / "training_config_local.yaml"
    args = list(sys.argv[1:])
    args = _maybe_resolve_wrapper_config(args, root)
    if "--config" not in args:
        args = ["--config", str(local_config)] + args
    cmd = [sys.executable, str(script)] + args
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
