#!/usr/bin/env python3
"""
Download NSCA checkpoints from Google Drive using `gdown`.

1. Share your Drive file or folder (anyone with link can view).
2. For a folder: `python scripts/download_checkpoints.py --folder-id <ID>`
3. For a single file: `python scripts/download_checkpoints.py --file-id <ID>`

Install: `pip install gdown`
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download checkpoints via gdown")
    parser.add_argument("--folder-id", type=str, default=None, help="Google Drive folder ID")
    parser.add_argument("--file-id", type=str, default=None, help="Google Drive file ID")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to write files into",
    )
    args = parser.parse_args()
    if not args.folder_id and not args.file_id:
        parser.error("Provide --folder-id or --file-id")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        print("Install gdown: pip install gdown", file=sys.stderr)
        return 1
    if args.folder_id:
        url = f"https://drive.google.com/drive/folders/{args.folder_id}"
        gdown.download_folder(url, output=str(out), quiet=False, use_cookies=False)
    else:
        url = f"https://drive.google.com/uc?id={args.file_id}"
        gdown.download(url, str(out / "checkpoint.bin"), quiet=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
