#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keep the first N files in a directory by natural numeric order, and delete all files after N.

Example:
    python keep_first_n_files.py -i ./images -n 200
    python keep_first_n_files.py -i ./images -n 200 --dry-run

Notes:
    - Only files directly under the input directory are processed by default.
    - Subdirectories are ignored unless --recursive is enabled.
    - Natural sort means frame_2.png comes before frame_10.png.
"""

import argparse
import re
from pathlib import Path
from typing import List, Union


def natural_key(path: Path) -> List[Union[int, str]]:
    """Return a key for natural sorting by filename."""
    name = path.name
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def collect_files(input_dir: Path, recursive: bool = False) -> List[Path]:
    """Collect files from input_dir."""
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file()]
    return sorted(files, key=natural_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep the first N files by natural numeric order and delete the rest."
    )
    parser.add_argument("-i", "--input", required=True, help="Input directory to process.")
    parser.add_argument("-n", "--num", type=int, required=True, help="Number of files to keep.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print files that would be deleted, without actually deleting them.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process files recursively. Default: only process files directly in the input directory.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if args.num < 0:
        raise ValueError("--num / -n must be >= 0")

    files = collect_files(input_dir, recursive=args.recursive)
    keep_files = files[:args.num]
    delete_files = files[args.num:]

    print(f"[INFO] Input directory : {input_dir}")
    print(f"[INFO] Total files     : {len(files)}")
    print(f"[INFO] Keep first N    : {args.num}")
    print(f"[INFO] Will keep       : {len(keep_files)}")
    print(f"[INFO] Will delete     : {len(delete_files)}")

    if delete_files:
        print("\n[INFO] Files to delete:")
        for p in delete_files:
            print(p)

    if args.dry_run:
        print("\n[DRY-RUN] No files were deleted.")
        return

    deleted = 0
    failed = 0
    for p in delete_files:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            failed += 1
            print(f"[WARN] Failed to delete {p}: {e}")

    print(f"\n[DONE] Deleted: {deleted}, Failed: {failed}")


if __name__ == "__main__":
    main()
