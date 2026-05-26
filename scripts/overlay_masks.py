#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overlay/fill masks on images using natural-order matching.

Default behavior keeps the original visualization style:
    mask region -> transparent green overlay with alpha=0.3

If you need images for previous --image_url experiments where masked regions
are covered by opaque white, explicitly pass --mode white.

Examples:
    # Default: transparent green visualization, alpha=0.3
    python overlay_masks_white.py \
        --imgs ./images \
        --masks ./masks \
        -o ./vis_green

    # Opaque white images for --image_url inference input:
    python overlay_masks_white.py \
        --imgs ./images \
        --masks ./masks \
        -o ./masked_white_imgs \
        --mode white \
        --save_list ./masked_white_imgs.txt
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def natural_key(path: Path):
    """
    Natural order:
        frame_2.png < frame_10.png
    """
    name = path.as_posix()
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", name)
    ]


def list_images(folder: Path):
    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    return sorted(files, key=natural_key)


def load_mask(mask_path: Path, target_hw, threshold=127):
    """
    Read mask and convert it to bool mask.
    Pixels larger than threshold are treated as occluded regions.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    if mask.ndim == 3:
        # RGB/RGBA mask -> grayscale.
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    h, w = target_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask > threshold


def fill_mask(img, mask_bool, fill_value=255):
    """
    Fill mask region with an opaque constant value.
    For white opaque mask, fill_value=255.
    """
    out = img.copy()
    fill_value = int(np.clip(fill_value, 0, 255))
    out[mask_bool] = (fill_value, fill_value, fill_value)
    return out


def overlay_green(img, mask_bool, alpha=0.3):
    """
    Overlay transparent green on mask region for visualization.
    OpenCV uses BGR, so green is (0, 255, 0).
    """
    out = img.copy()

    green = np.zeros_like(img, dtype=np.uint8)
    green[:, :] = (0, 255, 0)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = cv2.addWeighted(img, 1.0 - alpha, green, alpha, 0)

    out[mask_bool] = blended[mask_bool]
    return out


def make_output_path(img_path: Path, img_root: Path, out_root: Path):
    """
    Preserve the relative directory structure to avoid overwriting files
    with the same basename from different subfolders.
    """
    rel = img_path.relative_to(img_root)
    out_path = out_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def write_list(list_path: Path, paths):
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p.resolve()) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply masks to images using natural-order matching. "
            "Default mode overlays transparent green mask visualization with alpha=0.3."
        )
    )
    parser.add_argument(
        "--imgs",
        required=True,
        type=str,
        help="Input image directory."
    )
    parser.add_argument(
        "--masks",
        required=True,
        type=str,
        help="Input mask directory."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=str,
        help="Output directory."
    )
    parser.add_argument(
        "--mode",
        choices=["white", "fill", "green"],
        default="green",
        help=(
            "green: transparent green visualization; "
            "white/fill: fill mask region with an opaque constant value. Default: green"
        )
    )
    parser.add_argument(
        "--fill_value",
        type=int,
        default=255,
        help="Fill value for --mode white/fill. 255 means pure white. Default: 255"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Mask overlay transparency for --mode green. Default: 0.3"
    )
    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=127,
        help="Mask binarization threshold. Pixels > threshold are treated as mask. Default: 127"
    )
    parser.add_argument(
        "--repeat_masks",
        action="store_true",
        help=(
            "If mask count is smaller than image count, reuse masks cyclically: "
            "image[i] <-> mask[i % len(masks)]."
        )
    )
    parser.add_argument(
        "--save_list",
        type=str,
        default="",
        help=(
            "Optional path to save an absolute-path txt list of generated images. "
            "This list can be used directly as --image_url."
        )
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Compatibility argument. The script always searches recursively."
    )

    args = parser.parse_args()

    img_dir = Path(args.imgs)
    mask_dir = Path(args.masks)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    img_paths = list_images(img_dir)
    mask_paths = list_images(mask_dir)

    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in: {img_dir}")
    if len(mask_paths) == 0:
        raise RuntimeError(f"No masks found in: {mask_dir}")

    if args.repeat_masks:
        n = len(img_paths)
        print(
            f"[INFO] repeat_masks=True. Processing all {n} images with "
            f"{len(mask_paths)} masks cyclically."
        )
    else:
        n = min(len(img_paths), len(mask_paths))
        if len(img_paths) != len(mask_paths):
            print(
                f"[WARN] Image count != mask count: "
                f"{len(img_paths)} images vs {len(mask_paths)} masks. "
                f"Only processing first {n} pairs. "
                f"Use --repeat_masks to process all images cyclically."
            )

    print(f"[INFO] Found {len(img_paths)} images.")
    print(f"[INFO] Found {len(mask_paths)} masks.")
    print(f"[INFO] Processing {n} pairs.")
    print(f"[INFO] Mode = {args.mode}")

    if args.mode in ("white", "fill"):
        print(f"[INFO] Fill value = {args.fill_value}")
    else:
        print(f"[INFO] Alpha = {args.alpha}")

    written_paths = []

    for i in range(n):
        img_path = img_paths[i]
        mask_path = mask_paths[i % len(mask_paths)] if args.repeat_masks else mask_paths[i]

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read image, skip: {img_path}")
            continue

        h, w = img.shape[:2]
        mask_bool = load_mask(mask_path, target_hw=(h, w), threshold=args.mask_threshold)

        if args.mode in ("white", "fill"):
            out = fill_mask(img, mask_bool, fill_value=args.fill_value)
        else:
            out = overlay_green(img, mask_bool, alpha=args.alpha)

        out_path = make_output_path(img_path, img_dir, out_dir)
        ok = cv2.imwrite(str(out_path), out)
        if not ok:
            print(f"[WARN] Failed to write output: {out_path}")
            continue

        written_paths.append(out_path)

        print(
            f"[{i:05d}] "
            f"img={img_path.name}  "
            f"mask={mask_path.name}  "
            f"-> {out_path.relative_to(out_dir)}"
        )

    if args.save_list:
        write_list(Path(args.save_list), written_paths)
        print(f"[INFO] Saved generated image list: {Path(args.save_list).resolve()}")

    print(f"[DONE] Wrote {len(written_paths)} images to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
