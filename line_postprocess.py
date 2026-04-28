#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from collections import deque
from typing import List, Tuple

import numpy as np
from PIL import Image

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对线框二值图进行后处理：小连通域去除、可选闭运算、骨架细化（Zhang-Suen thinning）"
    )
    parser.add_argument("--input_path", type=str, required=True, help="输入线框图所在目录")
    parser.add_argument("--output_path", type=str, required=True, help="输出处理后线框图所在目录")
    parser.add_argument("--threshold", type=int, default=127, help="二值化阈值，默认 127")
    parser.add_argument(
        "--min_component_size",
        type=int,
        default=8,
        help="删除小连通域的最小像素数阈值，小于该值的连通域会被移除，默认 8",
    )
    parser.add_argument(
        "--closing_kernel",
        type=int,
        default=0,
        help="闭运算核大小（奇数更合适）。0 表示不做闭运算，推荐 0 或 3，默认 0",
    )
    parser.add_argument(
        "--no_skeletonize",
        action="store_true",
        help="不做骨架细化。默认会做 skeletonize，将线细化为接近 1 像素宽",
    )
    parser.add_argument(
        "--overwrite_ext_png",
        action="store_true",
        help="输出时统一保存为 png；默认保持原扩展名",
    )
    return parser.parse_args()


# -----------------------------
# File utils
# -----------------------------
def find_images(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(SUPPORTED_EXTS):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Image IO
# -----------------------------
def load_binary_mask(img_path: str, threshold: int = 127) -> np.ndarray:
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    mask = (arr > threshold).astype(np.uint8)
    return mask


def save_binary_mask(mask: np.ndarray, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    img.save(out_path)


# -----------------------------
# Morphology (pure numpy)
# -----------------------------
def _max_filter2d(mask: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    padded = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    h, w = mask.shape
    out = np.zeros_like(mask)
    for dy in range(k):
        for dx in range(k):
            out = np.maximum(out, padded[dy:dy + h, dx:dx + w])
    return out


def _min_filter2d(mask: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    padded = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    h, w = mask.shape
    out = np.ones_like(mask)
    for dy in range(k):
        for dx in range(k):
            out = np.minimum(out, padded[dy:dy + h, dx:dx + w])
    return out


def morphological_closing(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1
    dil = _max_filter2d(mask, kernel_size)
    ero = _min_filter2d(dil, kernel_size)
    return ero.astype(np.uint8)


# -----------------------------
# Connected components (8-neighborhood)
# -----------------------------
def remove_small_components(mask: np.ndarray, min_size: int = 8) -> np.ndarray:
    if min_size <= 1:
        return mask.copy()

    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    out = mask.copy()
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or visited[y, x] == 1:
                continue

            q = deque([(y, x)])
            visited[y, x] = 1
            coords = [(y, x)]

            while q:
                cy, cx = q.popleft()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 1 and visited[ny, nx] == 0:
                        visited[ny, nx] = 1
                        q.append((ny, nx))
                        coords.append((ny, nx))

            if len(coords) < min_size:
                for cy, cx in coords:
                    out[cy, cx] = 0

    return out.astype(np.uint8)


# -----------------------------
# Skeletonization (Zhang-Suen thinning)
# -----------------------------
def _neighbors(img: np.ndarray, y: int, x: int) -> Tuple[int, int, int, int, int, int, int, int]:
    # P2,P3,...,P9 in Zhang-Suen notation
    return (
        img[y - 1, x],     # P2
        img[y - 1, x + 1], # P3
        img[y, x + 1],     # P4
        img[y + 1, x + 1], # P5
        img[y + 1, x],     # P6
        img[y + 1, x - 1], # P7
        img[y, x - 1],     # P8
        img[y - 1, x - 1], # P9
    )


def _transitions(neis: Tuple[int, ...]) -> int:
    seq = list(neis) + [neis[0]]
    return sum((seq[i] == 0 and seq[i + 1] == 1) for i in range(8))


def zhang_suen_thinning(mask: np.ndarray) -> np.ndarray:
    img = mask.copy().astype(np.uint8)
    if img.ndim != 2:
        raise ValueError("Input mask must be 2D.")

    changed = True
    h, w = img.shape
    while changed:
        changed = False
        to_remove = []

        # Step 1
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] != 1:
                    continue
                neis = _neighbors(img, y, x)
                B = sum(neis)
                A = _transitions(neis)
                p2, p3, p4, p5, p6, p7, p8, p9 = neis
                if (2 <= B <= 6 and A == 1 and p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0):
                    to_remove.append((y, x))
        if to_remove:
            changed = True
            for y, x in to_remove:
                img[y, x] = 0

        to_remove = []
        # Step 2
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] != 1:
                    continue
                neis = _neighbors(img, y, x)
                B = sum(neis)
                A = _transitions(neis)
                p2, p3, p4, p5, p6, p7, p8, p9 = neis
                if (2 <= B <= 6 and A == 1 and p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0):
                    to_remove.append((y, x))
        if to_remove:
            changed = True
            for y, x in to_remove:
                img[y, x] = 0

    return img.astype(np.uint8)


# -----------------------------
# Pipeline
# -----------------------------
def postprocess_mask(
    mask: np.ndarray,
    min_component_size: int = 8,
    closing_kernel: int = 0,
    do_skeletonize: bool = True,
) -> np.ndarray:
    out = mask.copy().astype(np.uint8)

    if min_component_size > 1:
        out = remove_small_components(out, min_size=min_component_size)

    if closing_kernel > 1:
        out = morphological_closing(out, kernel_size=closing_kernel)
        # 做完闭运算后，再清一次小碎点，避免闭运算带来轻微粘连噪声
        if min_component_size > 1:
            out = remove_small_components(out, min_size=min_component_size)

    if do_skeletonize:
        out = zhang_suen_thinning(out)

    return out.astype(np.uint8)


def main() -> int:
    args = parse_args()

    if not os.path.isdir(args.input_path):
        print(f"[ERROR] --input_path is not a directory: {args.input_path}")
        return 1

    files = find_images(args.input_path)
    if not files:
        print(f"[ERROR] No images found in: {args.input_path}")
        return 2

    ensure_dir(args.output_path)

    print(f"[INFO] Found images: {len(files)}")
    print("[INFO] Settings:")
    print(f"  threshold          = {args.threshold}")
    print(f"  min_component_size = {args.min_component_size}")
    print(f"  closing_kernel     = {args.closing_kernel}")
    print(f"  skeletonize        = {not args.no_skeletonize}")
    print(f"  output_as_png      = {args.overwrite_ext_png}")
    print("=" * 100)

    for idx, in_path in enumerate(files, start=1):
        rel = os.path.relpath(in_path, args.input_path)
        stem, ext = os.path.splitext(rel)
        out_rel = stem + (".png" if args.overwrite_ext_png else ext)
        out_path = os.path.join(args.output_path, out_rel)

        mask = load_binary_mask(in_path, threshold=args.threshold)
        out = postprocess_mask(
            mask,
            min_component_size=args.min_component_size,
            closing_kernel=args.closing_kernel,
            do_skeletonize=not args.no_skeletonize,
        )
        save_binary_mask(out, out_path)

        print(f"[{idx:04d}/{len(files):04d}] {in_path} -> {out_path}")

    print("=" * 100)
    print(f"[DONE] Output saved to: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
