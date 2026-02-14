#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overlay masks on RGB images by list files.

Usage:
  python overlay_masks.py \
      --images_list .txt \
      --masks_list .txt \
      --output_path /abs/path/to/output_path \
      --color "#FFFFF" \
      --alpha 1 \
      --mask_threshold 128

Notes:
- images_list.txt 与 masks_list.txt 每行一个绝对路径，支持常见图像格式（png/jpg/webp等）。
- 若 mask 为三通道，会自动转灰度用于阈值判断；非零或高于阈值的像素作为覆盖区域。
- 叠加颜色用十六进制或 "R,G,B"（如 "0,255,0"）均可。
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image


def read_list_file(list_path: str) -> List[str]:
    if not os.path.isfile(list_path):
        raise FileNotFoundError(f"List file not found: {list_path}")
    with open(list_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # 过滤不存在的文件并给出提示
    paths = []
    for p in lines:
        if os.path.isfile(p):
            paths.append(p)
        else:
            print(f"[WARN] File not found (skip): {p}", file=sys.stderr)
    if not paths:
        raise ValueError(f"No valid paths found in {list_path}")
    return paths


def parse_color(color_str: str) -> Tuple[int, int, int]:
    """支持 '#RRGGBB' 或 'R,G,B' 两种格式"""
    color_str = color_str.strip()
    if color_str.startswith("#"):
        hexstr = color_str.lstrip("#")
        if len(hexstr) != 6:
            raise ValueError(f"Invalid hex color: {color_str}")
        r = int(hexstr[0:2], 16)
        g = int(hexstr[2:4], 16)
        b = int(hexstr[4:6], 16)
        return (r, g, b)
    else:
        try:
            parts = [int(x) for x in color_str.split(",")]
            assert len(parts) == 3
            for v in parts:
                if not (0 <= v <= 255):
                    raise ValueError
            return tuple(parts)  # type: ignore
        except Exception:
            raise ValueError(f"Invalid color format: {color_str}. Use '#RRGGBB' or 'R,G,B'.")


def load_image_as_rgb(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def load_mask_as_gray(path: str, size: Tuple[int, int]) -> Image.Image:
    """
    将 mask 读为灰度图并缩放到目标 size (W,H)。
    支持单通道或三通道 mask；三通道会转灰度用于阈值。
    """
    m = Image.open(path)
    if m.mode not in ("L", "I;16", "I", "F"):
        m = m.convert("L")
    # resize 到与 RGB 一致
    if m.size != size:
        m = m.resize(size, resample=Image.NEAREST)
    return m


def overlay_mask_on_rgb(
    rgb_img: Image.Image,
    mask_gray: Image.Image,
    color: Tuple[int, int, int],
    alpha: float,
    mask_threshold: int,
) -> Image.Image:
    """
    在 rgb_img 上用给定 color/alpha 叠加 mask_gray>threshold 的区域。
    返回新的 RGB 图像。
    """
    # 转 numpy
    rgb_np = np.asarray(rgb_img, dtype=np.uint8)
    if mask_gray.mode != "L":
        mask_gray = mask_gray.convert("L")
    mask_np = np.asarray(mask_gray, dtype=np.uint8)

    # 生成二值mask：> threshold 的位置为 True
    m = mask_np > mask_threshold  # shape: (H, W), bool

    if not m.any():
        # 没有覆盖区域，直接返回拷贝
        return rgb_img.copy()

    # 叠加层
    overlay = np.zeros_like(rgb_np, dtype=np.float32)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]

    out = rgb_np.astype(np.float32)
    # 仅在 m 区域内做 alpha 混合： out = (1-a)*rgb + a*color
    a = float(np.clip(alpha, 0.0, 1.0))
    # 扩展 mask 到三通道
    m3 = np.stack([m, m, m], axis=-1)

    out[m3] = (1.0 - a) * out[m3] + a * overlay[m3]

    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="Overlay masks on RGB images by list files.")
    parser.add_argument("--images_list", required=True, help="Text file: absolute paths of RGB images, one per line.")
    parser.add_argument("--masks_list", required=True, help="Text file: absolute paths of mask images, one per line.")
    parser.add_argument("--output_path", required=True, help="Directory to save overlaid images.")
    parser.add_argument("--color", default="#FFFFFF", help="Overlay color. '#FFFFFF' or #000000'")
    parser.add_argument("--alpha", type=float, default=1, help="Overlay transparency in [0,1]. Default 1.")
    parser.add_argument("--mask_threshold", type=int, default=128, help="Threshold (0-255). >threshold means masked.")
    parser.add_argument("--suffix", default="", help="Suffix for output filename. Default ''.")
    args = parser.parse_args()

    rgb_paths = read_list_file(args.images_list)
    mask_paths = read_list_file(args.masks_list)

    os.makedirs(args.output_path, exist_ok=True)
    color = parse_color(args.color)
    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    thr = int(np.clip(args.mask_threshold, 0, 255))

    n_rgb = len(rgb_paths)
    n_mask = len(mask_paths)

    print(f"[INFO] RGB images: {n_rgb}, masks: {n_mask}")
    if n_mask == 0:
        print("[ERROR] No valid masks.", file=sys.stderr)
        sys.exit(2)

    used_mask_count = 0
    for i, rgb_p in enumerate(rgb_paths):
        try:
            rgb = load_image_as_rgb(rgb_p)
        except Exception as e:
            print(f"[ERROR] Failed to read RGB: {rgb_p} | {e}", file=sys.stderr)
            continue

        # 选择 mask：若不足则循环复用；若更多则自然会有未使用的（被舍弃）
        mi = i % n_mask
        mask_p = mask_paths[mi]
        try:
            mask = load_mask_as_gray(mask_p, size=rgb.size)  # PIL size is (W, H)
        except Exception as e:
            print(f"[ERROR] Failed to read/resize mask: {mask_p} | {e}", file=sys.stderr)
            continue

        out_img = overlay_mask_on_rgb(rgb, mask, color=color, alpha=alpha, mask_threshold=thr)

        base = os.path.basename(rgb_p)
        stem, ext = os.path.splitext(base)
        out_name = f"{stem}{args.suffix}{ext if ext else '.png'}"
        out_path = os.path.join(args.output_path, out_name)
        try:
            out_img.save(out_path)
            used_mask_count += 1 if mi < n_mask else 0
            print(f"[OK] {i+1}/{n_rgb} -> {out_path}  (mask: {os.path.basename(mask_p)})")
        except Exception as e:
            print(f"[ERROR] Failed to save: {out_path} | {e}", file=sys.stderr)

    # 提示是否有 mask 未被使用（当 mask 多于 rgb）
    if n_mask > n_rgb:
        unused = n_mask - n_rgb
        print(f"[INFO] {unused} masks were not used (more masks than RGBs).")

    print("[DONE] All processed.")


if __name__ == "__main__":
    main()
