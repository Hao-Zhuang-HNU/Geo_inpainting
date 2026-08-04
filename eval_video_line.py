#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_video_line.py

Structure-oriented evaluation for video/image inpainting.
Only computes:
  - Line-F1_hole      ↑
  - Line-Chamfer_hole ↓
  - Canny-F1_hole     ↑
  - Gradient-L1_hole  ↓

The input/output/alignment style follows eval_video.py:
  --pre_path, --gt_path, --mask_path, --pix, --resize, --seq_name,
  --jump_image, --jump_mask, --out, --debug

Notes:
  1. "Line" is extracted with OpenCV LSD by default. This is a model-free
     line-segment proxy, not HAWP/LSM. If LSD fails, the script can fall back
     to Canny for the line map.
  2. F1 uses a small spatial tolerance through binary dilation. This makes the
     score less sensitive to 1-2 px shifts.
  3. Chamfer is the symmetric average nearest-line distance in pixels inside
     the hole region. Lower is better.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from scipy.ndimage import binary_dilation, distance_transform_edt

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ----------------------- IO / alignment helpers -----------------------
def natural_key(p: Path) -> List[Any]:
    s = p.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=natural_key)
    return files


def pil_to_np_rgb01(pil_img: Image.Image, size: Optional[int] = None) -> np.ndarray:
    img = pil_img.convert("RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def rgb01_to_gray_u8(rgb01: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError(f"opencv-python is required. Import error: {_CV2_IMPORT_ERROR}")
    rgb_u8 = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)


def load_mask(mask_path: Path, size: Optional[int] = None, invert: bool = False) -> np.ndarray:
    m = Image.open(mask_path).convert("L")
    if size is not None and m.size != (size, size):
        m = m.resize((size, size), resample=Image.NEAREST)
    arr = np.asarray(m).astype(np.float32) / 255.0
    mask = (arr > 0.5).astype(np.float32)  # 1=hole
    if invert:
        mask = 1.0 - mask
    return mask


def collect_masks_index_aligned(mask_root: Path, seq_name: str) -> List[Path]:
    """
    Prefer mask_root/seq_name/* if it exists; otherwise search masks under
    mask_root whose path contains seq_name; otherwise take all masks under
    mask_root. Always natural-sorted.
    """
    cand_dir = mask_root / seq_name
    if cand_dir.exists():
        return list_images(cand_dir)

    all_masks = list_images(mask_root)
    filtered = [p for p in all_masks if seq_name in p.as_posix().split("/")]
    if filtered:
        filtered.sort(key=natural_key)
        return filtered
    return all_masks


# ----------------------- JSON/CSV safety -----------------------
def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    return obj


def csv_safe_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, (np.floating, float)):
            fv = float(v)
            out[k] = "" if math.isnan(fv) or math.isinf(fv) else fv
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (list, tuple)):
            out[k] = json.dumps(make_json_safe(v), ensure_ascii=False)
        else:
            out[k] = v
    return out


def nanmean(xs: Sequence[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def valid_count(xs: Sequence[float]) -> int:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return 0
    return int(np.sum(np.isfinite(arr)))


# ----------------------- structure map extraction -----------------------
def canny_map(gray_u8: np.ndarray, low: int, high: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError(f"opencv-python is required. Import error: {_CV2_IMPORT_ERROR}")
    edges = cv2.Canny(gray_u8, threshold1=low, threshold2=high, apertureSize=3, L2gradient=True)
    return edges > 0


def lsd_line_map(
    gray_u8: np.ndarray,
    min_length: float = 8.0,
    thickness: int = 1,
    fallback_to_canny: bool = True,
    canny_low: int = 100,
    canny_high: int = 200,
) -> np.ndarray:
    """Extract a binary line-segment map with OpenCV LSD."""
    if cv2 is None:
        raise RuntimeError(f"opencv-python is required. Import error: {_CV2_IMPORT_ERROR}")

    h, w = gray_u8.shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)

    detector = None
    if hasattr(cv2, "createLineSegmentDetector"):
        try:
            detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        except Exception:
            try:
                detector = cv2.createLineSegmentDetector()
            except Exception:
                detector = None

    if detector is None:
        if fallback_to_canny:
            return canny_map(gray_u8, canny_low, canny_high)
        return canvas > 0

    try:
        detected = detector.detect(gray_u8)
        lines = detected[0] if isinstance(detected, tuple) else detected
    except Exception:
        lines = None

    if lines is None:
        if fallback_to_canny:
            return canny_map(gray_u8, canny_low, canny_high)
        return canvas > 0

    lines = np.asarray(lines).reshape(-1, 4)
    for x1, y1, x2, y2 in lines:
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        if length < min_length:
            continue
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.line(canvas, p1, p2, 255, thickness=max(1, int(thickness)), lineType=cv2.LINE_AA)

    return canvas > 0


def sobel_gradient_magnitude(gray_u8: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError(f"opencv-python is required. Import error: {_CV2_IMPORT_ERROR}")
    gray = gray_u8.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def restrict_to_hole(binary_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.logical_and(binary_map.astype(bool), mask > 0.5)


# ----------------------- metrics -----------------------
def binary_f1_with_tolerance(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> Tuple[float, float, float, int, int]:
    """
    Tolerant binary F1.
    pred/gt should already be restricted to the evaluation region.

    Returns: f1, precision, recall, pred_pixels, gt_pixels
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_count = int(pred.sum())
    gt_count = int(gt.sum())

    # Empty-GT frames contain no structural target in the hole; leave as NaN so
    # they do not dominate the dataset average. If only one side is empty, score 0.
    if pred_count == 0 and gt_count == 0:
        return float("nan"), float("nan"), float("nan"), pred_count, gt_count
    if pred_count == 0 or gt_count == 0:
        return 0.0, 0.0 if pred_count > 0 else float("nan"), 0.0 if gt_count > 0 else float("nan"), pred_count, gt_count

    if tolerance > 0:
        structure = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
        gt_dil = binary_dilation(gt, structure=structure)
        pred_dil = binary_dilation(pred, structure=structure)
    else:
        gt_dil = gt
        pred_dil = pred

    matched_pred = int(np.logical_and(pred, gt_dil).sum())
    matched_gt = int(np.logical_and(gt, pred_dil).sum())

    precision = matched_pred / max(pred_count, 1)
    recall = matched_gt / max(gt_count, 1)
    if precision + recall <= 1e-12:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return float(f1), float(precision), float(recall), pred_count, gt_count


def symmetric_chamfer_pixels(pred: np.ndarray, gt: np.ndarray, empty_penalty: float) -> float:
    """
    Symmetric Chamfer distance in pixels between binary maps.
    pred/gt should already be restricted to the hole.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_count = int(pred.sum())
    gt_count = int(gt.sum())

    if pred_count == 0 and gt_count == 0:
        return float("nan")
    if pred_count == 0 or gt_count == 0:
        return float(empty_penalty)

    # distance_transform_edt(~gt): for every pixel, distance to nearest gt=True pixel
    dt_to_gt = distance_transform_edt(~gt)
    dt_to_pred = distance_transform_edt(~pred)
    pred_to_gt = float(np.mean(dt_to_gt[pred]))
    gt_to_pred = float(np.mean(dt_to_pred[gt]))
    return 0.5 * (pred_to_gt + gt_to_pred)


def gradient_l1_hole(pred_grad: np.ndarray, gt_grad: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0.5
    denom = int(m.sum())
    if denom <= 0:
        return float("nan")
    return float(np.mean(np.abs(pred_grad[m] - gt_grad[m])))


def save_binary_png(path: Path, arr: np.ndarray) -> None:
    Image.fromarray((arr.astype(np.uint8) * 255), mode="L").save(path)


# ----------------------- main -----------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structure metrics for video/image inpainting.")
    parser.add_argument("--pre_path", type=str, required=True, help="Predicted/inpainted frames directory")
    parser.add_argument("--gt_path", type=str, required=True, help="Ground-truth frames directory")
    parser.add_argument("--mask_path", type=str, default="", help="Mask directory. White/nonzero means hole by default.")
    parser.add_argument("--pix", type=int, default=256, help="Square resize size used when --resize is set")
    parser.add_argument("--resize", action="store_true", help="Resize pred/gt/mask to --pix x --pix before evaluation")
    parser.add_argument("--seq_name", type=str, default="", help="Mask subdirectory name. Default: gt_path basename")
    parser.add_argument("--out", type=str, default="line_metrics_out", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Also save debug_alignment.json/csv and debug_summary.txt")

    # Alignment behavior follows eval_video.py.
    parser.add_argument("--jump_image", type=int, default=0,
                        help="Skip first N pred frames. Example: pred[i+N] <-> gt[i+N] <-> mask[i]")
    parser.add_argument("--jump_mask", type=int, default=0,
                        help="Skip first N gt/mask frames. Example: pred[i] <-> gt[i+N] <-> mask[i+N]")
    parser.add_argument("--repeat_short_masks", action="store_true",
                        help="If masks < frames, reuse masks by modulo. Default: missing later masks become None.")
    parser.add_argument("--invert_mask", action="store_true", help="Invert mask: black/non-hole becomes evaluation hole.")

    # Structure extraction and metric parameters.
    parser.add_argument("--line_extractor", type=str, default="lsd", choices=["lsd", "canny"],
                        help="Extractor for Line-F1/Line-Chamfer. Default: LSD line segments.")
    parser.add_argument("--lsd_min_length", type=float, default=8.0, help="Minimum LSD segment length in pixels")
    parser.add_argument("--line_thickness", type=int, default=1, help="Rasterized LSD line thickness")
    parser.add_argument("--canny_low", type=int, default=100, help="Canny low threshold")
    parser.add_argument("--canny_high", type=int, default=200, help="Canny high threshold")
    parser.add_argument("--f1_tolerance", type=int, default=2,
                        help="Pixel tolerance for tolerant F1 matching. Default: 2")
    parser.add_argument("--chamfer_empty_penalty", type=float, default=-1.0,
                        help="Chamfer value when only pred or gt has line pixels. Default: image diagonal")
    parser.add_argument("--save_debug_maps", action="store_true",
                        help="Save pred/gt line/canny maps for first --max_debug_maps frames")
    parser.add_argument("--max_debug_maps", type=int, default=20)

    args = parser.parse_args()

    if cv2 is None:
        raise RuntimeError(f"opencv-python is required for this script. Import error: {_CV2_IMPORT_ERROR}")

    pred_root = Path(args.pre_path)
    gt_root = Path(args.gt_path)
    mask_root = Path(args.mask_path) if args.mask_path else None

    if not pred_root.exists():
        raise FileNotFoundError(f"pre_path not found: {pred_root}")
    if not gt_root.exists():
        raise FileNotFoundError(f"gt_path not found: {gt_root}")
    if mask_root is not None and not mask_root.exists():
        raise FileNotFoundError(f"mask_path not found: {mask_root}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_dir = out_dir / "debug_maps"
    if args.save_debug_maps:
        map_dir.mkdir(parents=True, exist_ok=True)

    pix = args.pix if args.resize else None

    pred_frames = list_images(pred_root)
    gt_frames = list_images(gt_root)
    if len(pred_frames) == 0 or len(gt_frames) == 0:
        raise RuntimeError("pred or gt directory contains no images")

    N0 = min(len(pred_frames), len(gt_frames))
    if len(pred_frames) != len(gt_frames):
        print(f"[WARN] pred={len(pred_frames)} gt={len(gt_frames)} -> index-align first N={N0} frames")
    pred_frames = pred_frames[:N0]
    gt_frames = gt_frames[:N0]
    frame_indices = list(range(N0))

    mask_frames: List[Optional[Path]] = [None] * N0
    mask_indices: List[Optional[int]] = [None] * N0
    if mask_root is not None:
        seq_name = args.seq_name.strip() if args.seq_name.strip() else gt_root.name
        masks = collect_masks_index_aligned(mask_root, seq_name)
        if len(masks) == 0:
            print("[WARN] mask_path contains no mask images; hole metrics will be NaN")
        elif len(masks) < N0:
            if args.repeat_short_masks:
                print(f"[INFO] masks={len(masks)} < frames={N0} -> repeat masks by modulo")
                for i in range(N0):
                    mi = i % len(masks)
                    mask_frames[i] = masks[mi]
                    mask_indices[i] = mi
            else:
                print(f"[WARN] masks={len(masks)} < frames={N0} -> remaining frames have no mask")
                for i in range(len(masks)):
                    mask_frames[i] = masks[i]
                    mask_indices[i] = i
        else:
            if len(masks) > N0:
                print(f"[INFO] masks={len(masks)} > frames={N0} -> take first N masks by natural order")
            for i in range(N0):
                mask_frames[i] = masks[i]
                mask_indices[i] = i

    if args.jump_image < 0 or args.jump_mask < 0:
        raise ValueError("--jump_image and --jump_mask must be non-negative integers")

    ji = int(args.jump_image)
    jm = int(args.jump_mask)
    pred_start = ji
    gt_start = ji + jm
    mask_start = jm

    if ji > 0 or jm > 0:
        max_len = min(
            len(pred_frames) - pred_start,
            len(gt_frames) - gt_start,
            len(mask_frames) - mask_start,
        )
        if max_len <= 0:
            raise RuntimeError(
                f"After temporal jump, no valid pairs remain: "
                f"pred={len(pred_frames)}, gt={len(gt_frames)}, mask={len(mask_frames)}, "
                f"jump_image={ji}, jump_mask={jm}"
            )
        pred_frames = pred_frames[pred_start:pred_start + max_len]
        gt_frames = gt_frames[gt_start:gt_start + max_len]
        mask_frames = mask_frames[mask_start:mask_start + max_len]
        frame_indices = frame_indices[pred_start:pred_start + max_len]
        gt_indices = list(range(gt_start, gt_start + max_len))
        mask_indices = mask_indices[mask_start:mask_start + max_len]
        print(f"[INFO] Applied temporal jump: jump_image={ji}, jump_mask={jm}")
        print(f"[INFO] Pair rule: pred[i+{pred_start}] <-> gt[i+{gt_start}] <-> mask[i+{mask_start}]")
    else:
        N = min(len(pred_frames), len(gt_frames), len(mask_frames))
        pred_frames = pred_frames[:N]
        gt_frames = gt_frames[:N]
        mask_frames = mask_frames[:N]
        frame_indices = frame_indices[:N]
        gt_indices = list(range(N))
        mask_indices = mask_indices[:N]

    N = len(pred_frames)
    if N == 0:
        raise RuntimeError("No valid aligned frames remain")

    line_f1_list: List[float] = []
    line_precision_list: List[float] = []
    line_recall_list: List[float] = []
    line_chamfer_list: List[float] = []
    canny_f1_list: List[float] = []
    canny_precision_list: List[float] = []
    canny_recall_list: List[float] = []
    gradient_l1_list: List[float] = []
    debug_records: List[Dict[str, Any]] = []

    for idx, (p, g, m) in enumerate(tqdm(list(zip(pred_frames, gt_frames, mask_frames)), desc="Line metrics", total=N)):
        pred_pil = Image.open(p).convert("RGB")
        gt_pil = Image.open(g).convert("RGB")
        pred_orig_size = pred_pil.size
        gt_orig_size = gt_pil.size

        pred = pil_to_np_rgb01(pred_pil, size=pix)
        gt = pil_to_np_rgb01(gt_pil, size=pix)
        h, w = pred.shape[:2]
        empty_penalty = float(args.chamfer_empty_penalty)
        if empty_penalty < 0:
            empty_penalty = float(math.hypot(h, w))

        pred_gray = rgb01_to_gray_u8(pred)
        gt_gray = rgb01_to_gray_u8(gt)

        mask = None
        mask_ratio = float("nan")
        hole_pixels = None
        mask_size = None
        if m is not None and Path(m).exists():
            mask_pil = Image.open(m).convert("L")
            mask_size = mask_pil.size
            mask = load_mask(Path(m), size=pix, invert=args.invert_mask)
            mask_ratio = float(mask.mean())
            hole_pixels = int(np.sum(mask > 0.5))
        else:
            # No mask means no hole-region evaluation for this frame.
            mask = np.zeros((h, w), dtype=np.float32)

        if args.line_extractor == "lsd":
            pred_line = lsd_line_map(
                pred_gray,
                min_length=args.lsd_min_length,
                thickness=args.line_thickness,
                fallback_to_canny=True,
                canny_low=args.canny_low,
                canny_high=args.canny_high,
            )
            gt_line = lsd_line_map(
                gt_gray,
                min_length=args.lsd_min_length,
                thickness=args.line_thickness,
                fallback_to_canny=True,
                canny_low=args.canny_low,
                canny_high=args.canny_high,
            )
        else:
            pred_line = canny_map(pred_gray, args.canny_low, args.canny_high)
            gt_line = canny_map(gt_gray, args.canny_low, args.canny_high)

        pred_canny = canny_map(pred_gray, args.canny_low, args.canny_high)
        gt_canny = canny_map(gt_gray, args.canny_low, args.canny_high)

        pred_line_h = restrict_to_hole(pred_line, mask)
        gt_line_h = restrict_to_hole(gt_line, mask)
        pred_canny_h = restrict_to_hole(pred_canny, mask)
        gt_canny_h = restrict_to_hole(gt_canny, mask)

        line_f1, line_p, line_r, pred_line_pixels, gt_line_pixels = binary_f1_with_tolerance(
            pred_line_h, gt_line_h, tolerance=args.f1_tolerance
        )
        line_cd = symmetric_chamfer_pixels(pred_line_h, gt_line_h, empty_penalty=empty_penalty)

        canny_f1, canny_p, canny_r, pred_canny_pixels, gt_canny_pixels = binary_f1_with_tolerance(
            pred_canny_h, gt_canny_h, tolerance=args.f1_tolerance
        )

        pred_grad = sobel_gradient_magnitude(pred_gray)
        gt_grad = sobel_gradient_magnitude(gt_gray)
        grad_l1 = gradient_l1_hole(pred_grad, gt_grad, mask)

        line_f1_list.append(line_f1)
        line_precision_list.append(line_p)
        line_recall_list.append(line_r)
        line_chamfer_list.append(line_cd)
        canny_f1_list.append(canny_f1)
        canny_precision_list.append(canny_p)
        canny_recall_list.append(canny_r)
        gradient_l1_list.append(grad_l1)

        record = {
            "eval_index": idx,
            "frame_original_index": frame_indices[idx],
            "gt_original_index": gt_indices[idx],
            "mask_original_index": mask_indices[idx],
            "pred_path": str(p),
            "gt_path": str(g),
            "mask_path": str(m) if m is not None else None,
            "pred_name": p.name,
            "gt_name": g.name,
            "mask_name": Path(m).name if m is not None else None,
            "pred_stem": p.stem,
            "gt_stem": g.stem,
            "mask_stem": Path(m).stem if m is not None else None,
            "pred_orig_size": list(pred_orig_size),
            "gt_orig_size": list(gt_orig_size),
            "mask_orig_size": list(mask_size) if mask_size is not None else None,
            "pred_gt_same_name": p.stem == g.stem,
            "gt_mask_same_name": (g.stem == Path(m).stem) if m is not None else None,
            "mask_ratio": mask_ratio,
            "hole_pixels": hole_pixels,
            "Line_F1_hole": line_f1,
            "Line_Precision_hole": line_p,
            "Line_Recall_hole": line_r,
            "Line_Chamfer_hole": line_cd,
            "Canny_F1_hole": canny_f1,
            "Canny_Precision_hole": canny_p,
            "Canny_Recall_hole": canny_r,
            "Gradient_L1_hole": grad_l1,
            "pred_line_pixels_hole": pred_line_pixels,
            "gt_line_pixels_hole": gt_line_pixels,
            "pred_canny_pixels_hole": pred_canny_pixels,
            "gt_canny_pixels_hole": gt_canny_pixels,
            "line_extractor": args.line_extractor,
            "f1_tolerance": args.f1_tolerance,
            "canny_low": args.canny_low,
            "canny_high": args.canny_high,
            "jump_image": args.jump_image,
            "jump_mask": args.jump_mask,
            "aligned_pair": f"pred[{frame_indices[idx]}] <-> gt[{gt_indices[idx]}] <-> mask[{mask_indices[idx]}]",
        }
        debug_records.append(record)

        if args.save_debug_maps and idx < args.max_debug_maps:
            prefix = f"{idx:04d}_{p.stem}"
            save_binary_png(map_dir / f"{prefix}_pred_line.png", pred_line)
            save_binary_png(map_dir / f"{prefix}_gt_line.png", gt_line)
            save_binary_png(map_dir / f"{prefix}_pred_line_hole.png", pred_line_h)
            save_binary_png(map_dir / f"{prefix}_gt_line_hole.png", gt_line_h)
            save_binary_png(map_dir / f"{prefix}_pred_canny.png", pred_canny)
            save_binary_png(map_dir / f"{prefix}_gt_canny.png", gt_canny)
            save_binary_png(map_dir / f"{prefix}_mask.png", mask > 0.5)

    results: Dict[str, Any] = {
        "count_frames": N,
        "valid_Line_F1_hole_frames": valid_count(line_f1_list),
        "valid_Line_Chamfer_hole_frames": valid_count(line_chamfer_list),
        "valid_Canny_F1_hole_frames": valid_count(canny_f1_list),
        "valid_Gradient_L1_hole_frames": valid_count(gradient_l1_list),
        "Line_F1_hole": nanmean(line_f1_list),
        "Line_Precision_hole": nanmean(line_precision_list),
        "Line_Recall_hole": nanmean(line_recall_list),
        "Line_Chamfer_hole": nanmean(line_chamfer_list),
        "Canny_F1_hole": nanmean(canny_f1_list),
        "Canny_Precision_hole": nanmean(canny_precision_list),
        "Canny_Recall_hole": nanmean(canny_recall_list),
        "Gradient_L1_hole": nanmean(gradient_l1_list),
        "line_extractor": args.line_extractor,
        "lsd_min_length": float(args.lsd_min_length),
        "line_thickness": int(args.line_thickness),
        "f1_tolerance": int(args.f1_tolerance),
        "canny_low": int(args.canny_low),
        "canny_high": int(args.canny_high),
        "chamfer_empty_penalty": float(args.chamfer_empty_penalty),
        "resize": bool(args.resize),
        "pix": int(args.pix),
        "invert_mask": bool(args.invert_mask),
        "jump_image": int(args.jump_image),
        "jump_mask": int(args.jump_mask),
    }

    print("\n========== Line Metrics ==========")
    for k, v in results.items():
        if isinstance(v, float):
            if math.isnan(v):
                print(f"{k:>30s}: NaN")
            else:
                print(f"{k:>30s}: {v:.6f}")
        else:
            print(f"{k:>30s}: {v}")
    print("==================================\n")

    # Always save valid JSON and per-frame CSV/JSON.
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(results), f, indent=2, ensure_ascii=False, allow_nan=False)

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(csv_safe_row(results))

    per_frame_json = out_dir / "per_frame_metrics.json"
    per_frame_csv = out_dir / "per_frame_metrics.csv"
    with open(per_frame_json, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(debug_records), f, indent=2, ensure_ascii=False, allow_nan=False)

    if debug_records:
        with open(per_frame_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(debug_records[0].keys()))
            writer.writeheader()
            writer.writerows([csv_safe_row(r) for r in debug_records])

    if args.debug:
        debug_json = out_dir / "debug_alignment.json"
        debug_csv = out_dir / "debug_alignment.csv"
        debug_txt = out_dir / "debug_summary.txt"
        with open(debug_json, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(debug_records), f, indent=2, ensure_ascii=False, allow_nan=False)
        if debug_records:
            with open(debug_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(debug_records[0].keys()))
                writer.writeheader()
                writer.writerows([csv_safe_row(r) for r in debug_records])

        mismatch_pred_gt = sum([not x["pred_gt_same_name"] for x in debug_records])
        mismatch_gt_mask = sum([(x["gt_mask_same_name"] is False) for x in debug_records])
        valid_mask_ratios = [x["mask_ratio"] for x in debug_records if isinstance(x["mask_ratio"], float) and math.isfinite(x["mask_ratio"])]

        with open(debug_txt, "w", encoding="utf-8") as f:
            f.write("========== Debug Summary ==========\n")
            f.write(f"frames: {len(debug_records)}\n")
            if debug_records:
                f.write(f"first_aligned_pair: {debug_records[0]['aligned_pair']}\n")
                f.write(f"last_aligned_pair : {debug_records[-1]['aligned_pair']}\n")
            f.write(f"pred_gt_name_mismatch: {mismatch_pred_gt}\n")
            f.write(f"gt_mask_name_mismatch: {mismatch_gt_mask}\n")
            if valid_mask_ratios:
                f.write(f"mask_ratio_mean: {float(np.mean(valid_mask_ratios)):.6f}\n")
                f.write(f"mask_ratio_min : {float(np.min(valid_mask_ratios)):.6f}\n")
                f.write(f"mask_ratio_max : {float(np.max(valid_mask_ratios)):.6f}\n")
            f.write("\nMetric meanings:\n")
            f.write("Line_F1_hole: higher is better; tolerant F1 between LSD line maps in the hole.\n")
            f.write("Line_Chamfer_hole: lower is better; symmetric nearest-line distance in pixels.\n")
            f.write("Canny_F1_hole: higher is better; tolerant F1 between Canny edge maps in the hole.\n")
            f.write("Gradient_L1_hole: lower is better; Sobel gradient magnitude L1 error in the hole.\n")

        print(f"Saved: {debug_json}")
        print(f"Saved: {debug_csv}")
        print(f"Saved: {debug_txt}")

    print(f"Saved: {out_dir / 'metrics.json'}")
    print(f"Saved: {out_dir / 'metrics.csv'}")
    print(f"Saved: {per_frame_json}")
    print(f"Saved: {per_frame_csv}")
    if args.save_debug_maps:
        print(f"Saved debug maps under: {map_dir}")


if __name__ == "__main__":
    main()
