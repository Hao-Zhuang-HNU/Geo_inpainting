#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import io
import math
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# PKL loading / rendering
# -----------------------------

def _shim_numpy_core_for_pickle() -> None:
    """
    兼容部分旧 pkl 中可能引用的 numpy.core / numpy._core 路径。
    大多数情况下其实用不到，但保留这个 shim 更稳妥。
    """
    try:
        import numpy.core.multiarray  # noqa: F401
    except Exception:
        pass


def load_lines_from_pkl(pkl_path: str) -> Tuple[List[Tuple[float, float, float, float]], Optional[object]]:
    _shim_numpy_core_for_pickle()
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # 常见格式：{"lines": [[x1,y1,x2,y2], ...], "scores": [...]}
    if isinstance(data, dict) and "lines" in data:
        lines = data["lines"]
        scores = data.get("scores", None)
    else:
        # 兜底：如果 pkl 直接就是 lines
        lines = data
        scores = None

    if not isinstance(lines, (list, tuple, np.ndarray)):
        raise ValueError(f"[{pkl_path}] Unsupported 'lines' type: {type(lines)}")

    cleaned: List[Tuple[float, float, float, float]] = []
    for ln in lines:
        if isinstance(ln, np.ndarray):
            ln = ln.tolist()
        if isinstance(ln, (list, tuple)) and len(ln) == 4:
            x1, y1, x2, y2 = ln
            cleaned.append((float(x1), float(y1), float(x2), float(y2)))

    return cleaned, scores


def render_wireframe_mask(
    lines: List[Tuple[float, float, float, float]],
    size: int = 256,
    line_width: float = 1.0,
    invert_y: bool = False,
) -> np.ndarray:
    """
    按你给出的 save_wireframe 逻辑，直接渲染为 numpy 二值图。
    输出: HxW, uint8, 0/1
    """
    W = H = int(size)

    # 判断是否归一化坐标（0~1）：最大值 <= 1.5 视为归一化
    max_v = 0.0
    for x1, y1, x2, y2 in lines:
        max_v = max(max_v, abs(x1), abs(y1), abs(x2), abs(y2))
    normalized = (max_v <= 1.5)

    dpi = size
    fig = plt.figure(figsize=(1, 1), dpi=dpi, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
    ax.set_xlim(0, W - 1)
    ax.set_ylim(0, H - 1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    for x1, y1, x2, y2 in lines:
        if normalized:
            x1p = x1 * (W - 1)
            y1p = y1 * (H - 1)
            x2p = x2 * (W - 1)
            y2p = y2 * (H - 1)
        else:
            x1p, y1p, x2p, y2p = x1, y1, x2, y2

        # 按要求交换 x/y：plot([y1,y2], [x1,x2])
        ax.plot([y1p, y2p], [x1p, x2p], color="white", linewidth=line_width, solid_capstyle="round")

    if invert_y:
        ax.invert_yaxis()

    # 渲染到内存，不落盘
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="black")
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("L")
    arr = np.array(img, dtype=np.uint8)

    # Matplotlib 抗锯齿后是灰度，阈值化成二值图
    mask = (arr > 127).astype(np.uint8)
    return mask


def save_mask_png(mask: np.ndarray, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(output_path)


# -----------------------------
# Input image loading
# -----------------------------

def load_pred_png_mask(png_path: str, size: int = 256, threshold: int = 127) -> np.ndarray:
    img = Image.open(png_path).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.NEAREST)
    arr = np.array(img, dtype=np.uint8)
    return (arr > threshold).astype(np.uint8)


# -----------------------------
# Metrics
# -----------------------------

def precision_recall_f1(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, int, int, int]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, np.logical_not(gt_b)).sum())
    fn = int(np.logical_and(np.logical_not(pred_b), gt_b).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


def _extract_points(mask: np.ndarray) -> np.ndarray:
    # 返回坐标为 [N, 2]，顺序为 (row, col)
    pts = np.argwhere(mask > 0)
    return pts.astype(np.float32)


def _mean_min_distance(src_pts: np.ndarray, dst_pts: np.ndarray, chunk_size: int = 2048) -> float:
    """
    计算 src 中每个点到 dst 最近点的平均欧氏距离。
    用 chunk 避免一次性构建过大的距离矩阵。
    """
    if len(src_pts) == 0 or len(dst_pts) == 0:
        return float("inf")

    mins = []
    dst = dst_pts[None, :, :]  # [1, M, 2]
    for i in range(0, len(src_pts), chunk_size):
        chunk = src_pts[i:i + chunk_size][:, None, :]  # [B, 1, 2]
        d2 = np.sum((chunk - dst) ** 2, axis=2)  # [B, M]
        mins.append(np.sqrt(np.min(d2, axis=1)))
    return float(np.mean(np.concatenate(mins, axis=0)))


def chamfer_distance(pred: np.ndarray, gt: np.ndarray, empty_value: float = float("inf")) -> float:
    pred_pts = _extract_points(pred)
    gt_pts = _extract_points(gt)

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return empty_value

    d_pred_to_gt = _mean_min_distance(pred_pts, gt_pts)
    d_gt_to_pred = _mean_min_distance(gt_pts, pred_pts)
    return 0.5 * (d_pred_to_gt + d_gt_to_pred)


# -----------------------------
# File matching
# -----------------------------

def find_files(root: str, suffixes: Tuple[str, ...]) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(tuple(s.lower() for s in suffixes)):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def rel_stem(path: str, root: str) -> str:
    rel = os.path.relpath(path, root)
    no_ext = os.path.splitext(rel)[0]
    return no_ext.replace("\\", "/")


def base_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def build_match_pairs(pred_root: str, gt_root: str) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    pred_files = find_files(pred_root, (".png",))
    gt_files = find_files(gt_root, (".pkl",))

    pred_rel = {rel_stem(p, pred_root): p for p in pred_files}
    gt_rel = {rel_stem(p, gt_root): p for p in gt_files}

    pairs: List[Tuple[str, str, str]] = []
    warnings: List[str] = []

    # 1) 优先按相对路径 stem 精确匹配
    matched_pred = set()
    matched_gt = set()
    common_rel = sorted(set(pred_rel.keys()) & set(gt_rel.keys()))
    for key in common_rel:
        pairs.append((key, pred_rel[key], gt_rel[key]))
        matched_pred.add(pred_rel[key])
        matched_gt.add(gt_rel[key])

    # 2) 再按 basename stem 匹配（仅当唯一）
    pred_base_map: Dict[str, List[str]] = defaultdict(list)
    gt_base_map: Dict[str, List[str]] = defaultdict(list)
    for p in pred_files:
        if p not in matched_pred:
            pred_base_map[base_stem(p)].append(p)
    for g in gt_files:
        if g not in matched_gt:
            gt_base_map[base_stem(g)].append(g)

    common_base = sorted(set(pred_base_map.keys()) & set(gt_base_map.keys()))
    for key in common_base:
        pred_list = pred_base_map[key]
        gt_list = gt_base_map[key]
        if len(pred_list) == 1 and len(gt_list) == 1:
            pairs.append((key, pred_list[0], gt_list[0]))
            matched_pred.add(pred_list[0])
            matched_gt.add(gt_list[0])
        else:
            warnings.append(
                f"[WARN] Basename '{key}' is ambiguous: pred={len(pred_list)}, gt={len(gt_list)}. Skip basename fallback."
            )

    # 3) 记录未匹配文件
    for p in pred_files:
        if p not in matched_pred:
            warnings.append(f"[WARN] Unmatched pred png: {p}")
    for g in gt_files:
        if g not in matched_gt:
            warnings.append(f"[WARN] Unmatched gt pkl: {g}")

    pairs.sort(key=lambda x: x[0])
    return pairs, warnings


# -----------------------------
# Main evaluation
# -----------------------------

def safe_mean(values: List[float], ignore_inf: bool = False) -> float:
    if ignore_inf:
        values = [v for v in values if np.isfinite(v)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def evaluate_one_pair(
    pred_png: str,
    gt_pkl: str,
    size: int,
    line_width: float,
    invert_y: bool,
    threshold: int,
    chamfer_empty_value: float,
    gt_render_save_path: Optional[str] = None,
) -> Dict[str, float]:
    pred_mask = load_pred_png_mask(pred_png, size=size, threshold=threshold)

    gt_lines, _ = load_lines_from_pkl(gt_pkl)
    gt_mask = render_wireframe_mask(
        gt_lines,
        size=size,
        line_width=line_width,
        invert_y=invert_y,
    )

    if gt_render_save_path is not None:
        save_mask_png(gt_mask, gt_render_save_path)

    precision, recall, f1, tp, fp, fn = precision_recall_f1(pred_mask, gt_mask)
    chamfer = chamfer_distance(pred_mask, gt_mask, empty_value=chamfer_empty_value)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "chamfer_distance": chamfer,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_pixels": int(pred_mask.sum()),
        "gt_pixels": int(gt_mask.sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="递归遍历生成线框 png 与真实线框 pkl，渲染 GT 后评估 Precision / Recall / F1 / Chamfer Distance"
    )
    parser.add_argument("--input_line", type=str, required=True, help="生成线框 png 根目录")
    parser.add_argument("--gt_line", type=str, required=True, help="真实线框 pkl 根目录")
    parser.add_argument("--size", type=int, default=256, help="渲染/比较图尺寸，默认 256")
    parser.add_argument("--line_width", type=float, default=1.0, help="GT 线宽，默认 1.0")
    parser.add_argument("--invert_y", action="store_true", help="渲染 GT 时是否 invert_y")
    parser.add_argument("--threshold", type=int, default=127, help="png 二值化阈值，默认 127")
    parser.add_argument(
        "--save_gt_render_dir",
        type=str,
        default=None,
        help="可选：将 GT pkl 渲染后的 png 保存到该目录，保持相对路径结构",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default=None,
        help="可选：逐文件结果保存为 csv",
    )
    parser.add_argument(
        "--chamfer_empty_value",
        type=float,
        default=1e6,
        help="当 pred 或 gt 为空时，Chamfer Distance 返回值，默认 1e6",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isdir(args.input_line):
        print(f"[ERROR] --input_line is not a directory: {args.input_line}", file=sys.stderr)
        return 1
    if not os.path.isdir(args.gt_line):
        print(f"[ERROR] --gt_line is not a directory: {args.gt_line}", file=sys.stderr)
        return 1

    pairs, warnings = build_match_pairs(args.input_line, args.gt_line)
    for w in warnings:
        print(w)

    if not pairs:
        print("[ERROR] No matched png/pkl pairs found.", file=sys.stderr)
        return 2

    rows = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    chamfers: List[float] = []

    print(f"[INFO] Matched pairs: {len(pairs)}")
    print("=" * 120)

    for idx, (key, pred_png, gt_pkl) in enumerate(pairs, start=1):
        gt_render_save_path = None
        if args.save_gt_render_dir is not None:
            gt_render_save_path = os.path.join(args.save_gt_render_dir, key + ".png")

        metrics = evaluate_one_pair(
            pred_png=pred_png,
            gt_pkl=gt_pkl,
            size=args.size,
            line_width=args.line_width,
            invert_y=args.invert_y,
            threshold=args.threshold,
            chamfer_empty_value=args.chamfer_empty_value,
            gt_render_save_path=gt_render_save_path,
        )

        row = {
            "id": key,
            "pred_png": pred_png,
            "gt_pkl": gt_pkl,
            **metrics,
        }
        rows.append(row)

        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1"])
        chamfers.append(metrics["chamfer_distance"])

        print(
            f"[{idx:04d}/{len(pairs):04d}] {key}\n"
            f"  Precision={metrics['precision']:.6f}  Recall={metrics['recall']:.6f}  "
            f"F1={metrics['f1']:.6f}  Chamfer={metrics['chamfer_distance']:.6f}"
        )

    mean_precision = safe_mean(precisions)
    mean_recall = safe_mean(recalls)
    mean_f1 = safe_mean(f1s)
    mean_chamfer = safe_mean(chamfers)
    mean_chamfer_ignore_inf = safe_mean(chamfers, ignore_inf=True)

    print("=" * 120)
    print("[SUMMARY]")
    print(f"Pairs               : {len(rows)}")
    print(f"Mean Precision      : {mean_precision:.6f}")
    print(f"Mean Recall         : {mean_recall:.6f}")
    print(f"Mean F1             : {mean_f1:.6f}")
    print(f"Mean Chamfer Dist   : {mean_chamfer:.6f}")
    print(f"Mean Chamfer (finite only): {mean_chamfer_ignore_inf:.6f}")

    if args.csv_out is not None:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        fieldnames = [
            "id", "pred_png", "gt_pkl",
            "precision", "recall", "f1", "chamfer_distance",
            "tp", "fp", "fn", "pred_pixels", "gt_pixels"
        ]
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            writer.writerow({
                "id": "__MEAN__",
                "precision": mean_precision,
                "recall": mean_recall,
                "f1": mean_f1,
                "chamfer_distance": mean_chamfer,
            })
        print(f"[INFO] CSV saved to: {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
