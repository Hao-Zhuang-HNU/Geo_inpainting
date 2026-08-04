#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_3ch_line_cases.py

从 prepare_soft_lines.py --save_debug 生成的 line_debug/ 中，自动寻找最能体现
3-channel reliability-aware line guidance 作用的帧，并可生成热力图/拼图。

核心目标：找到 soft 与 center 有明显差异、reliability 对 soft 候选线有明显降权、
同时线条密度不过低的代表性帧。若提供 --pred_root，则同时把原始 predicted line 纳入拼图和评分。

输出：
  - scores.csv: 每帧评分与指标
  - topk.txt: top-K 帧路径
  - topk_sheets/: 每个候选帧的论文/调试拼图
  - heatmaps/: 所有或 top-K 帧的 center/soft/reliability/rgb/suppressed 热力图

典型用法：
  python select_3ch_line_cases.py \
    --line_debug_root ./soft_line/D_mask_30_40/line_debug \
    --pred_root ./geo_output/D_mask_30_40/line \
    --out_root ./soft_line/D_mask_30_40/selected_3ch_cases \
    --top_k 25 \
    --make_heatmaps --make_sheets
"""

import argparse
import csv
import math
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]
DEBUG_SUFFIXES = ["_center", "_soft", "_reliability", "_final", "_rgb_field"]


def natural_key(path):
    s = str(path).replace("\\", "/")
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def read_gray01(path, resize_to=None):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if resize_to is not None and img.shape[:2] != resize_to:
        h, w = resize_to
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def read_rgb(path, resize_to=None):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if resize_to is not None and img.shape[:2] != resize_to:
        h, w = resize_to
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_u8(x):
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def gray_to_bgr(x):
    return cv2.cvtColor(to_u8(x), cv2.COLOR_GRAY2BGR)


def heatmap_bgr(x, colormap=cv2.COLORMAP_TURBO, robust=False):
    x = np.asarray(x, dtype=np.float32)
    if robust:
        nz = x[x > 1e-6]
        if nz.size > 10:
            lo, hi = np.percentile(nz, 1), np.percentile(nz, 99)
        else:
            lo, hi = 0.0, 1.0
        x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    u8 = to_u8(x)
    return cv2.applyColorMap(u8, colormap)


def add_title_bgr(img, title, height=28):
    h, w = img.shape[:2]
    bar = np.full((height, w, 3), 255, dtype=np.uint8)
    cv2.putText(bar, title, (6, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def pad_to_same_height(images, pad_value=255):
    max_h = max(im.shape[0] for im in images)
    out = []
    for im in images:
        h, w = im.shape[:2]
        if h == max_h:
            out.append(im)
            continue
        pad = np.full((max_h - h, w, 3), pad_value, dtype=np.uint8)
        out.append(np.vstack([im, pad]))
    return out


def make_sheet(columns, gap=6):
    columns = pad_to_same_height(columns)
    h = columns[0].shape[0]
    gap_img = np.full((h, gap, 3), 255, dtype=np.uint8)
    out = []
    for i, im in enumerate(columns):
        if i > 0:
            out.append(gap_img)
        out.append(im)
    return np.hstack(out)


def strip_debug_suffix(stem):
    for suf in DEBUG_SUFFIXES:
        if stem.endswith(suf):
            return stem[:-len(suf)], suf[1:]
    return None, None


def collect_debug_groups(line_debug_root):
    root = Path(line_debug_root)
    groups = {}
    for p in root.rglob("*.png"):
        base, kind = strip_debug_suffix(p.stem)
        if base is None:
            continue
        rel_parent = p.parent.relative_to(root)
        key = (rel_parent, base)
        groups.setdefault(key, {})[kind] = p
    valid = []
    for (rel_parent, base), d in groups.items():
        if "soft" in d and "center" in d and "reliability" in d:
            valid.append({"rel_parent": rel_parent, "base": base, **d})
    valid.sort(key=lambda x: natural_key(Path(x["rel_parent"]) / x["base"]))
    return valid


def find_predicted(pred_root, rel_parent, base):
    if pred_root is None:
        return None
    pred_root = Path(pred_root)
    parent = pred_root / rel_parent
    # 先找同名常见后缀
    for ext in IMG_EXTS:
        p = parent / f"{base}{ext}"
        if p.is_file():
            return p
    # 再在 parent 下模糊找
    if parent.is_dir():
        candidates = []
        for ext in IMG_EXTS:
            candidates.extend(parent.glob(f"{base}*{ext}"))
        if candidates:
            return sorted(candidates, key=natural_key)[0]
    return None


def compute_metrics(soft, center, rel, pred=None, soft_thr=0.08, center_thr=0.50):
    eps = 1e-8
    soft_mask = soft > soft_thr
    center_mask = center > center_thr

    soft_sum = float(soft.sum()) + eps
    soft_area = float(soft_mask.mean())
    center_area = float(center_mask.mean())

    # soft 中没有进入 center 的候选结构比例：越高，越能说明 soft 与 center 有分工。
    extra_mask = soft_mask & (~center_mask)
    extra_soft_mass = float(soft[extra_mask].sum()) / soft_sum
    extra_soft_area_ratio = float(extra_mask.sum()) / (float(soft_mask.sum()) + eps)

    # reliability 对 soft 候选结构的降权强度：越高，越能说明 reliability 在起作用。
    lowrel_mass = float((soft * (1.0 - rel)).sum()) / soft_sum
    lowrel_extra_mass = float((soft * (1.0 - rel) * extra_mask.astype(np.float32)).sum()) / soft_sum

    if soft_mask.any():
        rel_vals = rel[soft_mask]
        rel_mean_on_soft = float(rel_vals.mean())
        rel_std_on_soft = float(rel_vals.std())
        rel_p10 = float(np.percentile(rel_vals, 10))
        rel_p90 = float(np.percentile(rel_vals, 90))
        rel_contrast = rel_p90 - rel_p10
        lowrel_pixel_ratio = float((rel_vals < 0.35).mean())
    else:
        rel_mean_on_soft = rel_std_on_soft = rel_p10 = rel_p90 = rel_contrast = lowrel_pixel_ratio = 0.0

    # trusted map 与 suppressed map 的可见差异。
    trusted = soft * rel
    suppressed = soft * (1.0 - rel)
    trusted_mass = float(trusted.sum()) / soft_sum
    suppressed_mass = float(suppressed.sum()) / soft_sum

    # predicted 与 soft 的差异。如果 predicted 不提供，这项为 0。
    if pred is not None:
        if pred.shape != soft.shape:
            pred = cv2.resize(pred, (soft.shape[1], soft.shape[0]), interpolation=cv2.INTER_AREA)
        pred_soft_l1 = float(np.mean(np.abs(pred - soft)))
        pred_soft_mass_l1 = float(np.sum(np.abs(pred - soft))) / (float(pred.sum()) + eps)
    else:
        pred_soft_l1 = 0.0
        pred_soft_mass_l1 = 0.0

    # 避免选出几乎空白帧或全屏噪声帧。
    density_bonus = math.exp(-abs(soft_area - 0.035) / 0.06)  # soft_area 在 3.5% 左右比较适合展示
    nonempty_bonus = 0.0 if soft_area < 0.001 else 1.0
    noisy_penalty = max(0.0, soft_area - 0.20) * 2.0

    # 评分重点：soft-center 分工 + reliability 降权 + reliability 对比度。
    score = (
        0.30 * extra_soft_mass +
        0.25 * lowrel_mass +
        0.20 * lowrel_extra_mass +
        0.15 * rel_contrast +
        0.07 * rel_std_on_soft +
        0.03 * min(pred_soft_mass_l1, 1.0)
    ) * density_bonus * nonempty_bonus - noisy_penalty

    return {
        "score": score,
        "soft_area": soft_area,
        "center_area": center_area,
        "extra_soft_mass": extra_soft_mass,
        "extra_soft_area_ratio": extra_soft_area_ratio,
        "lowrel_mass": lowrel_mass,
        "lowrel_extra_mass": lowrel_extra_mass,
        "lowrel_pixel_ratio": lowrel_pixel_ratio,
        "rel_mean_on_soft": rel_mean_on_soft,
        "rel_std_on_soft": rel_std_on_soft,
        "rel_p10": rel_p10,
        "rel_p90": rel_p90,
        "rel_contrast": rel_contrast,
        "trusted_mass": trusted_mass,
        "suppressed_mass": suppressed_mass,
        "pred_soft_l1": pred_soft_l1,
        "pred_soft_mass_l1": pred_soft_mass_l1,
    }


def save_case_outputs(case, soft, center, rel, pred, rgb, out_dirs, metrics, args, rank=None):
    rel_parent = case["rel_parent"]
    base = case["base"]
    rel_tag = rel_parent.as_posix().replace("/", "__") if str(rel_parent) != "." else "root"
    prefix = f"rank{rank:03d}_" if rank is not None else ""
    out_name = f"{prefix}{rel_tag}__{base}"

    suppressed = soft * (1.0 - rel)
    trusted = soft * rel

    # 单图热力图
    if args.make_heatmaps:
        hm_dir = out_dirs["heatmaps"] / rel_parent
        hm_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(hm_dir / f"{base}_center_heat.png"), heatmap_bgr(center, robust=False))
        cv2.imwrite(str(hm_dir / f"{base}_soft_heat.png"), heatmap_bgr(soft, robust=True))
        cv2.imwrite(str(hm_dir / f"{base}_reliability_heat.png"), heatmap_bgr(rel, robust=True))
        cv2.imwrite(str(hm_dir / f"{base}_trusted_soft_heat.png"), heatmap_bgr(trusted, robust=True))
        cv2.imwrite(str(hm_dir / f"{base}_suppressed_by_reliability_heat.png"), heatmap_bgr(suppressed, robust=True))

    # top-K 拼图
    if args.make_sheets and rank is not None:
        h, w = soft.shape
        cols = []
        if pred is not None:
            cols.append(add_title_bgr(gray_to_bgr(pred), "Predicted"))
        cols.append(add_title_bgr(gray_to_bgr(center), "Center (R)"))
        cols.append(add_title_bgr(heatmap_bgr(soft, robust=True), "Soft heat (G)"))
        cols.append(add_title_bgr(heatmap_bgr(rel, robust=True), "Reliability heat (B)"))
        cols.append(add_title_bgr(heatmap_bgr(suppressed, robust=True), "Suppressed: soft*(1-rel)"))
        cols.append(add_title_bgr(heatmap_bgr(trusted, robust=True), "Trusted: soft*rel"))
        if rgb is not None:
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cols.append(add_title_bgr(rgb_bgr, "RGB field"))

        sheet = make_sheet(cols, gap=6)
        # 底部指标条
        text = (f"rank={rank} score={metrics['score']:.4f}  "
                f"extra_soft={metrics['extra_soft_mass']:.3f}  "
                f"lowrel={metrics['lowrel_mass']:.3f}  "
                f"rel_contrast={metrics['rel_contrast']:.3f}  "
                f"soft_area={metrics['soft_area']:.3f}")
        bar = np.full((30, sheet.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)
        sheet = np.vstack([sheet, bar])
        out_dirs["sheets"].mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dirs["sheets"] / f"{out_name}_sheet.png"), sheet)

        # 同时复制原始 debug 图，便于手动打开看。
        raw_dir = out_dirs["raw_topk"] / f"{out_name}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for key in ["center", "soft", "reliability", "final", "rgb_field"]:
            p = case.get(key)
            if p and Path(p).is_file():
                shutil.copy2(p, raw_dir / Path(p).name)
        pred_path = case.get("pred_path")
        if pred_path and Path(pred_path).is_file():
            shutil.copy2(pred_path, raw_dir / f"{base}_predicted{Path(pred_path).suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--line_debug_root", required=True,
                        help="prepare_soft_lines.py --save_debug 生成的 line_debug 目录")
    parser.add_argument("--pred_root", default=None,
                        help="原始 predicted line 目录，例如 ./geo_output/.../line。可选，但强烈建议提供。")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--soft_thr", type=float, default=0.08)
    parser.add_argument("--center_thr", type=float, default=0.50)
    parser.add_argument("--make_heatmaps", action="store_true")
    parser.add_argument("--make_sheets", action="store_true")
    parser.add_argument("--all_heatmaps", action="store_true",
                        help="默认只给 top-K 生成热力图；启用后给所有帧生成热力图，可能较慢、文件较多。")
    args = parser.parse_args()

    line_debug_root = Path(args.line_debug_root).resolve()
    out_root = Path(args.out_root).resolve()
    pred_root = Path(args.pred_root).resolve() if args.pred_root else None
    out_root.mkdir(parents=True, exist_ok=True)

    out_dirs = {
        "heatmaps": out_root / "heatmaps",
        "sheets": out_root / "topk_sheets",
        "raw_topk": out_root / "topk_raw_debug",
    }

    cases = collect_debug_groups(line_debug_root)
    if not cases:
        raise RuntimeError(f"No valid debug groups found under: {line_debug_root}")

    print(f"[INFO] Found valid debug frame groups: {len(cases)}")
    if pred_root:
        print(f"[INFO] Predicted line root: {pred_root}")

    rows = []
    cache = {}
    for idx, case in enumerate(cases):
        soft = read_gray01(case["soft"])
        center = read_gray01(case["center"], resize_to=soft.shape)
        rel = read_gray01(case["reliability"], resize_to=soft.shape)
        pred_path = find_predicted(pred_root, case["rel_parent"], case["base"]) if pred_root else None
        pred = read_gray01(pred_path, resize_to=soft.shape) if pred_path else None
        rgb = read_rgb(case["rgb_field"], resize_to=soft.shape) if "rgb_field" in case else None
        case["pred_path"] = pred_path

        metrics = compute_metrics(
            soft, center, rel, pred=pred,
            soft_thr=args.soft_thr,
            center_thr=args.center_thr,
        )
        rel_name = (case["rel_parent"] / case["base"]).as_posix()
        row = {
            "rel_name": rel_name,
            "center_path": str(case["center"]),
            "soft_path": str(case["soft"]),
            "reliability_path": str(case["reliability"]),
            "rgb_field_path": str(case.get("rgb_field", "")),
            "predicted_path": str(pred_path) if pred_path else "",
            **metrics,
        }
        rows.append(row)
        cache[rel_name] = (case, soft, center, rel, pred, rgb, metrics)

        if args.make_heatmaps and args.all_heatmaps:
            save_case_outputs(case, soft, center, rel, pred, rgb, out_dirs, metrics, args, rank=None)

        if (idx + 1) % 200 == 0:
            print(f"[INFO] Scored {idx + 1}/{len(cases)} frames")

    rows.sort(key=lambda r: r["score"], reverse=True)

    # 写 CSV
    csv_path = out_root / "scores.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # 写 top-K 文本与生成拼图/热力图
    top_k = min(args.top_k, len(rows))
    txt_path = out_root / "topk.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for rank, r in enumerate(rows[:top_k], 1):
            f.write(f"#{rank}\tscore={r['score']:.6f}\t{r['rel_name']}\n")
            f.write(f"  pred={r['predicted_path']}\n")
            f.write(f"  center={r['center_path']}\n")
            f.write(f"  soft={r['soft_path']}\n")
            f.write(f"  reliability={r['reliability_path']}\n")
            f.write(f"  rgb={r['rgb_field_path']}\n")

            case, soft, center, rel, pred, rgb, metrics = cache[r["rel_name"]]
            if args.make_heatmaps and not args.all_heatmaps:
                save_case_outputs(case, soft, center, rel, pred, rgb, out_dirs, metrics, args, rank=rank)
            elif args.make_sheets:
                save_case_outputs(case, soft, center, rel, pred, rgb, out_dirs, metrics, args, rank=rank)

    print("[DONE]")
    print(f"Scores CSV : {csv_path}")
    print(f"Top-K list : {txt_path}")
    if args.make_sheets:
        print(f"Top-K sheets: {out_dirs['sheets']}")
    if args.make_heatmaps:
        print(f"Heatmaps   : {out_dirs['heatmaps']}")
    print("\n[Top 10]")
    for rank, r in enumerate(rows[:min(10, len(rows))], 1):
        print(f"#{rank:02d} score={r['score']:.4f} rel={r['rel_name']} "
              f"extra_soft={r['extra_soft_mass']:.3f} lowrel={r['lowrel_mass']:.3f} "
              f"rel_contrast={r['rel_contrast']:.3f} soft_area={r['soft_area']:.3f}")


if __name__ == "__main__":
    main()
