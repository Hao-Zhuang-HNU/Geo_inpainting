#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find Top-K frames with the largest structural / line gap in the hole region.

This script is designed for eval_video_line.py outputs, especially:
  - debug_alignment.csv / debug_alignment.json
  - per_frame_metrics.csv / per_frame_metrics.json

Two supported modes:

1) Single-file mode: find the worst Top-K frames inside one method.
   Default ranking: Line_Chamfer_hole descending, because larger Chamfer means larger line-position gap.

   python find_top_line_gap.py \
     --input debug_alignment.csv \
     -o top25_line_gap.csv

2) Two-file comparison mode: find Top-K frames where two methods differ most.
   Ranking: absolute metric difference between input_1 and input_2.

   python find_top_line_gap.py \
     --input_1 ours/debug_alignment.csv \
     --input_2 propainter/debug_alignment.csv \
     -o top25_line_diff.csv

Recommended metric:
  - Line_Chamfer_hole: best for line-structure discrepancy; lower is better.
  - Line_F1_hole: useful as auxiliary; higher is better.
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_METRIC = "Line_Chamfer_hole"
COMMON_JSON_LIST_KEYS = ["records", "data", "frames", "results", "items", "metrics", "per_frame"]

METRIC_DIRECTIONS = {
    "Line_F1_hole": "higher",
    "Line_Precision_hole": "higher",
    "Line_Recall_hole": "higher",
    "Line_Chamfer_hole": "lower",
    "Canny_F1_hole": "higher",
    "Canny_Precision_hole": "higher",
    "Canny_Recall_hole": "higher",
    "Gradient_L1_hole": "lower",
}

DEFAULT_CONTEXT_COLS = [
    "eval_index", "frame_original_index", "gt_original_index", "mask_original_index",
    "pred_name", "gt_name", "mask_name", "pred_path", "gt_path", "mask_path",
    "mask_ratio", "hole_pixels", "aligned_pair",
    "pred_line_pixels_hole", "gt_line_pixels_hole",
    "pred_canny_pixels_hole", "gt_canny_pixels_hole",
    "line_extractor", "f1_tolerance", "canny_low", "canny_high",
]

LINE_METRIC_COLS = [
    "Line_F1_hole", "Line_Precision_hole", "Line_Recall_hole", "Line_Chamfer_hole",
    "Canny_F1_hole", "Canny_Precision_hole", "Canny_Recall_hole", "Gradient_L1_hole",
]


def read_table(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return normalize_json_records(obj, path)

    raise ValueError(f"Unsupported input format: {path}. Only .csv and .json are supported.")


def normalize_json_records(obj: Any, path: str) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict):
        records = None
        for key in COMMON_JSON_LIST_KEYS:
            if key in obj and isinstance(obj[key], list):
                records = obj[key]
                break
        if records is None:
            if all(isinstance(v, dict) for v in obj.values()):
                records = []
                for k, v in obj.items():
                    row = dict(v)
                    row.setdefault("pred_name", k)
                    records.append(row)
            else:
                raise ValueError(
                    f"Cannot find a record list in JSON file: {path}. "
                    f"Expected list[dict], dict[pred_name]->dict, or dict with one of: {COMMON_JSON_LIST_KEYS}"
                )
    else:
        raise ValueError(f"Unsupported JSON root type in {path}: {type(obj).__name__}")

    if not all(isinstance(r, dict) for r in records):
        raise ValueError(f"JSON records in {path} must be dictionaries.")
    return [dict(r) for r in records]


def to_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "inf", "-inf"}:
        return math.nan
    try:
        return float(s)
    except Exception:
        return math.nan


def is_finite_number(value: Any) -> bool:
    v = to_float(value)
    return math.isfinite(v)


def build_index(rows: List[Dict[str, Any]], key: str, name: str) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    dup_count = 0
    for row in rows:
        if key not in row:
            raise KeyError(f"Column/key '{key}' not found in {name}.")
        k = str(row.get(key, ""))
        if k in index:
            dup_count += 1
            continue
        index[k] = row
    if dup_count > 0:
        print(
            f"[WARN] {name}: found {dup_count} duplicate '{key}' values; kept the first occurrence. "
            f"If this is a multi-sequence file, use a more unique --key, such as gt_name or aligned_pair.",
            file=sys.stderr,
        )
    return index


def metric_direction(metric: str, user_direction: str) -> str:
    if user_direction != "auto":
        return user_direction
    return METRIC_DIRECTIONS.get(metric, "lower" if "Chamfer" in metric or "L1" in metric else "higher")


def require_metric(rows: List[Dict[str, Any]], metric: str, name: str) -> None:
    if not rows:
        raise ValueError(f"{name} contains no rows.")
    available = set()
    for row in rows[:20]:
        available.update(row.keys())
    if metric not in available:
        line_like = [c for c in sorted(available) if "Line" in c or "Canny" in c or "Gradient" in c]
        raise KeyError(
            f"Metric '{metric}' not found in {name}.\n"
            f"Available line-like columns: {line_like}\n"
            f"Note: use eval_video_line.py output, not the old eval_video.py debug_alignment.csv."
        )


def copy_context(row: Dict[str, Any], cols: Iterable[str], suffix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in cols:
        if c in row:
            out[f"{c}{suffix}"] = row.get(c, "")
    return out


def copy_all_line_metrics(row: Dict[str, Any], suffix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in LINE_METRIC_COLS:
        if c in row:
            v = to_float(row.get(c))
            out[f"{c}{suffix}"] = "" if not math.isfinite(v) else v
    return out


def single_file_topk(
    rows: List[Dict[str, Any]],
    metric: str,
    topk: int,
    direction: str,
    context_cols: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    require_metric(rows, metric, "input")

    candidates: List[Dict[str, Any]] = []
    skipped = 0
    for row in rows:
        v = to_float(row.get(metric))
        if not math.isfinite(v):
            skipped += 1
            continue
        # Worst-frame ranking:
        #   higher-is-better metric -> smaller value is worse
        #   lower-is-better metric  -> larger value is worse
        score = -v if direction == "higher" else v
        out = {
            "rank": 0,
            "rank_metric": metric,
            "metric_direction": direction,
            "worst_score": score,
            "rank_metric_value": v,
        }
        out.update(copy_all_line_metrics(row))
        out.update(copy_context(row, context_cols))
        candidates.append(out)

    candidates.sort(key=lambda x: x["worst_score"], reverse=True)
    output = []
    for rank, row in enumerate(candidates[:topk], start=1):
        row["rank"] = rank
        output.append(row)

    stats = {
        "input_rows": len(rows),
        "valid_metric_rows": len(candidates),
        "skipped_nan_metric_values": skipped,
    }
    return output, stats


def two_file_topk(
    rows1: List[Dict[str, Any]],
    rows2: List[Dict[str, Any]],
    key: str,
    metric: str,
    topk: int,
    direction: str,
    context_cols: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    require_metric(rows1, metric, "input_1")
    require_metric(rows2, metric, "input_2")

    idx1 = build_index(rows1, key, "input_1")
    idx2 = build_index(rows2, key, "input_2")
    common_keys = sorted(set(idx1.keys()) & set(idx2.keys()))
    if not common_keys:
        raise ValueError(f"No common rows found by key '{key}'.")

    candidates: List[Dict[str, Any]] = []
    skipped = 0
    for k in common_keys:
        r1, r2 = idx1[k], idx2[k]
        v1, v2 = to_float(r1.get(metric)), to_float(r2.get(metric))
        if not math.isfinite(v1) or not math.isfinite(v2):
            skipped += 1
            continue

        signed_diff_1_minus_2 = v1 - v2
        abs_diff = abs(signed_diff_1_minus_2)

        if direction == "higher":
            if v1 > v2:
                better = "input_1"
            elif v2 > v1:
                better = "input_2"
            else:
                better = "tie"
        else:
            if v1 < v2:
                better = "input_1"
            elif v2 < v1:
                better = "input_2"
            else:
                better = "tie"

        row = {
            "rank": 0,
            "rank_metric": metric,
            "metric_direction": direction,
            key: k,
            "rank_metric_value_1": v1,
            "rank_metric_value_2": v2,
            "rank_metric_signed_diff_1_minus_2": signed_diff_1_minus_2,
            "rank_metric_abs_diff": abs_diff,
            "better": better,
        }
        row.update(copy_all_line_metrics(r1, "_1"))
        row.update(copy_all_line_metrics(r2, "_2"))
        row.update(copy_context(r1, context_cols, "_1"))
        row.update(copy_context(r2, context_cols, "_2"))
        candidates.append(row)

    candidates.sort(key=lambda x: x["rank_metric_abs_diff"], reverse=True)
    output = []
    for rank, row in enumerate(candidates[:topk], start=1):
        row["rank"] = rank
        output.append(row)

    stats = {
        "input_1_rows": len(rows1),
        "input_2_rows": len(rows2),
        "matched_rows": len(common_keys),
        "unmatched_input_1": len(set(idx1.keys()) - set(idx2.keys())),
        "unmatched_input_2": len(set(idx2.keys()) - set(idx1.keys())),
        "valid_metric_pairs": len(candidates),
        "skipped_nan_metric_values": skipped,
    }
    return output, stats


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, stats: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Top-K hole-region line-structure gap frames from eval_video_line.py debug/per-frame CSV or JSON."
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--input", help="Single eval_video_line.py debug_alignment/per_frame_metrics file, .csv or .json")
    parser.add_argument("--input_1", help="First method eval_video_line.py debug/per-frame file, .csv or .json")
    parser.add_argument("--input_2", help="Second method eval_video_line.py debug/per-frame file, .csv or .json")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    parser.add_argument("--topk", type=int, default=25, help="Top K frames. Default: 25")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        help="Ranking metric. Recommended default: Line_Chamfer_hole. Alternative: Line_F1_hole")
    parser.add_argument("--direction", choices=["auto", "higher", "lower"], default="auto",
                        help="Whether the metric is higher-better or lower-better. Default: auto")
    parser.add_argument("--key", default="pred_name",
                        help="Alignment key for two-file mode. Default: pred_name. For multi-sequence files, prefer gt_name or aligned_pair.")
    parser.add_argument("--context_cols", nargs="+", default=DEFAULT_CONTEXT_COLS,
                        help="Extra context columns to copy into output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be positive.")

    direction = metric_direction(args.metric, args.direction)

    if args.input:
        if args.input_1 or args.input_2:
            raise ValueError("Use either --input for single-file mode, or --input_1 and --input_2 for two-file mode, not both.")
        rows = read_table(args.input)
        output_rows, stats = single_file_topk(
            rows=rows,
            metric=args.metric,
            topk=args.topk,
            direction=direction,
            context_cols=args.context_cols,
        )
        stats.update({"mode": "single", "metric": args.metric, "metric_direction": direction, "topk": args.topk})
    else:
        if not args.input_1 or not args.input_2:
            raise ValueError("Two-file mode requires both --input_1 and --input_2. Single-file mode requires --input.")
        rows1 = read_table(args.input_1)
        rows2 = read_table(args.input_2)
        output_rows, stats = two_file_topk(
            rows1=rows1,
            rows2=rows2,
            key=args.key,
            metric=args.metric,
            topk=args.topk,
            direction=direction,
            context_cols=args.context_cols,
        )
        stats.update({"mode": "compare", "metric": args.metric, "metric_direction": direction, "topk": args.topk, "key": args.key})

    write_csv(args.output, output_rows)
    stats_path = os.path.splitext(args.output)[0] + ".summary.json"
    write_json(stats_path, stats)

    print("========== Top Line Gap ==========")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"Saved CSV: {args.output}")
    print(f"Saved summary: {stats_path}")


if __name__ == "__main__":
    main()
