#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two per-frame evaluation files and export frames with largest metric gaps.

By default, rows from the two inputs are aligned by mask_name, because different
methods may produce different pred_name values while using the same mask.

Supported input formats:
  - .csv: normal table with header
  - .json: list[dict], or dict containing a list under common keys such as
           records/data/frames/results/items, or dict[pred_name] -> record

Example:
  # Default: match by mask_name
  python compare_metric_diff.py \
    --input_1 ours/debug_alignment.csv \
    --input_2 propainter/debug_alignment.csv \
    -o diff_top25.csv

  # Old behavior: match by pred_name
  python compare_metric_diff.py \
    --input_1 ours/debug_alignment.csv \
    --input_2 propainter/debug_alignment.csv \
    -o diff_top25.csv \
    --match_by pred_name
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Tuple

DEFAULT_METRICS = ["PSNR_hole", "SSIM_hole"]
DEFAULT_MATCH_BY = "mask_name"
COMMON_JSON_LIST_KEYS = ["records", "data", "frames", "results", "items", "metrics", "per_frame"]


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
    """Convert common JSON metric layouts into list[dict]."""
    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict):
        records = None
        for key in COMMON_JSON_LIST_KEYS:
            if key in obj and isinstance(obj[key], list):
                records = obj[key]
                break

        # Support {"0000.png": {"PSNR_hole": ..., ...}, ...}
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
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return math.nan
    return float(s)


def build_index(rows: List[Dict[str, Any]], key: str, name: str) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    dup_count = 0
    missing_count = 0

    for row in rows:
        if key not in row:
            missing_count += 1
            continue

        k = str(row.get(key, "")).strip()
        if k == "":
            missing_count += 1
            continue

        if k in index:
            dup_count += 1
            # Keep the first row to avoid accidental many-to-many explosion.
            # If you compare multiple sequences and mask_name repeats, use --key with a more unique column.
            continue
        index[k] = row

    if missing_count == len(rows):
        raise KeyError(f"Column/key '{key}' not found or empty in all rows of {name}.")

    if missing_count > 0:
        print(
            f"[WARN] {name}: skipped {missing_count} rows with missing/empty '{key}'.",
            file=sys.stderr,
        )

    if dup_count > 0:
        print(
            f"[WARN] {name}: found {dup_count} duplicate '{key}' values; kept the first occurrence. "
            f"If this is a multi-sequence file, use --key with a more unique column.",
            file=sys.stderr,
        )
    return index


def pick_context_columns(row1: Dict[str, Any], row2: Dict[str, Any], base_cols: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in base_cols:
        if c in row1:
            out[f"{c}_1"] = row1.get(c, "")
        if c in row2:
            out[f"{c}_2"] = row2.get(c, "")
    return out


def metric_value_columns(
    row1: Dict[str, Any],
    row2: Dict[str, Any],
    metrics: List[str],
) -> Dict[str, Any]:
    """Always include all compared metric values, not only the ranking metric."""
    out: Dict[str, Any] = {}
    for m in metrics:
        v1 = to_float(row1.get(m)) if m in row1 else math.nan
        v2 = to_float(row2.get(m)) if m in row2 else math.nan
        out[f"{m}_1"] = v1
        out[f"{m}_2"] = v2
        out[f"{m}_signed_diff_1_minus_2"] = "" if math.isnan(v1) or math.isnan(v2) else v1 - v2
        out[f"{m}_abs_diff"] = "" if math.isnan(v1) or math.isnan(v2) else abs(v1 - v2)
    return out


def compare(
    rows1: List[Dict[str, Any]],
    rows2: List[Dict[str, Any]],
    key: str,
    metrics: List[str],
    topk: int,
    context_cols: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    idx1 = build_index(rows1, key, "input_1")
    idx2 = build_index(rows2, key, "input_2")
    common_keys = sorted(set(idx1.keys()) & set(idx2.keys()))

    if not common_keys:
        raise ValueError(f"No common rows found by key '{key}'.")

    output_rows: List[Dict[str, Any]] = []
    skipped = 0

    for metric in metrics:
        candidates: List[Dict[str, Any]] = []
        for k in common_keys:
            r1, r2 = idx1[k], idx2[k]
            if metric not in r1:
                raise KeyError(f"Metric '{metric}' not found in input_1.")
            if metric not in r2:
                raise KeyError(f"Metric '{metric}' not found in input_2.")

            v1, v2 = to_float(r1.get(metric)), to_float(r2.get(metric))
            if math.isnan(v1) or math.isnan(v2):
                skipped += 1
                continue

            signed_diff = v1 - v2
            abs_diff = abs(signed_diff)
            if signed_diff > 0:
                better = "input_1"
            elif signed_diff < 0:
                better = "input_2"
            else:
                better = "tie"

            row = {
                "rank_by_metric": metric,
                "rank": 0,  # filled after sorting
                key: k,
                "rank_metric_value_1": v1,
                "rank_metric_value_2": v2,
                "rank_metric_signed_diff_1_minus_2": signed_diff,
                "rank_metric_abs_diff": abs_diff,
                "better_higher_is_better": better,
            }
            row.update(metric_value_columns(r1, r2, metrics))
            row.update(pick_context_columns(r1, r2, context_cols))
            candidates.append(row)

        candidates.sort(key=lambda x: x["rank_metric_abs_diff"], reverse=True)
        for rank, row in enumerate(candidates[:topk], start=1):
            row["rank"] = rank
            output_rows.append(row)

    stats = {
        "alignment_key": key,
        "input_1_rows": len(rows1),
        "input_2_rows": len(rows2),
        "matched_rows": len(common_keys),
        "unmatched_input_1": len(set(idx1.keys()) - set(idx2.keys())),
        "unmatched_input_2": len(set(idx2.keys()) - set(idx1.keys())),
        "skipped_nan_metric_values": skipped,
    }
    return output_rows, stats


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")

    # Stable field order: first keys from the first row, then any later extras.
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find top-K frames with the largest PSNR_hole/SSIM_hole differences between two methods."
    )
    parser.add_argument("--input_1", required=True, help="First metric file, .csv or .json")
    parser.add_argument("--input_2", required=True, help="Second metric file, .csv or .json")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--match_by",
        choices=["mask_name", "pred_name"],
        default=DEFAULT_MATCH_BY,
        help="Common alignment mode. Default: mask_name. Use pred_name to reproduce the old behavior.",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Custom key column used for alignment. Overrides --match_by when provided.",
    )
    parser.add_argument("--topk", type=int, default=25, help="Top K per metric. Default: 25")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to compare. Default: PSNR_hole SSIM_hole",
    )
    parser.add_argument(
        "--context_cols",
        nargs="*",
        default=[
            "eval_index",
            "pred_path",
            "gt_path",
            "mask_path",
            "gt_name",
            "mask_name",
            "mask_ratio",
            "hole_pixels",
            "PSNR_all",
            "SSIM_all",
            "LPIPS",
        ],
        help="Extra columns copied from both inputs when available.",
    )
    args = parser.parse_args()
    if args.key is None:
        args.key = args.match_by
    return args


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be positive.")

    rows1 = read_table(args.input_1)
    rows2 = read_table(args.input_2)

    out_rows, stats = compare(
        rows1=rows1,
        rows2=rows2,
        key=args.key,
        metrics=args.metrics,
        topk=args.topk,
        context_cols=args.context_cols,
    )
    write_csv(args.output, out_rows)

    print(f"[OK] Saved: {args.output}")
    print(
        "[INFO] "
        + ", ".join(f"{k}={v}" for k, v in stats.items())
    )


if __name__ == "__main__":
    main()
