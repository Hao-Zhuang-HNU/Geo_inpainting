#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import pickle
import argparse
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _shim_numpy_core_for_pickle():
    """
    兼容某些由 numpy 2.x 环境生成的 pickle（引用 numpy._core），
    在 numpy 1.x 环境反序列化会报 ModuleNotFoundError: numpy._core。
    """
    try:
        import numpy as np
        import numpy.core as ncore
        sys.modules.setdefault("numpy._core", ncore)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
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

    if not isinstance(lines, (list, tuple)):
        raise ValueError(f"[{pkl_path}] Unsupported 'lines' type: {type(lines)}")

    cleaned = []
    for ln in lines:
        if isinstance(ln, (list, tuple)) and len(ln) == 4:
            x1, y1, x2, y2 = ln
            cleaned.append((float(x1), float(y1), float(x2), float(y2)))

    return cleaned, scores


def save_wireframe(
    lines: List[Tuple[float, float, float, float]],
    output_path: str,
    size: int = 256,
    line_width: float = 1.0,
    invert_y: bool = False,
):
    W = H = int(size)

    # 判断是否归一化坐标（0~1）：最大值 <= 1.5 视为归一化
    max_v = 0.0
    for x1, y1, x2, y2 in lines:
        max_v = max(max_v, abs(x1), abs(y1), abs(x2), abs(y2))
    normalized = (max_v <= 1.5)

    # 精确输出 size x size：dpi=size, figsize=(1,1)
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

        # 按你要求交换 x/y：plot([y1,y2], [x1,x2])
        ax.plot([y1p, y2p], [x1p, x2p], color="white", linewidth=line_width, solid_capstyle="round")

    if invert_y:
        ax.invert_yaxis()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="black")
    plt.close(fig)


def resolve_inputs(pkl_args: List[str], recursive: bool = False) -> List[str]:
    """
    支持：
    - 目录：收集目录下的 *.pkl（可递归）
    - glob：/path/*.pkl
    - 单文件：/path/xxx.pkl
    - 多个输入：--pkl_path a.pkl b.pkl /dir "*.pkl"
    """
    results = []

    for item in pkl_args:
        # 目录
        if os.path.isdir(item):
            pattern = "**/*.pkl" if recursive else "*.pkl"
            results.extend(glob.glob(os.path.join(item, pattern), recursive=recursive))
            continue

        # glob 模式或文件
        matches = glob.glob(item, recursive=True)
        if matches:
            results.extend(matches)
        elif os.path.isfile(item):
            results.append(item)

    # 去重 + 排序（稳定可复现）
    results = sorted(set(os.path.abspath(p) for p in results))
    return results


def build_output_path(
    in_path: str,
    out_dir: Optional[str],
    out_pattern: Optional[str],
    ext: str = ".png",
) -> str:
    base = os.path.splitext(os.path.basename(in_path))[0]

    # 自定义 pattern：支持 {stem} {name} {basename}
    # - stem: 不含后缀
    # - basename: 含后缀
    # - name: 等同 stem
    if out_pattern:
        basename = os.path.basename(in_path)
        rel = out_pattern.format(stem=base, name=base, basename=basename)
        return os.path.abspath(rel)

    if not out_dir:
        # 默认同目录输出
        out_dir = os.path.dirname(in_path) or "."
    return os.path.abspath(os.path.join(out_dir, base + ext))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl_path",
        type=str,
        nargs="+",
        required=True,
        help="批量输入：支持目录 / glob / 多文件。例：--pkl_path /data/pkls '*.pkl' a.pkl",
    )
    parser.add_argument(
        "--output_path","-o",
        type=str,
        default=None,
        help="批量输出目录（推荐）：如 /data/vis，会生成同名 png；若不填则输出到输入同目录",
    )
    parser.add_argument(
        "--output_path_pattern",
        type=str,
        default=None,
        help="可选：自定义输出路径模板，如 '/data/vis/{stem}_wire.png' 或 './out/{basename}.png'",
    )
    parser.add_argument("--size", type=int, default=256, help="输出分辨率，默认 256")
    parser.add_argument("--line_width", type=float, default=1.0, help="线宽，默认 1.0")
    parser.add_argument("--invert_y", action="store_true", help="可选：反转 y 轴（更像图像坐标：y向下）")
    parser.add_argument("--recursive", action="store_true", help="若输入为目录则递归搜索 *.pkl")
    args = parser.parse_args()

    in_files = resolve_inputs(args.pkl_path, recursive=args.recursive)
    if not in_files:
        raise RuntimeError("未找到任何 pkl 文件。请检查 --pkl_path 参数（目录 / glob / 文件路径）。")

    out_dir = args.output_path
    if out_dir is not None and args.output_path_pattern is None:
        os.makedirs(out_dir, exist_ok=True)

    ok, failed = 0, 0
    for pkl_file in in_files:
        try:
            lines, _ = load_lines_from_pkl(pkl_file)
            if not lines:
                raise RuntimeError("pkl 中未找到有效 lines（期望每条为 [x1,y1,x2,y2]）")

            out_path = build_output_path(
                in_path=pkl_file,
                out_dir=out_dir,
                out_pattern=args.output_path_pattern,
                ext=".png",
            )
            save_wireframe(
                lines=lines,
                output_path=out_path,
                size=args.size,
                line_width=args.line_width,
                invert_y=args.invert_y,
            )
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {pkl_file} -> {e}", file=sys.stderr)

    print(f"[DONE] total={len(in_files)}, ok={ok}, failed={failed}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
