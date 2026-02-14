#!/usr/bin/env python3
"""
format_rename.py

递归遍历 input_path 下指定格式的文件（--format，例如 png），
将其按规则重命名并放到 output_path 下（保留原相对目录结构）：

新文件名规则:
    <一级子目录>_<二级子目录>_<原图名>
如果没有二级子目录则为:
    <一级子目录>_<原图名>
如果文件直接位于 input_path 下，则一级子目录取 input_path 的目录名。

参数:
  --input_path  必需，输入根目录
  --output_path 必需，输出根目录
  --format      必需，格式，例如 png 或 .png（大小写不敏感）
  --move        可选，提供则移动文件（默认复制）

示例:
  python format_rename.py --input_path DL3DV --output_path out --format png --move
"""
import argparse
import os
import shutil
from pathlib import Path
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Find files by format under input_path and copy/move+rename them into output_path.")
    p.add_argument("--input_path", required=True, help="输入目录路径（会递归遍历）")
    p.add_argument("--output_path", required=True, help="输出目录路径（会创建必要的目录）")
    p.add_argument("--format", required=True, help="要检索的格式（例如 png 或 .png），大小写不敏感")
    p.add_argument("--move", action="store_true", help="如果提供则移动文件（默认复制）")
    return p.parse_args()

def norm_ext(fmt: str) -> str:
    fmt = fmt.strip().lower()
    if not fmt:
        return ""
    if not fmt.startswith("."):
        fmt = "." + fmt
    return fmt

def find_files_by_ext(root: Path, ext: str):
    # ext should be like ".png"
    for p in root.rglob("*"):
        if p.is_file():
            if p.suffix.lower() == ext:
                yield p

def make_newname_and_dest(file_path: Path, input_root: Path, output_root: Path):
    """
    返回 (dest_dir: Path, new_name: str)
    dest_dir 是 output_root 下应创建的目录（保留原相对目录结构）
    new_name 是按照规则生成的新文件名
    """
    rel = os.path.relpath(str(file_path), str(input_root))
    rel_parts = Path(rel).parts  # tuple of path parts
    if len(rel_parts) == 0:
        first = input_root.name
        second = None
        original = file_path.name
        dest_rel_dir = Path("")
    else:
        original = rel_parts[-1]
        if len(rel_parts) >= 2:
            first = rel_parts[0]
        else:
            first = input_root.name
        second = rel_parts[1] if len(rel_parts) >= 3 else None
        dest_rel_dir = Path(*rel_parts[:-1]) if len(rel_parts) > 1 else Path("")
    if second:
        new_name = f"{first}_{second}_{original}"
    else:
        new_name = f"{first}_{original}"
    dest_dir = output_root.joinpath(dest_rel_dir)
    return dest_dir, new_name

def safe_remove_if_exists(path: Path):
    try:
        if path.exists():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    except Exception as e:
        print(f"WARNING: 删除已存在目标失败 {path}: {e}", file=sys.stderr)

def main():
    args = parse_args()
    input_root = Path(args.input_path).resolve()
    output_root = Path(args.output_path).resolve()
    if not input_root.exists() or not input_root.is_dir():
        print(f"ERROR: input_path 不存在或不是目录: {input_root}", file=sys.stderr)
        sys.exit(2)
    output_root.mkdir(parents=True, exist_ok=True)

    ext = norm_ext(args.format)
    if not ext:
        print("ERROR: 无效的 --format 参数", file=sys.stderr)
        sys.exit(2)

    files = list(find_files_by_ext(input_root, ext))
    if not files:
        print(f"未找到任何 {ext} 文件于 {input_root}")
        return

    total = len(files)
    processed = 0
    mode = "MOVE" if args.move else "COPY"
    print(f"找到 {total} 个 {ext} 文件，模式: {mode}，开始处理到: {output_root}")

    for idx, f in enumerate(files, start=1):
        try:
            dest_dir, new_name = make_newname_and_dest(f, input_root, output_root)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / new_name

            # 如果目标已存在，先删除（实现覆盖行为）
            if dest_path.exists():
                safe_remove_if_exists(dest_path)

            if args.move:
                # 使用 shutil.move（会在不同文件系统之间处理）
                shutil.move(str(f), str(dest_path))
            else:
                # 使用 copy2 保留元数据
                shutil.copy2(str(f), str(dest_path))

            processed += 1
            print(f"[{idx}/{total}] {f} -> {dest_path}")
        except Exception as e:
            print(f"[{idx}/{total}] 处理文件失败: {f}. 错误: {e}", file=sys.stderr)

    print(f"完成: 共找到 {total} 个文件，成功处理 {processed} 个（模式: {mode}）。")

if __name__ == "__main__":
    main()
