#!/usr/bin/env python3
"""
rename_by_parents_and_copy.py

用法:
  python rename_by_parents_and_copy.py --input_path /path/in --output_path /path/out --extension pkl

说明:
  - 递归查找指定后缀的文件（大小写不敏感）。
  - 新文件名格式: {祖父目录名}_{父目录名}_{原文件名}.后缀
    例如: /data/PersonA/run_01/frame001.jpg
    -> PersonA_run_01_frame001.jpg
  - 复制并保留相对于 input_path 的目录结构到 output_path。
"""
import argparse
from pathlib import Path
import shutil
import sys

def parse_args():
    p = argparse.ArgumentParser(description="递归复制文件，并根据上两级目录名重命名。")
    p.add_argument("--input_path", required=True, help="输入目录（递归查找）")
    p.add_argument("--output_path", required=True, help="输出目录（自动创建）")
    p.add_argument("--extension", required=True, help="目标文件后缀，例如: pkl, jpg, png (不需要带点)")
    return p.parse_args()

def make_new_name(file_path: Path) -> str:
    """
    根据文件的上级和上上级目录生成新文件名。
    格式: Grandparent_Parent_OriginalStem
    """
    # 获取文件名（不含后缀）
    orig_stem = file_path.stem
    
    # 获取父目录名 (Parent)
    parent_name = file_path.parent.name
    
    # 获取祖父目录名 (Grandparent)
    # resolve() 确保路径是绝对路径，避免 '..' 等问题
    # parent.parent.name 获取上上级
    grandparent_name = file_path.resolve().parent.parent.name
    
    # 拼接新文件名
    # 注意：如果目录结构很浅（例如文件就在根目录下），grandparent可能会是系统根目录或盘符
    new_stem = f"{grandparent_name}_{parent_name}_{orig_stem}"
    
    return new_stem

def main():
    args = parse_args()
    input_root = Path(args.input_path).resolve()
    output_root = Path(args.output_path).resolve()
    
    # 处理后缀输入，确保格式统一（去掉前面的点，转小写）
    target_ext = args.extension.lstrip(".").lower()
    if not target_ext:
        print("ERROR: 必须指定有效的文件后缀。", file=sys.stderr)
        sys.exit(1)

    if not input_root.exists() or not input_root.is_dir():
        print(f"ERROR: input_path 不存在或不是目录: {input_root}", file=sys.stderr)
        sys.exit(2)
    
    output_root.mkdir(parents=True, exist_ok=True)

    # 递归查找指定后缀的文件
    # rglob("*") 遍历所有文件，然后判断 suffix
    print(f"正在 {input_root} 下查找 .{target_ext} 文件...")
    files = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() == f".{target_ext}"]
    
    total = len(files)
    if total == 0:
        print(f"未在 {input_root} 下找到任何 .{target_ext} 文件。")
        return

    print(f"找到 {total} 个 .{target_ext} 文件，开始处理...")

    copied = 0
    for idx, p in enumerate(files, start=1):
        try:
            # 计算相对路径，用于在 output 中保持原有目录树结构
            # 如果不需要保持目录结构，只想把所有文件平铺，可以将 rel_parent 设为空或固定目录
            rel = p.relative_to(input_root)
            rel_parent = rel.parent
            
            dest_dir = output_root.joinpath(rel_parent)
            dest_dir.mkdir(parents=True, exist_ok=True)

            # 生成新文件名
            new_stem = make_new_name(p)
            new_filename = f"{new_stem}{p.suffix}" # 加上原后缀
            
            dest_path = dest_dir / new_filename

            # 若目标存在则覆盖
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except Exception:
                    pass

            shutil.copy2(str(p), str(dest_path))
            copied += 1
            print(f"[{idx}/{total}] {p.name} -> {new_filename}")
            
        except Exception as e:
            print(f"[{idx}/{total}] 处理失败: {p}，错误: {e}", file=sys.stderr)

    print(f"完成：共找到 {total} 个 .{target_ext}，成功复制并重命名 {copied} 个。")

if __name__ == "__main__":
    main()