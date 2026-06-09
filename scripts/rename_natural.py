import os
import argparse
import re
import uuid


def natural_sort_key(filename: str):
    """
    标准文件名自然升序排序 key：
    - 从左到右逐段比较；
    - 数字段按整数大小比较，且在同一位置数字排在字母前；
    - 非数字片段按小写字符串字母序比较；
    - 不额外按前缀/DSC/拍摄逻辑分组。

    例如：
    4b1cd91b_DSC05818 < 4b1cd91b_DSC05822 < d7d00859_DSC04676 < d7d00859_DSC04680
    """
    base = os.path.splitext(filename)[0]
    parts = re.split(r'(\d+)', base)

    key = []
    for part in parts:
        if part == '':
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))

    # 最后加原文件名兜底，保证排序稳定且可复现
    key.append((2, filename.lower()))
    return key


def main():
    parser = argparse.ArgumentParser(description='批量按文件名自然升序重命名文件')
    parser.add_argument('--Name', type=str, required=True, help='新文件基础名，如 img')
    parser.add_argument('--Path', type=str, required=True, help='目标文件夹路径')
    parser.add_argument('--start', type=int, default=0, help='起始编号，默认 0')
    parser.add_argument('--dry-run', action='store_true', help='只打印重命名结果，不实际修改文件')
    args = parser.parse_args()

    dir_path = args.Path
    base_name = args.Name

    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f'目标路径不是文件夹: {dir_path}')

    files = [
        f for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]
    files_sorted = sorted(files, key=natural_sort_key)

    if len(files_sorted) == 0:
        print('[INFO] 目录中没有可重命名的文件。')
        return

    max_index = args.start + len(files_sorted) - 1
    num_digits = len(str(max_index))

    rename_pairs = []
    for idx, old_name in enumerate(files_sorted, args.start):
        ext = os.path.splitext(old_name)[1]
        new_name = f'{base_name}{str(idx).zfill(num_digits)}{ext}'
        rename_pairs.append((old_name, new_name))

    for old_name, new_name in rename_pairs:
        print(f'{old_name} --> {new_name}')

    if args.dry_run:
        print('[INFO] dry-run 模式：未实际重命名。')
        return

    # 两阶段重命名，避免目标文件名与原文件名冲突。
    token = uuid.uuid4().hex[:8]
    temp_pairs = []

    for i, (old_name, new_name) in enumerate(rename_pairs):
        if old_name == new_name:
            continue
        old_path = os.path.join(dir_path, old_name)
        tmp_name = f'.__rename_tmp_{token}_{i}{os.path.splitext(old_name)[1]}'
        tmp_path = os.path.join(dir_path, tmp_name)
        os.rename(old_path, tmp_path)
        temp_pairs.append((tmp_name, new_name))

    for tmp_name, new_name in temp_pairs:
        tmp_path = os.path.join(dir_path, tmp_name)
        new_path = os.path.join(dir_path, new_name)
        if os.path.exists(new_path):
            raise FileExistsError(f'目标文件已存在，避免覆盖，已中止: {new_path}')
        os.rename(tmp_path, new_path)


if __name__ == '__main__':
    main()
