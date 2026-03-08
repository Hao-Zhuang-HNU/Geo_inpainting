import argparse
import collections
import os
import pickle
import re
from multiprocessing import Pool

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Filtering by Line Count")
    parser.add_argument('--img_list', type=str, required=True, help='Original image list')
    parser.add_argument('--pkl_list', type=str, required=True, help='Original pkl list')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory to save new lists and reject logs')

    parser.add_argument('--line_th', type=float, default=40.0, help='Threshold for sequence mean line count')
    parser.add_argument('--rm_list', type=str, default=None, help='Path to blacklist txt (optional)')
    parser.add_argument('--num_workers', type=int, default=16, help='Multiprocessing workers')
    return parser.parse_args()


def get_seq_id(path):
    """
    从左到右查找目录名，命中 ._imgs/._pkls/._npzs/._flow（. 代表任意一个字符）后，
    继续在其下级目录中查找，返回第一个长度 > 5 的目录名作为序列ID。
    """
    parts = path.split(os.sep)
    marker_pattern = re.compile(r'^._(imgs|pkls|npzs|flow)$')
    for i, part in enumerate(parts):
        if marker_pattern.match(part):
            for child in parts[i + 1:]:
                if len(child) > 5:
                    return child
            break
    return os.path.basename(os.path.dirname(path))


def get_seq_path(path):
    """
    获取序列根目录路径（包含 marker 后第一个长度 > 5 的目录层），用于路径匹配。
    """
    parts = path.split(os.sep)
    marker_pattern = re.compile(r'^._(imgs|pkls|npzs|flow)$')
    for i, part in enumerate(parts):
        if marker_pattern.match(part):
            for j in range(i + 1, len(parts)):
                if len(parts[j]) > 5:
                    return os.sep.join(parts[:j + 1])
            break
    return os.path.dirname(path)


def load_blacklist(rm_list_path):
    """
    加载黑名单，支持两种格式：
    1. 纯路径: /path/to/seq
    2. 带原因: /path/to/seq | Reason...
    返回一个 set 包含标准化的路径。
    """
    if not rm_list_path or not os.path.exists(rm_list_path):
        return set()

    print(f"Loading blacklist from {rm_list_path}...")
    blacklist = set()
    with open(rm_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            path_part = line.split('|')[0].strip()
            blacklist.add(os.path.normpath(path_part))

    print(f"Loaded {len(blacklist)} blacklisted sequences.")
    return blacklist


def process_one_item(args):
    """Worker: 读取单个样本的线条数量"""
    idx, pkl_path = args
    line_count = 0

    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            lines = data.get('lines', data.get('lines_pred', [])) if isinstance(data, dict) else data
            if lines is not None:
                line_count = len(lines)
        except Exception:
            pass

    return idx, line_count


def is_blacklisted(norm_seq_dir, sid, blacklist_set):
    if norm_seq_dir in blacklist_set:
        return True

    for bad_path in blacklist_set:
        if bad_path.endswith(sid) or (sid in bad_path):
            if os.path.basename(bad_path) == sid:
                return True
    return False


def main():
    args = parse_args()

    print("Reading lists...")
    with open(args.img_list) as f:
        imgs = [l.strip() for l in f.readlines()]
    with open(args.pkl_list) as f:
        pkls = [l.strip() for l in f.readlines()]
    if not (len(imgs) == len(pkls)):
        print("Error: List lengths do not match!")
        return

    blacklist_set = load_blacklist(args.rm_list)

    print("Grouping sequences...")
    seq_map = collections.defaultdict(list)
    seq_path_map = {}

    for i, path in enumerate(imgs):
        sid = get_seq_id(path)
        seq_map[sid].append(i)
        if sid not in seq_path_map:
            seq_path_map[sid] = get_seq_path(path)

    print(f"Total Sequences: {len(seq_map)}")
    print(f"Total Images:    {len(imgs)}")

    print(f"Analyzing line counts with {args.num_workers} workers...")
    tasks = [(i, pkls[i]) for i in range(len(imgs))]

    data_cache = {}
    with Pool(args.num_workers) as p:
        for idx, cnt in tqdm(p.imap_unordered(process_one_item, tasks, chunksize=100), total=len(imgs)):
            data_cache[idx] = cnt

    print("Filtering sequences...")
    keep_indices = []
    rejected_sequences = []

    count_blacklist = 0
    count_low_lines = 0

    for sid in sorted(seq_map.keys()):
        indices = seq_map[sid]
        seq_dir = seq_path_map[sid]
        norm_seq_dir = os.path.normpath(seq_dir)

        if is_blacklisted(norm_seq_dir, sid, blacklist_set):
            rejected_sequences.append(seq_dir)
            count_blacklist += 1
            continue

        lines = [data_cache[i] for i in indices]
        mean_lines = sum(lines) / len(lines) if lines else 0.0

        if mean_lines < args.line_th:
            rejected_sequences.append(seq_dir)
            count_low_lines += 1
            continue

        keep_indices.extend(indices)

    keep_indices.sort()
    os.makedirs(args.output_dir, exist_ok=True)

    reject_file = os.path.join(args.output_dir, "rejected_sequences.txt")
    with open(reject_file, 'w') as f:
        f.write('\n'.join(rejected_sequences))

    out_img = os.path.join(args.output_dir, "train_imgs_clean.txt")
    out_pkl = os.path.join(args.output_dir, "train_pkls_clean.txt")
    print("Writing clean lists...")
    with open(out_img, 'w') as f:
        f.write('\n'.join([imgs[i] for i in keep_indices]))
    with open(out_pkl, 'w') as f:
        f.write('\n'.join([pkls[i] for i in keep_indices]))

    total_seqs = len(seq_map)
    kept_seqs = total_seqs - len(rejected_sequences)

    print("\n" + "=" * 40)
    print("FILTERING SUMMARY")
    print("=" * 40)
    print(f"Total Sequences:  {total_seqs}")
    print(f"Kept Sequences:   {kept_seqs} ({kept_seqs / total_seqs:.1%})")
    print(f"Rejected Total:   {len(rejected_sequences)}")
    print(f"  - Blacklist:    {count_blacklist}")
    print(f"  - Low Lines:    {count_low_lines}")
    print("-" * 40)
    print(f"Rejected list saved to: {reject_file}")
    print(f"Clean lists saved to:   {args.output_dir}")


if __name__ == "__main__":
    main()
