import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import collections
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset filtering with chamfer-distance metric and blacklist")
    parser.add_argument('--img_list', type=str, required=True, help='Original image list')
    parser.add_argument('--pkl_list', type=str, required=True, help='Original pkl list')
    parser.add_argument('--npz_list', type=str, required=True, help='Original npz list')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory to save new lists and reject logs')

    parser.add_argument('--cham_th', type=float, default=4.0,
                        help='Threshold for sequence mean chamfer distance (lower is better)')
    parser.add_argument('--line_th', type=float, default=40.0, help='Threshold for Sequence Mean Line Count')
    parser.add_argument('--rm_list', type=str, default=None,
                        help='Path to a blacklist txt (optional). Sequences in this list will be removed.')
    parser.add_argument('--num_workers', type=int, default=16, help='Multiprocessing workers')
    return parser.parse_args()


def get_seq_id(path):
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
    """Worker: 读取单个样本的 Chamfer Distance 和线条数量"""
    idx, npz_path, pkl_path = args

    cham_dist = 1e6
    line_count = 0

    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            if 'endpoint_err' in data:
                cham_dist = float(data['endpoint_err'])
        except Exception:
            pass

    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            lines = data.get('lines', data.get('lines_pred', [])) if isinstance(data, dict) else data
            if lines is not None:
                line_count = len(lines)
        except Exception:
            pass

    return idx, cham_dist, line_count


def main():
    args = parse_args()

    print("Reading lists...")
    with open(args.img_list) as f:
        imgs = [l.strip() for l in f.readlines()]
    with open(args.pkl_list) as f:
        pkls = [l.strip() for l in f.readlines()]
    with open(args.npz_list) as f:
        npzs = [l.strip() for l in f.readlines()]

    if not (len(imgs) == len(pkls) == len(npzs)):
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

    print(f"Analyzing data with {args.num_workers} workers...")
    tasks = [(i, npzs[i], pkls[i]) for i in range(len(imgs))]

    data_cache = {}
    with Pool(args.num_workers) as p:
        for res in tqdm(p.imap_unordered(process_one_item, tasks, chunksize=100), total=len(imgs)):
            idx, cham_dist, cnt = res
            data_cache[idx] = (cham_dist, cnt)

    print("Filtering sequences...")
    keep_indices = []
    rejected_info = []

    count_high_cham = 0
    count_low_lines = 0
    count_blacklist = 0

    sorted_sids = sorted(seq_map.keys())

    for sid in sorted_sids:
        indices = seq_map[sid]
        seq_dir = seq_path_map[sid]
        norm_seq_dir = os.path.normpath(seq_dir)

        is_blacklisted = False
        if norm_seq_dir in blacklist_set:
            is_blacklisted = True
        else:
            for bad_path in blacklist_set:
                if bad_path.endswith(sid) or (sid in bad_path):
                    if os.path.basename(bad_path) == sid:
                        is_blacklisted = True
                        break

        if is_blacklisted:
            rejected_info.append(f"{seq_dir} | Manual Blacklist (--rm_list)")
            count_blacklist += 1
            continue

        seq_data = [data_cache[i] for i in indices]
        cham_dists = [d[0] for d in seq_data]
        lines = [d[1] for d in seq_data]

        valid_cham = cham_dists[1:] if len(cham_dists) > 1 else cham_dists
        mean_cham = sum(valid_cham) / len(valid_cham) if valid_cham else 1e6
        mean_lines = sum(lines) / len(lines) if lines else 0.0

        if mean_lines < args.line_th:
            rejected_info.append(f"{seq_dir} | Too few lines (Avg {mean_lines:.1f} < {args.line_th})")
            count_low_lines += 1
            continue

        if mean_cham > args.cham_th:
            rejected_info.append(f"{seq_dir} | High Chamfer Distance (Avg {mean_cham:.4f} > {args.cham_th})")
            count_high_cham += 1
            continue

        keep_indices.extend(indices)

    keep_indices.sort()
    os.makedirs(args.output_dir, exist_ok=True)

    reject_file = os.path.join(args.output_dir, "rejected_sequences.txt")
    with open(reject_file, 'w') as f:
        f.write("# Rejected Sequences Report\n")
        f.write(f"# Thresholds: Chamfer Distance > {args.cham_th}, Lines < {args.line_th}\n")
        if args.rm_list:
            f.write(f"# Blacklist used: {args.rm_list}\n")
        f.write(f"# Total Rejected: {len(rejected_info)}\n")
        f.write("\n".join(rejected_info))

    out_img = os.path.join(args.output_dir, "train_imgs_clean.txt")
    out_pkl = os.path.join(args.output_dir, "train_pkls_clean.txt")
    out_npz = os.path.join(args.output_dir, "train_npzs_clean.txt")

    print("Writing clean lists...")
    with open(out_img, 'w') as f:
        f.write('\n'.join([imgs[i] for i in keep_indices]))
    with open(out_pkl, 'w') as f:
        f.write('\n'.join([pkls[i] for i in keep_indices]))
    with open(out_npz, 'w') as f:
        f.write('\n'.join([npzs[i] for i in keep_indices]))

    total_seqs = len(seq_map)
    kept_seqs = total_seqs - len(rejected_info)

    print("\n" + "=" * 40)
    print("FILTERING SUMMARY")
    print("=" * 40)
    print(f"Total Sequences:  {total_seqs}")
    print(f"Kept Sequences:   {kept_seqs} ({kept_seqs / total_seqs:.1%})")
    print(f"Rejected Total:   {len(rejected_info)}")
    print(f"  - Blacklist:    {count_blacklist}")
    print(f"  - Low Lines:    {count_low_lines}")
    print(f"  - High Chamfer: {count_high_cham}")
    print("-" * 40)
    print(f"Rejected list saved to: {reject_file}")
    print(f"Clean lists saved to:   {args.output_dir}")


if __name__ == "__main__":
    main()
