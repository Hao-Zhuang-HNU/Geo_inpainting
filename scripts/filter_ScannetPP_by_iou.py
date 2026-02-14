import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import collections
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Dataset Filtering with Blacklist")
    parser.add_argument('--img_list', type=str, required=True, help='Original image list')
    parser.add_argument('--pkl_list', type=str, required=True, help='Original pkl list')
    parser.add_argument('--npz_list', type=str, required=True, help='Original npz list')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory to save new lists and reject logs')
    
    # 阈值设置
    parser.add_argument('--iou_th', type=float, default=0.15, help='Threshold for Sequence Mean IoU')
    parser.add_argument('--line_th', type=float, default=15.0, help='Threshold for Sequence Mean Line Count')
    
    # 【新增】黑名单参数
    parser.add_argument('--rm_list', type=str, default=None, help='Path to a blacklist txt (optional). Sequences in this list will be removed.')
    
    parser.add_argument('--num_workers', type=int, default=22, help='Multiprocessing workers')
    return parser.parse_args()

def get_seq_id(path):
    """
    遍历路径分割后的每一部分，返回第一个长度超过15个字符的文件夹名作为序列ID。
    """
    parts = path.split(os.sep)
    for part in parts:
        if len(part) >= 15:
            return part
    return os.path.basename(os.path.dirname(path))

def get_seq_path(path):
    """
    获取序列的根目录路径（包含那个长哈希名的那一层），用于路径匹配
    """
    parts = path.split(os.sep)
    for i, part in enumerate(parts):
        if len(part) > 20:
            return os.sep.join(parts[:i+1])
    return os.path.dirname(path)

def load_blacklist(rm_list_path):
    """
    加载黑名单，支持两种格式：
    1. 纯路径: /path/to/seq
    2. 带原因: /path/to/seq | Reason...
    返回一个 set 包含标准化的路径
    """
    if not rm_list_path or not os.path.exists(rm_list_path):
        return set()
    
    print(f"Loading blacklist from {rm_list_path}...")
    blacklist = set()
    with open(rm_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            
            # 提取路径部分（分割 | 符号）
            path_part = line.split('|')[0].strip()
            
            # 标准化路径（去除末尾斜杠，统一分隔符），防止匹配失败
            norm_path = os.path.normpath(path_part)
            blacklist.add(norm_path)
            
    print(f"Loaded {len(blacklist)} blacklisted sequences.")
    return blacklist

def process_one_item(args):
    """Worker: 读取单个样本的 IoU 和 线条数量"""
    idx, npz_path, pkl_path = args
    
    iou = 0.0
    line_count = 0
    
    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            if 'confidence_iou' in data:
                iou = float(data['confidence_iou'])
        except: pass
            
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            lines = data.get('lines', data.get('lines_pred', [])) if isinstance(data, dict) else data
            if lines is not None:
                line_count = len(lines)
        except: pass
            
    return idx, iou, line_count

def main():
    args = parse_args()
    
    print(f"Reading lists...")
    with open(args.img_list) as f: imgs = [l.strip() for l in f.readlines()]
    with open(args.pkl_list) as f: pkls = [l.strip() for l in f.readlines()]
    with open(args.npz_list) as f: npzs = [l.strip() for l in f.readlines()]
    
    if not (len(imgs) == len(pkls) == len(npzs)):
        print("Error: List lengths do not match!")
        return

    # 【新增】加载黑名单
    blacklist_set = load_blacklist(args.rm_list)

    # 1. 按序列分组
    print("Grouping sequences...")
    seq_map = collections.defaultdict(list)
    seq_path_map = {} 
    
    for i, path in enumerate(imgs):
        sid = get_seq_id(path)
        seq_map[sid].append(i)
        if sid not in seq_path_map:
            # 记录用于匹配的路径
            seq_path_map[sid] = get_seq_path(path)
            
    print(f"Total Sequences: {len(seq_map)}")
    print(f"Total Images:    {len(imgs)}")

    # 2. 多进程读取数据
    print(f"Analyzing data with {args.num_workers} workers...")
    tasks = [(i, npzs[i], pkls[i]) for i in range(len(imgs))]
    
    data_cache = {}
    with Pool(args.num_workers) as p:
        for res in tqdm(p.imap_unordered(process_one_item, tasks, chunksize=100), total=len(imgs)):
            idx, iou, cnt = res
            data_cache[idx] = (iou, cnt)

    # 3. 序列级筛选
    print("Filtering sequences...")
    keep_indices = []
    rejected_info = [] 
    
    count_low_iou = 0
    count_low_lines = 0
    count_blacklist = 0 # 新增计数
    
    sorted_sids = sorted(seq_map.keys())

    for sid in sorted_sids:
        indices = seq_map[sid]
        seq_dir = seq_path_map[sid]
        
        # -------------------------------------------------
        # 【新增】优先级 0: 黑名单检查
        # -------------------------------------------------
        # 标准化当前序列路径以便比对
        norm_seq_dir = os.path.normpath(seq_dir)
        
        # 检查是否在黑名单中（精确匹配 或 作为子目录）
        # 考虑到路径写法可能不同，我们尝试只要 path 包含黑名单里的 key 就算
        is_blacklisted = False
        if norm_seq_dir in blacklist_set:
            is_blacklisted = True
        else:
            # 备用检查：防止绝对路径前缀不一致
            # 检查 seq_id 是否出现在黑名单的路径字符串中
            for bad_path in blacklist_set:
                if bad_path.endswith(sid) or (sid in bad_path):
                    # 这是一个比较宽松的匹配，确保能踢掉
                    if os.path.basename(bad_path) == sid: 
                        is_blacklisted = True
                        break
        
        if is_blacklisted:
            reason = "Manual Blacklist (--rm_list)"
            rejected_info.append(f"{seq_dir} | {reason}")
            count_blacklist += 1
            continue
        # -------------------------------------------------

        # 获取数据
        seq_data = [data_cache[i] for i in indices]
        ious = [d[0] for d in seq_data]
        lines = [d[1] for d in seq_data]
        
        valid_ious = ious[1:] if len(ious) > 1 else ious
        mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
        mean_lines = sum(lines) / len(lines) if lines else 0.0
        
        # 优先级 1: 没线条
        if mean_lines < args.line_th:
            reason = f"Too few lines (Avg {mean_lines:.1f} < {args.line_th})"
            rejected_info.append(f"{seq_dir} | {reason}")
            count_low_lines += 1
            continue
            
        # 优先级 2: 对不齐
        if mean_iou < args.iou_th:
            reason = f"Low IoU (Avg {mean_iou:.4f} < {args.iou_th})"
            rejected_info.append(f"{seq_dir} | {reason}")
            count_low_iou += 1
            continue
            
        keep_indices.extend(indices)

    # 4. 排序并保存
    keep_indices.sort()
    os.makedirs(args.output_dir, exist_ok=True)
    
    reject_file = os.path.join(args.output_dir, "rejected_sequences.txt")
    with open(reject_file, 'w') as f:
        f.write(f"# Rejected Sequences Report\n")
        f.write(f"# Thresholds: IoU < {args.iou_th}, Lines < {args.line_th}\n")
        if args.rm_list:
            f.write(f"# Blacklist used: {args.rm_list}\n")
        f.write(f"# Total Rejected: {len(rejected_info)}\n")
        f.write("\n".join(rejected_info))
        
    out_img = os.path.join(args.output_dir, "train_imgs_clean.txt")
    out_pkl = os.path.join(args.output_dir, "train_pkls_clean.txt")
    out_npz = os.path.join(args.output_dir, "train_npzs_clean.txt")
    
    print(f"Writing clean lists...")
    with open(out_img, 'w') as f: f.write('\n'.join([imgs[i] for i in keep_indices]))
    with open(out_pkl, 'w') as f: f.write('\n'.join([pkls[i] for i in keep_indices]))
    with open(out_npz, 'w') as f: f.write('\n'.join([npzs[i] for i in keep_indices]))
    
    total_seqs = len(seq_map)
    kept_seqs = total_seqs - len(rejected_info)
    
    print("\n" + "="*40)
    print(f"FILTERING SUMMARY")
    print("="*40)
    print(f"Total Sequences:  {total_seqs}")
    print(f"Kept Sequences:   {kept_seqs} ({kept_seqs/total_seqs:.1%})")
    print(f"Rejected Total:   {len(rejected_info)}")
    print(f"  - Blacklist:    {count_blacklist}")
    print(f"  - Low Lines:    {count_low_lines}")
    print(f"  - Low IoU:      {count_low_iou}")
    print("-" * 40)
    print(f"Rejected list saved to: {reject_file}")
    print(f"Clean lists saved to:   {args.output_dir}")

if __name__ == "__main__":
    main()