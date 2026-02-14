import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

def get_seq_id(path_str):
    """
    从路径中提取序列ID。
    逻辑：从左往右找到第一个长度大于15的目录名。
    """
    try:
        # 统一路径分隔符
        p = Path(path_str)
        # parts包含了路径的各个部分 (e.g., '/', 'root', 'data', 'seq_hash', 'img.jpg')
        # 我们排除最后一个部分（文件名）
        for part in p.parts[:-1]:
            if len(part) > 15:
                return part
        
        # Fallback: 如果没找到，使用父目录名
        return p.parent.name
    except:
        return "unknown_seq"

def read_list(path):
    if not path or not os.path.exists(path):
        print(f"[Error] List file not found: {path}")
        sys.exit(1)
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Dataset Sampler & Copier")
    parser.add_argument("--img_list", type=str, required=True, help="Path to image list txt")
    parser.add_argument("--pkl_list", type=str, required=True, help="Path to pkl list txt")
    parser.add_argument("--npz_list", type=str, required=True, help="Path to npz list txt")
    parser.add_argument("--ratio", type=float, default=0.2, help="Sampling ratio (0.0 - 1.0)")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Destination directory to copy files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # 1. 设置随机种子
    random.seed(args.seed)
    
    # 2. 读取清单
    print(f"[1/5] Reading list files...")
    imgs = read_list(args.img_list)
    pkls = read_list(args.pkl_list)
    npzs = read_list(args.npz_list)
    
    if len(imgs) != len(pkls) or len(imgs) != len(npzs):
        print(f"[Error] List lengths mismatch! Img:{len(imgs)}, Pkl:{len(pkls)}, Npz:{len(npzs)}")
        sys.exit(1)
        
    print(f"      Total items: {len(imgs)}")

    # 3. 按序列归类
    print(f"[2/5] Grouping by sequence ID (folder name > 15 chars)...")
    seq_map = defaultdict(list)
    
    for i in tqdm(range(len(imgs)), desc="Grouping"):
        seq_id = get_seq_id(imgs[i])
        # 存储三元组 (img_path, pkl_path, npz_path)
        seq_map[seq_id].append((imgs[i], pkls[i], npzs[i]))
    
    all_seq_names = list(seq_map.keys())
    total_seqs = len(all_seq_names)
    print(f"      Found {total_seqs} unique sequences.")
    
    # 4. 随机采样
    target_count = int(total_seqs * args.ratio)
    target_count = max(1, target_count) # 至少选1个
    
    print(f"[3/5] Sampling {args.ratio * 100}% sequences...")
    print(f"      Target: {target_count} / {total_seqs} sequences.")
    
    selected_seq_names = random.sample(all_seq_names, target_count)
    
    # 5. 拷贝文件
    print(f"[4/5] Copying files to {args.output_dir} ...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 用于生成新的清单文件
    new_img_list = []
    new_pkl_list = []
    new_npz_list = []
    
    # 统计文件总数用于进度条
    total_files_to_copy = sum(len(seq_map[s]) for s in selected_seq_names)
    
    with tqdm(total=total_files_to_copy, desc="Copying") as pbar:
        for seq_id in selected_seq_names:
            items = seq_map[seq_id]
            
            # 在输出目录下为每个序列创建一个子文件夹，避免文件名冲突
            # 结构: output_dir / seq_id / ...
            seq_out_dir = os.path.join(args.output_dir, seq_id)
            os.makedirs(seq_out_dir, exist_ok=True)
            
            for (src_img, src_pkl, src_npz) in items:
                # 定义目标文件名 (保持原文件名)
                name_img = os.path.basename(src_img)
                name_pkl = os.path.basename(src_pkl)
                name_npz = os.path.basename(src_npz)
                
                dst_img = os.path.join(seq_out_dir, name_img)
                dst_pkl = os.path.join(seq_out_dir, name_pkl)
                dst_npz = os.path.join(seq_out_dir, name_npz)
                
                # 执行拷贝
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_pkl, dst_pkl)
                shutil.copy2(src_npz, dst_npz)
                
                # 记录新的绝对路径
                new_img_list.append(os.path.abspath(dst_img))
                new_pkl_list.append(os.path.abspath(dst_pkl))
                new_npz_list.append(os.path.abspath(dst_npz))
                
                pbar.update(1)

    # 6. 生成新清单
    print(f"[5/5] Generating new list files in output directory...")
    
    path_new_img_txt = os.path.join(args.output_dir, "sampled_img_list.txt")
    path_new_pkl_txt = os.path.join(args.output_dir, "sampled_pkl_list.txt")
    path_new_npz_txt = os.path.join(args.output_dir, "sampled_npz_list.txt")
    
    with open(path_new_img_txt, 'w') as f:
        f.write('\n'.join(new_img_list))
        
    with open(path_new_pkl_txt, 'w') as f:
        f.write('\n'.join(new_pkl_list))
        
    with open(path_new_npz_txt, 'w') as f:
        f.write('\n'.join(new_npz_list))
        
    print(f"\n[Done] Successfully sampled and copied.")
    print(f"      New lists generated at:")
    print(f"      - {path_new_img_txt}")
    print(f"      - {path_new_pkl_txt}")
    print(f"      - {path_new_npz_txt}")

if __name__ == "__main__":
    main()