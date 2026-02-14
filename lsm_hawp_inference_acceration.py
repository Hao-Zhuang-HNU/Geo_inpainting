from src.lsm_hawp.lsm_hawp_model import LSM_HAWP
import src.lsm_hawp.lsm_hawp_model as lsm_hawp_module 

import torch
import os
import argparse
from pathlib import Path
import hashlib
import shutil
import multiprocessing as mp
import cv2
import time
from tqdm import tqdm 
from PIL import ImageFile

# --- 修复 1: 允许加载截断的图片 (针对轻微损坏) ---
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='HAWP Testing (Unified Progress Bar)')
parser.add_argument("--ckpt_path", type=str, default="ckpt/best_lsm_hawp.pth", help='ckpt path of HAWP')
parser.add_argument("-i", "--input_path", type=str, required=True, help='input image root')
parser.add_argument("-o", "--output_path", type=str, required=True, help='output pkl dir')
parser.add_argument("--gpu_ids", type=str, default='0')
parser.add_argument("--num_workers", type=int, default=16, help="parallel workers") 
parser.add_argument("--max_images", type=int, default=0, help="limit total images; 0=all")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
EXTS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp'}

class SilentTqdm:
    """静默进度条，用于屏蔽子进程输出"""
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable else []
    def __iter__(self):
        return iter(self.iterable)
    def update(self, *args, **kwargs): pass
    def close(self): pass
    def write(self, *args, **kwargs): pass

def stage_images(in_root: Path, out_root: Path, mapping: dict):
    stage_dir = out_root / '__stage_images__'
    tmp_root  = out_root / '__tmp_pkls__'
    stage_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    print(f'[INFO] Scanning images recursively in {in_root} ...')
    all_candidates = [p for p in in_root.rglob('*') if p.is_file() and p.suffix in EXTS]
    
    if args.max_images and args.max_images > 0:
        all_candidates = all_candidates[:args.max_images]

    staged_paths = []
    skipped_count = 0

    def _safe_stage_name(p: Path) -> str:
        rel = p.relative_to(in_root).as_posix()
        h = hashlib.sha1(rel.encode('utf-8')).hexdigest()[:10]
        return f'{h}_{p.stem}{p.suffix.lower()}'

    print('[INFO] Preparing stage environment...')
    for img in tqdm(all_candidates, desc="Staging", unit="img"):
        relative_dir = img.parent.relative_to(in_root)
        final_pkl_path = out_root / relative_dir / f'{img.stem}.pkl'

        if final_pkl_path.exists():
            skipped_count += 1
            continue

        staged_name = _safe_stage_name(img)
        dst = stage_dir / staged_name
        mapping[staged_name] = img
        
        if not dst.exists():
            try:
                os.symlink(img.as_posix(), dst.as_posix())
            except OSError:
                shutil.copy2(img.as_posix(), dst.as_posix())
        
        staged_paths.append(str(dst))

    if skipped_count > 0:
        print(f'[INFO] Skipped {skipped_count} images (already done).')
        
    return stage_dir, tmp_root, staged_paths

def shard(lst, n):
    n = max(1, n)
    L = len(lst)
    size = (L + n - 1) // n
    return [lst[i*size:(i+1)*size] for i in range(n)]

def worker_run(staged_list, tmp_pkl_dir, ckpt_path, queue):
    if not staged_list:
        return

    # Monkey patch tqdm
    lsm_hawp_module.tqdm = SilentTqdm 

    torch.set_num_threads(1)
    cv2.setNumThreads(1) 
    torch.backends.cudnn.benchmark = True
    
    try:
        model = LSM_HAWP(threshold=0.8, size=512)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.lsm_hawp.load_state_dict(state['model'])
        model.lsm_hawp.cuda()
        model.lsm_hawp.eval()
    except Exception as e:
        import traceback
        print(f'[ERROR] Worker setup failed: {e}')
        traceback.print_exc()
        return

    BATCH_SIZE = 16
    
    with torch.no_grad():
        total = len(staged_list)
        for i in range(0, total, BATCH_SIZE):
            batch = staged_list[i : i + BATCH_SIZE]
            
            # --- 修复 2: 鲁棒的批处理逻辑 ---
            try:
                # 尝试正常处理一个 Batch
                model.wireframe_detect(batch, str(tmp_pkl_dir))
                queue.put(len(batch))
            
            except Exception as e:
                # 如果 Batch 中有任何一张图坏了，就会进入这里
                # 我们切换到“逐张处理模式”，找出坏图并跳过
                print(f"[WARN] Batch failed due to corruption. Switching to safe mode for this batch. Error: {e}")
                
                success_count_in_batch = 0
                for img_path in batch:
                    try:
                        # 传入单个文件列表
                        model.wireframe_detect([img_path], str(tmp_pkl_dir))
                        success_count_in_batch += 1
                    except Exception as inner_e:
                        # 打印具体的坏文件路径，方便你后续删除
                        print(f"\n[CORRUPT FILE SKIPPED] >>> {img_path}")
                        print(f"[Error Info] {inner_e}\n")
                
                # 只汇报成功的数量，保证进度条准确
                if success_count_in_batch > 0:
                    queue.put(success_count_in_batch)

def move_and_rename(stage_dir: Path, tmp_root: Path, mapping: dict, in_root: Path, out_root: Path):
    moved = 0
    tmp_lookup = {}
    
    for d in tmp_root.iterdir():
        if d.is_dir():
            for f in d.glob('*.pkl'):
                tmp_lookup[f.name] = f

    print('[INFO] Moving files to final structure...')
    for staged_name, orig_img_path in tqdm(mapping.items(), desc="Finalizing", unit="file"):
        staged_stem = Path(staged_name).stem
        pkl_name = f'{staged_stem}.pkl'
        
        if pkl_name not in tmp_lookup:
            continue
            
        src_pkl = tmp_lookup[pkl_name]
        relative_dir = orig_img_path.parent.relative_to(in_root)
        dst_dir = out_root / relative_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_pkl = dst_dir / f'{orig_img_path.stem}.pkl'

        shutil.move(src_pkl.as_posix(), dst_pkl.as_posix())
        moved += 1
    print(f'[INFO] Done. {moved} files processed.')

def main():
    mp.set_start_method('spawn', force=True)
    
    in_root  = Path(args.input_path).resolve()
    out_root = Path(args.output_path).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    mapping = {}
    stage_dir, tmp_root, staged_list = stage_images(in_root, out_root, mapping)
    total_imgs = len(staged_list)
    
    if total_imgs == 0:
        print("All caught up. No images to process.")
        return

    n_workers = args.num_workers
    if total_imgs < n_workers:
        n_workers = total_imgs
        
    shards = shard(staged_list, n_workers)
    
    tmp_dirs = []
    for i in range(n_workers):
        d = tmp_root / f'worker_{i}'
        d.mkdir(parents=True, exist_ok=True)
        tmp_dirs.append(d)

    manager = mp.Manager()
    queue = manager.Queue()

    print(f'[INFO] Launching {n_workers} workers on RTX 3090...')
    
    procs = []
    for i in range(n_workers):
        if not shards[i]: continue
        p = mp.Process(target=worker_run, args=(shards[i], tmp_dirs[i], args.ckpt_path, queue))
        p.start()
        procs.append(p)

    pbar = tqdm(total=total_imgs, desc="Processing", unit="img", dynamic_ncols=True)
    
    processed_count = 0
    while processed_count < total_imgs:
        alive_procs = [p for p in procs if p.is_alive()]
        
        # 如果所有进程都死光了，且任务没完成，才退出
        if not alive_procs and queue.empty():
            print("\n[WARNING] All workers died or finished. Stopping monitor.")
            break
            
        try:
            increment = queue.get(timeout=0.5)
            processed_count += increment
            pbar.update(increment)
        except:
            continue
            
    pbar.close()

    for p in procs:
        p.join()

    move_and_rename(stage_dir, tmp_root, mapping, in_root, out_root)

    try:
        shutil.rmtree(stage_dir)
        shutil.rmtree(tmp_root)
    except Exception:
        pass

if __name__ == '__main__':
    main()