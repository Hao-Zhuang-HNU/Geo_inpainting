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
from tqdm import tqdm
from PIL import ImageFile

# 允许加载轻微损坏/截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='HAWP Testing: recursive images -> pkl, robust finalize')
parser.add_argument("--ckpt_path", type=str, default="ckpt/best_lsm_hawp.pth", help='ckpt path of HAWP')
parser.add_argument("-i", "--input_path", type=str, required=True, help='input image root')
parser.add_argument("-o", "--output_path", type=str, required=True, help='output pkl root')
parser.add_argument("--gpu_ids", type=str, default='0')
parser.add_argument("--num_workers", type=int, default=16, help="parallel workers")
parser.add_argument("--max_images", type=int, default=0, help="limit total images; 0=all")
parser.add_argument("--threshold", type=float, default=0.8, help="LSM_HAWP threshold")
parser.add_argument("--hawp_size", type=int, default=512, help="LSM_HAWP input size")
parser.add_argument("--batch_size", type=int, default=16, help="batch size per worker")
parser.add_argument("--keep_tmp", action="store_true", help="keep __stage_images__ and __tmp_pkls__ for debugging")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

# suffix 已统一 lower，所以这里只放小写
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


class SilentTqdm:
    """静默进度条，用于屏蔽子进程内部 tqdm 输出"""
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else []

    def __iter__(self):
        return iter(self.iterable)

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def write(self, *args, **kwargs):
        pass


def safe_unlink_or_rmtree(path: Path):
    if not path.exists():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def stage_images(in_root: Path, out_root: Path, mapping: dict):
    """
    将原始图片映射到一个扁平临时目录中，避免 HAWP 对深层目录/同名文件处理出错。
    mapping:
        staged_name -> original_image_path
    """
    stage_dir = out_root / '__stage_images__'
    tmp_root = out_root / '__tmp_pkls__'
    stage_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    print(f'[INFO] Scanning images recursively in {in_root} ...')
    all_candidates = [
        p for p in in_root.rglob('*')
        if p.is_file() and p.suffix.lower() in EXTS
    ]
    all_candidates = sorted(all_candidates)

    if args.max_images and args.max_images > 0:
        all_candidates = all_candidates[:args.max_images]

    staged_paths = []
    skipped_count = 0

    def _safe_stage_name(p: Path) -> str:
        rel = p.relative_to(in_root).as_posix()
        h = hashlib.sha1(rel.encode('utf-8')).hexdigest()[:10]
        return f'{h}_{p.stem}{p.suffix.lower()}'

    print('[INFO] Preparing stage environment...')
    for img in tqdm(all_candidates, desc="Staging", unit="img", dynamic_ncols=True):
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
        print(f'[INFO] Skipped {skipped_count} images because final pkl already exists.')

    return stage_dir, tmp_root, staged_paths


def shard(lst, n):
    n = max(1, n)
    L = len(lst)
    size = (L + n - 1) // n
    return [lst[i * size:(i + 1) * size] for i in range(n)]


def worker_run(staged_list, tmp_pkl_dir, ckpt_path, queue, worker_id):
    if not staged_list:
        return

    # Monkey patch：避免每个子进程都输出独立进度条
    lsm_hawp_module.tqdm = SilentTqdm

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True

    try:
        model = LSM_HAWP(threshold=args.threshold, size=args.hawp_size)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.lsm_hawp.load_state_dict(state['model'])
        model.lsm_hawp.cuda()
        model.lsm_hawp.eval()
    except Exception as e:
        import traceback
        print(f'[ERROR][worker {worker_id}] setup failed: {e}')
        traceback.print_exc()
        return

    batch_size = max(1, args.batch_size)

    with torch.no_grad():
        total = len(staged_list)
        for i in range(0, total, batch_size):
            batch = staged_list[i:i + batch_size]

            # 这里只表示 wireframe_detect 调用完成；真正生成了多少 pkl 会在 finalize 阶段统计。
            try:
                model.wireframe_detect(batch, str(tmp_pkl_dir))
                queue.put(("DONE", len(batch)))
            except Exception as e:
                print(f"[WARN][worker {worker_id}] Batch failed. Switch to image-by-image mode. Error: {e}")

                done_count = 0
                fail_count = 0
                for img_path in batch:
                    try:
                        model.wireframe_detect([img_path], str(tmp_pkl_dir))
                        done_count += 1
                    except Exception as inner_e:
                        fail_count += 1
                        print(f"\n[FAILED IMAGE][worker {worker_id}] {img_path}")
                        print(f"[Error Info] {inner_e}\n")

                if done_count > 0:
                    queue.put(("DONE", done_count))
                if fail_count > 0:
                    queue.put(("FAIL", fail_count))


def build_tmp_lookup(tmp_root: Path):
    """
    鲁棒收集临时 pkl。
    原脚本只找 tmp_root/worker_x/*.pkl。
    但有些 wireframe_detect 实现可能输出为：
        worker_x/*.pkl
        worker_x/subdir/*.pkl
        worker_x/hash_name.jpg.pkl
        worker_x/hash_name.png.pkl
    因此这里递归 rglob，并为多个候选 key 建索引。
    """
    tmp_lookup = {}
    all_pkls = sorted(tmp_root.rglob('*.pkl'))

    for f in all_pkls:
        name = f.name                  # xxx.pkl 或 xxx.jpg.pkl
        stem = f.stem                  # xxx 或 xxx.jpg
        keys = {name, stem}

        # 处理 xxx.jpg.pkl / xxx.png.pkl 这种保存方式
        for ext in EXTS:
            if stem.lower().endswith(ext):
                keys.add(stem[:-len(ext)])     # 去掉 .jpg/.png

        for k in keys:
            if k and k not in tmp_lookup:
                tmp_lookup[k] = f

    return tmp_lookup, all_pkls


def move_and_rename(stage_dir: Path, tmp_root: Path, mapping: dict, in_root: Path, out_root: Path):
    moved = 0
    missing = []

    tmp_lookup, all_pkls = build_tmp_lookup(tmp_root)

    print(f'[INFO] Found {len(all_pkls)} temporary pkl files under {tmp_root}.')
    if len(all_pkls) > 0:
        print('[INFO] Sample temporary pkl names:')
        for f in all_pkls[:5]:
            print(f'       {f.relative_to(tmp_root)}')

    print('[INFO] Moving files to final structure...')
    for staged_name, orig_img_path in tqdm(mapping.items(), desc="Finalizing", unit="file", dynamic_ncols=True):
        staged_stem = Path(staged_name).stem

        # 兼容多种输出命名：
        # 1) hash_frame.pkl
        # 2) hash_frame.jpg.pkl
        # 3) hash_frame.png.pkl
        candidate_keys = [
            f'{staged_stem}.pkl',
            staged_stem,
            f'{staged_name}.pkl',
        ]

        src_pkl = None
        for key in candidate_keys:
            if key in tmp_lookup:
                src_pkl = tmp_lookup[key]
                break

        if src_pkl is None:
            missing.append(staged_name)
            continue

        relative_dir = orig_img_path.parent.relative_to(in_root)
        dst_dir = out_root / relative_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_pkl = dst_dir / f'{orig_img_path.stem}.pkl'

        # 如果目标文件已存在，覆盖前先删除，避免不同文件系统 move 行为不一致
        if dst_pkl.exists():
            dst_pkl.unlink()

        shutil.move(src_pkl.as_posix(), dst_pkl.as_posix())
        moved += 1

    print(f'[INFO] Done. {moved} files processed.')
    print(f'[INFO] Missing pkl after HAWP inference: {len(missing)}')

    if missing:
        log_path = out_root / 'missing_pkl_after_hawp.log'
        with open(log_path, 'w', encoding='utf-8') as f:
            for staged_name in missing:
                orig = mapping.get(staged_name)
                f.write(f'{staged_name}\t{orig}\n')
        print(f'[WARN] Missing list saved to: {log_path}')

    return moved, len(all_pkls), len(missing)


def main():
    mp.set_start_method('spawn', force=True)

    in_root = Path(args.input_path).resolve()
    out_root = Path(args.output_path).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    mapping = {}
    stage_dir, tmp_root, staged_list = stage_images(in_root, out_root, mapping)
    total_imgs = len(staged_list)

    if total_imgs == 0:
        print("All caught up. No images to process.")
        return

    n_workers = min(max(1, args.num_workers), total_imgs)
    shards = shard(staged_list, n_workers)

    tmp_dirs = []
    for i in range(n_workers):
        d = tmp_root / f'worker_{i}'
        d.mkdir(parents=True, exist_ok=True)
        tmp_dirs.append(d)

    manager = mp.Manager()
    queue = manager.Queue()

    print(f'[INFO] Launching {n_workers} workers on CUDA_VISIBLE_DEVICES={args.gpu_ids}...')

    procs = []
    for i in range(n_workers):
        if not shards[i]:
            continue
        p = mp.Process(
            target=worker_run,
            args=(shards[i], tmp_dirs[i], args.ckpt_path, queue, i)
        )
        p.start()
        procs.append(p)

    pbar = tqdm(total=total_imgs, desc="Processing", unit="img", dynamic_ncols=True)

    processed_count = 0
    failed_count = 0
    while processed_count + failed_count < total_imgs:
        alive_procs = [p for p in procs if p.is_alive()]

        if not alive_procs and queue.empty():
            print("\n[WARNING] All workers exited before progress reached total. Stopping monitor.")
            break

        try:
            msg, num = queue.get(timeout=0.5)
            if msg == "DONE":
                processed_count += num
                pbar.update(num)
            elif msg == "FAIL":
                failed_count += num
                pbar.update(num)
        except Exception:
            continue

    pbar.close()

    for p in procs:
        p.join()

    moved, tmp_pkl_count, missing_count = move_and_rename(stage_dir, tmp_root, mapping, in_root, out_root)

    # 如果 moved=0 或还有 missing，保留临时目录，方便你检查到底 HAWP 输出在哪里、输出叫什么。
    should_keep_tmp = args.keep_tmp or moved == 0 or missing_count > 0

    if should_keep_tmp:
        print('[WARN] Temporary directories are kept for debugging:')
        print(f'       stage_dir = {stage_dir}')
        print(f'       tmp_root  = {tmp_root}')
        print('       You can inspect them with:')
        print(f'       find {tmp_root} -name "*.pkl" | head')
    else:
        try:
            safe_unlink_or_rmtree(stage_dir)
            safe_unlink_or_rmtree(tmp_root)
        except Exception as e:
            print(f'[WARN] Failed to remove temporary dirs: {e}')

    print('\n' + '=' * 40)
    print('Summary')
    print(f'Total staged images : {total_imgs}')
    print(f'Worker reported done: {processed_count}')
    print(f'Worker reported fail: {failed_count}')
    print(f'Temporary pkls found: {tmp_pkl_count}')
    print(f'Final pkls moved    : {moved}')
    print(f'Missing pkls        : {missing_count}')
    print('=' * 40)


if __name__ == '__main__':
    main()
