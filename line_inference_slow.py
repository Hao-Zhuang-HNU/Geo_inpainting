# -*- coding: utf-8 -*-
import argparse
import hashlib
import multiprocessing as mp
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFile
from tqdm import tqdm

from datasets.dataset_TSR import ContinuousEdgeLineDatasetMask
from src.models.TSR_model_RefKV import EdgeLineGPTConfig, EdgeLineGPT256RelBCE
from src.utils_RefKV import set_seed, SampleEdgeLineLogitsWithRefExtraction


# 允许加载轻微损坏/截断的图片，和原 lsm_hawp_inference_acceration.py 保持一致。
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
PKL_OUTPUT_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def natural_key(path):
    """Natural order: frame_2.png < frame_10.png."""
    name = Path(path).as_posix()
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', name)]


def list_images_recursive(folder):
    folder = Path(folder)
    files = [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files, key=natural_key)


def read_path_list(list_path):
    """
    Read a txt path list. Empty lines and lines starting with # are ignored.
    Relative paths are resolved relative to the txt file location.
    """
    list_path = Path(list_path).expanduser().resolve()
    if not list_path.is_file():
        raise FileNotFoundError(f'Path list not found: {list_path}')

    paths = []
    base_dir = list_path.parent
    with open(list_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            q = Path(line).expanduser()
            if not q.is_absolute():
                q = base_dir / q
            paths.append(q.resolve())

    if not paths:
        raise RuntimeError(f'No valid paths found in list: {list_path}')
    return paths


def validate_image_paths(paths, list_name):
    valid = []
    bad = []
    for p in paths:
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            valid.append(p)
        else:
            bad.append(p)
    if bad:
        msg = '\n'.join(str(x) for x in bad[:20])
        raise RuntimeError(
            f'{list_name} contains {len(bad)} invalid or missing image files. '
            f'Examples:\n{msg}'
        )
    return valid


def common_parent_root(paths):
    """Return the deepest common parent directory for path-list inputs."""
    parents = [str(Path(p).resolve().parent) for p in paths]
    root = Path(os.path.commonpath(parents)).resolve()
    return root


def write_path_list(list_path, paths):
    list_path = Path(list_path)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(list_path, 'w', encoding='utf-8') as f:
        for p in paths:
            f.write(str(Path(p).resolve()) + '\n')


def safe_unlink_or_rmtree(path):
    path = Path(path)
    if not path.exists():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


class SilentTqdm:
    """静默进度条，用于屏蔽 HAWP 子进程内部 tqdm 输出。"""
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


def _ensure_4d(t):
    if t.dim() == 2:
        return t.unsqueeze(0).unsqueeze(0)
    if t.dim() == 3:
        return t.unsqueeze(0)
    return t


def _tensor_to_np(t):
    if isinstance(t, torch.Tensor):
        return np.squeeze(t.detach().cpu().numpy())
    return np.squeeze(np.asarray(t))


def dilate_tensor(x, kernel_size=3):
    if kernel_size <= 1:
        return x
    padding = kernel_size // 2
    return F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)


def prepare_gt_line_input(gt_line, mask, dilate_line=1):
    """Prepare known-region GT line input (line dilation is applied at pkl-loading stage)."""
    _ = dilate_line
    return gt_line * (1 - mask)


def patch_dataset_line_loader_with_dilation(dataset, dilate_line):
    """Dilate line map right after loading from pkl in dataset.load_wireframe."""
    if dilate_line <= 1:
        return

    raw_load_wireframe = dataset.load_wireframe

    def _load_wireframe_with_dilate(selected_basename, size):
        line = raw_load_wireframe(selected_basename, size)
        line = np.asarray(line, dtype=np.float32)
        if line.ndim == 3:
            line = line[..., 0]
        line = np.clip(line, 0.0, 1.0)

        k = np.ones((dilate_line, dilate_line), dtype=np.uint8)
        line_u8 = (line * 255.0).astype(np.uint8)
        line_u8 = cv2.dilate(line_u8, k, iterations=1)
        line = line_u8.astype(np.float32) / 255.0
        return np.clip(line, 0.0, 1.0).astype(np.float32)

    dataset.load_wireframe = _load_wireframe_with_dilate


def _safe_stage_name(img_path, image_base_root):
    rel = img_path.relative_to(image_base_root).as_posix()
    h = hashlib.sha1(rel.encode('utf-8')).hexdigest()[:10]
    return f'{h}_{img_path.stem}{img_path.suffix.lower()}'


def _stage_images_for_hawp_from_list(img_paths, image_base_root, out_root, max_images=0):
    """
    将 imgs_list 中的原始图片映射到扁平临时目录，避免 HAWP 对深层目录/同名文件处理出错。
    返回：used_img_paths, mapping, stage_dir, tmp_root, staged_paths。

    final pkl 会写到：
        out_root / 原图片相对 image_base_root 的目录 / image_stem.pkl
    """
    image_base_root = Path(image_base_root).resolve()
    out_root = Path(out_root).resolve()
    stage_dir = out_root / '__stage_images__'
    tmp_root = out_root / '__tmp_pkls__'
    stage_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    all_img_paths = [Path(p).resolve() for p in img_paths]
    if max_images and max_images > 0:
        all_img_paths = all_img_paths[:max_images]

    if not all_img_paths:
        raise RuntimeError('No images provided for HAWP preprocessing.')

    mapping = {}
    staged_paths = []
    skipped_count = 0

    print('[PRE] Preparing HAWP stage environment from imgs_list...')
    for img in tqdm(all_img_paths, desc='Staging', unit='img', dynamic_ncols=True):
        relative_dir = img.parent.relative_to(image_base_root)
        final_pkl_path = out_root / relative_dir / f'{img.stem}.pkl'

        if final_pkl_path.exists():
            skipped_count += 1
            continue

        staged_name = _safe_stage_name(img, image_base_root)
        dst = stage_dir / staged_name
        mapping[staged_name] = img

        if not dst.exists():
            try:
                os.symlink(img.as_posix(), dst.as_posix())
            except OSError:
                shutil.copy2(img.as_posix(), dst.as_posix())

        staged_paths.append(str(dst))

    if skipped_count > 0:
        print(f'[PRE] Reusing {skipped_count} existing pkl files.')

    return all_img_paths, mapping, stage_dir, tmp_root, staged_paths


def _shard(lst, n):
    n = max(1, n)
    size = (len(lst) + n - 1) // n
    return [lst[i * size:(i + 1) * size] for i in range(n)]


def _hawp_worker_run(staged_list, tmp_pkl_dir, ckpt_path, threshold, hawp_size, batch_size, queue, worker_id):
    if not staged_list:
        return

    # 子进程内导入，避免在不启用 HAWP 预处理时强依赖 lsm_hawp。
    from src.lsm_hawp.lsm_hawp_model import LSM_HAWP
    import src.lsm_hawp.lsm_hawp_model as lsm_hawp_module

    lsm_hawp_module.tqdm = SilentTqdm

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True

    try:
        model = LSM_HAWP(threshold=threshold, size=hawp_size)
        try:
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            state = torch.load(ckpt_path, map_location='cpu')
        model.lsm_hawp.load_state_dict(state['model'])
        model.lsm_hawp.cuda()
        model.lsm_hawp.eval()
    except Exception as e:
        import traceback
        print(f'[ERROR][HAWP worker {worker_id}] setup failed: {e}')
        traceback.print_exc()
        queue.put(('FAIL', len(staged_list)))
        return

    batch_size = max(1, int(batch_size))
    with torch.no_grad():
        total = len(staged_list)
        for i in range(0, total, batch_size):
            batch = staged_list[i:i + batch_size]
            try:
                model.wireframe_detect(batch, str(tmp_pkl_dir))
                queue.put(('DONE', len(batch)))
            except Exception as e:
                print(f'[WARN][HAWP worker {worker_id}] Batch failed; switch to image-by-image. Error: {e}')
                done_count = 0
                fail_count = 0
                for img_path in batch:
                    try:
                        model.wireframe_detect([img_path], str(tmp_pkl_dir))
                        done_count += 1
                    except Exception as inner_e:
                        fail_count += 1
                        print(f'\n[FAILED IMAGE][HAWP worker {worker_id}] {img_path}')
                        print(f'[Error Info] {inner_e}\n')
                if done_count > 0:
                    queue.put(('DONE', done_count))
                if fail_count > 0:
                    queue.put(('FAIL', fail_count))


def _build_tmp_pkl_lookup(tmp_root):
    """递归收集 HAWP 临时 pkl，并兼容 xxx.pkl / xxx.jpg.pkl / xxx.png.pkl 等命名。"""
    tmp_lookup = {}
    all_pkls = sorted(tmp_root.rglob('*.pkl'))

    for f in all_pkls:
        name = f.name
        stem = f.stem
        keys = {name, stem}

        for ext in PKL_OUTPUT_EXTS:
            if stem.lower().endswith(ext):
                keys.add(stem[:-len(ext)])

        for k in keys:
            if k and k not in tmp_lookup:
                tmp_lookup[k] = f

    return tmp_lookup, all_pkls


def _move_and_rename_hawp_pkls(tmp_root, mapping, image_base_root, out_root):
    moved = 0
    missing = []
    image_base_root = Path(image_base_root).resolve()
    tmp_lookup, all_pkls = _build_tmp_pkl_lookup(tmp_root)

    print(f'[PRE] Found {len(all_pkls)} temporary pkl files under {tmp_root}.')
    if all_pkls:
        print('[PRE] Sample temporary pkl names:')
        for f in all_pkls[:5]:
            print(f'      {f.relative_to(tmp_root)}')

    print('[PRE] Moving HAWP pkls to final image-relative structure...')
    for staged_name, orig_img_path in tqdm(mapping.items(), desc='Finalizing pkls', unit='file', dynamic_ncols=True):
        staged_stem = Path(staged_name).stem
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

        relative_dir = orig_img_path.parent.relative_to(image_base_root)
        dst_dir = out_root / relative_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_pkl = dst_dir / f'{orig_img_path.stem}.pkl'

        if dst_pkl.exists():
            dst_pkl.unlink()
        shutil.move(src_pkl.as_posix(), dst_pkl.as_posix())
        moved += 1

    print(f'[PRE] HAWP final pkls moved: {moved}')
    print(f'[PRE] Missing pkl after HAWP inference: {len(missing)}')

    if missing:
        log_path = out_root / 'missing_pkl_after_hawp.log'
        with open(log_path, 'w', encoding='utf-8') as f:
            for staged_name in missing:
                orig = mapping.get(staged_name)
                f.write(f'{staged_name}\t{orig}\n')
        print(f'[WARN] Missing pkl list saved to: {log_path}')

    return moved, len(all_pkls), len(missing)


def _pkl_path_for_image(img_path, image_base_root, pkl_root):
    rel_dir = Path(img_path).resolve().parent.relative_to(Path(image_base_root).resolve())
    return Path(pkl_root).resolve() / rel_dir / f'{Path(img_path).stem}.pkl'


def run_hawp_preprocess(opts, img_paths, image_base_root, pkl_root, pkl_list_path):
    """
    1) 在 tmp/pkls 下生成并整理 pkl；
    2) 严格按 imgs_list 的顺序写出 tmp/pkls.txt；
    3) 将 opts.test_line_list 指向这个临时 pkl list。
    """
    image_base_root = Path(image_base_root).resolve()
    pkl_root = Path(pkl_root).resolve()
    pkl_list_path = Path(pkl_list_path).resolve()
    pkl_root.mkdir(parents=True, exist_ok=True)

    all_img_paths, mapping, stage_dir, tmp_pkl_root, staged_list = _stage_images_for_hawp_from_list(
        img_paths, image_base_root, pkl_root, max_images=opts.hawp_max_images
    )
    total_imgs = len(staged_list)

    if total_imgs > 0:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        n_workers = min(max(1, int(opts.hawp_num_workers)), total_imgs)
        shards = _shard(staged_list, n_workers)
        tmp_dirs = []
        for i in range(n_workers):
            d = tmp_pkl_root / f'worker_{i}'
            d.mkdir(parents=True, exist_ok=True)
            tmp_dirs.append(d)

        manager = mp.Manager()
        queue = manager.Queue()

        print(f'[PRE] Launching {n_workers} HAWP workers on CUDA_VISIBLE_DEVICES={opts.GPU_ids}...')
        procs = []
        for i in range(n_workers):
            if not shards[i]:
                continue
            p = mp.Process(
                target=_hawp_worker_run,
                args=(
                    shards[i],
                    tmp_dirs[i],
                    opts.hawp_ckpt_path,
                    opts.hawp_threshold,
                    opts.hawp_size,
                    opts.hawp_batch_size,
                    queue,
                    i,
                ),
            )
            p.start()
            procs.append(p)

        pbar = tqdm(total=total_imgs, desc='HAWP', unit='img', dynamic_ncols=True)
        processed_count = 0
        failed_count = 0
        while processed_count + failed_count < total_imgs:
            alive_procs = [p for p in procs if p.is_alive()]
            if not alive_procs and queue.empty():
                print('\n[WARNING] All HAWP workers exited before progress reached total. Stop monitor.')
                break
            try:
                msg, num = queue.get(timeout=0.5)
                if msg == 'DONE':
                    processed_count += num
                    pbar.update(num)
                elif msg == 'FAIL':
                    failed_count += num
                    pbar.update(num)
            except Exception:
                continue
        pbar.close()

        for p in procs:
            p.join()

        moved, tmp_pkl_count, missing_count = _move_and_rename_hawp_pkls(
            tmp_pkl_root, mapping, image_base_root, pkl_root
        )
        should_keep_tmp = opts.keep_tmp or moved == 0 or missing_count > 0
    else:
        print('[PRE] All HAWP pkls already exist; skip HAWP inference.')
        processed_count = 0
        failed_count = 0
        moved = 0
        tmp_pkl_count = 0
        missing_count = 0
        should_keep_tmp = opts.keep_tmp

    pkl_paths = []
    missing_final = []
    for img_path in all_img_paths:
        pkl_path = _pkl_path_for_image(img_path, image_base_root, pkl_root)
        if pkl_path.exists():
            pkl_paths.append(pkl_path)
        else:
            missing_final.append((img_path, pkl_path))

    if missing_final:
        log_path = pkl_root / 'missing_final_pkls.log'
        with open(log_path, 'w', encoding='utf-8') as f:
            for img_path, pkl_path in missing_final:
                f.write(f'{img_path}\t{pkl_path}\n')
        msg = f'Missing final pkl files: {len(missing_final)}. See {log_path}'
        if not opts.allow_missing_pkls:
            raise RuntimeError(msg)
        print(f'[WARN] {msg}')

    write_path_list(pkl_list_path, pkl_paths)
    opts.test_line_list = str(pkl_list_path)

    if should_keep_tmp:
        print('[WARN] HAWP temporary stage dirs are kept for debugging:')
        print(f'       stage_dir = {stage_dir}')
        print(f'       tmp_root  = {tmp_pkl_root}')
    else:
        try:
            safe_unlink_or_rmtree(stage_dir)
            safe_unlink_or_rmtree(tmp_pkl_root)
        except Exception as e:
            print(f'[WARN] Failed to remove HAWP temporary dirs: {e}')

    print('\n' + '=' * 40)
    print('HAWP Preprocess Summary')
    print(f'Input images          : {len(all_img_paths)}')
    print(f'New staged images     : {total_imgs}')
    print(f'Worker reported done  : {processed_count}')
    print(f'Worker reported fail  : {failed_count}')
    print(f'Temporary pkls found  : {tmp_pkl_count}')
    print(f'Final pkls moved      : {moved}')
    print(f'Missing during move   : {missing_count}')
    print(f'Pkl list written      : {pkl_list_path}')
    print('=' * 40 + '\n')

    return str(pkl_list_path)


def load_mask(mask_path, target_hw, threshold=127):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f'Failed to read mask: {mask_path}')

    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    h, w = target_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask > threshold


def fill_mask(img, mask_bool, fill_value=255):
    out = img.copy()
    fill_value = int(np.clip(fill_value, 0, 255))
    out[mask_bool] = (fill_value, fill_value, fill_value)
    return out


def overlay_green(img, mask_bool, alpha=0.3):
    out = img.copy()
    green = np.zeros_like(img, dtype=np.uint8)
    green[:, :] = (0, 255, 0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = cv2.addWeighted(img, 1.0 - alpha, green, alpha, 0)
    out[mask_bool] = blended[mask_bool]
    return out


def make_output_path(img_path, img_base_root, out_root):
    rel = Path(img_path).resolve().relative_to(Path(img_base_root).resolve())
    out_path = Path(out_root).resolve() / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def apply_masks_to_images_from_lists(
    img_paths,
    mask_paths,
    img_base_root,
    out_root,
    list_path,
    mode='white',
    fill_value=255,
    alpha=0.3,
    mask_threshold=127,
    repeat_masks=True,
):
    """
    将 imgs_list 中的原始 GT 图片按清单顺序和 mask_list 匹配，输出 masked image。
    默认 mode='white'，即补全器输入常用的白色不透明遮挡。
    """
    img_base_root = Path(img_base_root).resolve()
    out_root = Path(out_root).resolve()
    list_path = Path(list_path).resolve()

    # 避免 tmp/masked_images 里残留上一次不同序列的图片，导致 dataset 读入脏数据。
    if out_root.exists():
        safe_unlink_or_rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    img_paths = [Path(p).resolve() for p in img_paths]
    mask_paths = [Path(p).resolve() for p in mask_paths]

    if not img_paths:
        raise RuntimeError('No images provided by --imgs_list.')
    if not mask_paths:
        raise RuntimeError('No masks provided by --mask_list.')

    if repeat_masks:
        n = len(img_paths)
        if len(img_paths) != len(mask_paths):
            print(
                f'[PRE] Image count != mask count: {len(img_paths)} images vs {len(mask_paths)} masks. '
                f'Masks will be reused cyclically.'
            )
    else:
        n = min(len(img_paths), len(mask_paths))
        if len(img_paths) != len(mask_paths):
            print(
                f'[WARN] Image count != mask count: {len(img_paths)} images vs {len(mask_paths)} masks. '
                f'Only processing first {n} pairs because repeat_masks=False.'
            )

    print(f'[PRE] Mask overlay input images : {len(img_paths)}')
    print(f'[PRE] Mask overlay masks        : {len(mask_paths)}')
    print(f'[PRE] Mask overlay processing   : {n}')
    print(f'[PRE] Mask overlay mode         : {mode}')

    written_paths = []
    used_mask_paths = []
    for i in tqdm(range(n), desc='Overlay masks', unit='img', dynamic_ncols=True):
        img_path = img_paths[i]
        mask_path = mask_paths[i % len(mask_paths)] if repeat_masks else mask_paths[i]

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'Failed to read image: {img_path}')

        h, w = img.shape[:2]
        mask_bool = load_mask(mask_path, target_hw=(h, w), threshold=mask_threshold)

        if mode in ('white', 'fill'):
            out = fill_mask(img, mask_bool, fill_value=fill_value)
        elif mode == 'green':
            out = overlay_green(img, mask_bool, alpha=alpha)
        else:
            raise ValueError(f'Unsupported overlay mode: {mode}')

        out_path = make_output_path(img_path, img_base_root, out_root)
        ok = cv2.imwrite(str(out_path), out)
        if not ok:
            raise RuntimeError(f'Failed to write output: {out_path}')
        written_paths.append(out_path)
        used_mask_paths.append(mask_path)

    write_path_list(list_path, written_paths)
    print(f'[PRE] Masked images written to : {out_root}')
    print(f'[PRE] Masked image list saved : {list_path}')
    print(f'[PRE] Valid masked images     : {len(written_paths)}')

    return str(list_path), written_paths, used_mask_paths


def _make_effective_pairs(img_paths, mask_paths, repeat_masks=True):
    """Return image/mask paths with one mask per image, preserving imgs_list order."""
    if not img_paths:
        raise RuntimeError('No images provided by --imgs_list.')
    if not mask_paths:
        raise RuntimeError('No masks provided by --mask_list.')

    if repeat_masks:
        if len(img_paths) != len(mask_paths):
            print(
                f'[PRE] Image count != mask count: {len(img_paths)} images vs {len(mask_paths)} masks. '
                f'Masks will be reused cyclically.'
            )
        used_img_paths = list(img_paths)
        used_mask_paths = [mask_paths[i % len(mask_paths)] for i in range(len(img_paths))]
    else:
        n = min(len(img_paths), len(mask_paths))
        if len(img_paths) != len(mask_paths):
            print(
                f'[WARN] Image count != mask count: {len(img_paths)} images vs {len(mask_paths)} masks. '
                f'Only first {n} pairs will be used because --no_repeat_masks is enabled.'
            )
        used_img_paths = list(img_paths[:n])
        used_mask_paths = list(mask_paths[:n])

    return used_img_paths, used_mask_paths


def preprocess_inputs(opts):
    """Run HAWP pkl generation and masked-image overlay before line inference."""
    if getattr(opts, 'image_url', None) and not getattr(opts, 'imgs_list', None):
        print('[WARN] --image_url is deprecated; use --imgs_list instead.')
        opts.imgs_list = opts.image_url
    if getattr(opts, 'mask_url', None) and not getattr(opts, 'mask_list', None):
        print('[WARN] --mask_url is deprecated; use --mask_list instead.')
        opts.mask_list = opts.mask_url

    if not opts.imgs_list:
        raise ValueError('--imgs_list is required and must point to an imgs.txt path-list file.')
    if not opts.mask_list:
        raise ValueError('--mask_list is required and must point to a mask txt path-list file.')

    if not opts.auto_preprocess:
        print('[PRE] auto_preprocess=False; use --imgs_list/--mask_list directly.')
        return opts

    tmp_root = Path(opts.tmp_root).resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)

    img_paths = validate_image_paths(read_path_list(opts.imgs_list), '--imgs_list')
    mask_paths = validate_image_paths(read_path_list(opts.mask_list), '--mask_list')
    used_img_paths, used_mask_paths = _make_effective_pairs(
        img_paths, mask_paths, repeat_masks=not opts.no_repeat_masks
    )
    if opts.hawp_max_images and opts.hawp_max_images > 0:
        # Keep image/mask/pkl lists aligned when using max_images for a small debug run.
        used_img_paths = used_img_paths[:opts.hawp_max_images]
        used_mask_paths = used_mask_paths[:opts.hawp_max_images]
        print(f'[PRE] hawp_max_images={opts.hawp_max_images}; truncate effective inference pairs accordingly.')

    image_base_root = common_parent_root(used_img_paths)
    print(f'[PRE] Loaded imgs_list : {Path(opts.imgs_list).resolve()}')
    print(f'[PRE] Loaded mask_list : {Path(opts.mask_list).resolve()}')
    print(f'[PRE] Effective images : {len(used_img_paths)}')
    print(f'[PRE] Effective masks  : {len(used_mask_paths)}')
    print(f'[PRE] Image base root  : {image_base_root}')

    effective_imgs_list = tmp_root / 'effective_imgs.txt'
    effective_masks_list = tmp_root / 'effective_masks.txt'
    write_path_list(effective_imgs_list, used_img_paths)
    write_path_list(effective_masks_list, used_mask_paths)

    pkl_root = tmp_root / 'pkls'
    pkl_list_path = tmp_root / 'pkls.txt'
    masked_img_root = tmp_root / 'masked_images'
    masked_img_list = tmp_root / 'masked_images.txt'

    if opts.force_hawp or not opts.test_line_list:
        print('[PRE] Start HAWP pkl preprocessing...')
        run_hawp_preprocess(opts, used_img_paths, image_base_root, pkl_root, pkl_list_path)
    else:
        print(f'[PRE] Reuse user-provided --test_line_list: {opts.test_line_list}')

    # Dataset must read a one-mask-per-image list after possible cyclic mask expansion/truncation.
    opts.mask_list = str(effective_masks_list)

    if not opts.no_mask_overlay:
        print('[PRE] Start masked-image preprocessing...')
        masked_list, masked_paths, used_mask_paths_after_overlay = apply_masks_to_images_from_lists(
            used_img_paths,
            used_mask_paths,
            image_base_root,
            masked_img_root,
            masked_img_list,
            mode=opts.overlay_mode,
            fill_value=opts.fill_value,
            alpha=opts.mask_alpha,
            mask_threshold=opts.mask_threshold,
            repeat_masks=True,
        )
        if len(masked_paths) != len(used_mask_paths_after_overlay):
            raise RuntimeError('Internal error: masked image list and mask list lengths differ.')
        opts.imgs_list = masked_list
        opts.masked_image_list = masked_list
        print(f'[PRE] line_inference will use masked image list as --imgs_list: {opts.imgs_list}')
    else:
        opts.imgs_list = str(effective_imgs_list)
        print(f'[PRE] no_mask_overlay=True; line_inference will use effective source image list: {opts.imgs_list}')

    print(f'[PRE] line_inference will use mask list: {opts.mask_list}')
    print(f'[PRE] line_inference will use pkl list : {opts.test_line_list}')

    return opts


def build_model(opts):
    cfg = EdgeLineGPTConfig(
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        n_embd=opts.n_embd,
        block_size=32,
        attn_pdrop=0.0,
        n_layer=opts.n_layer,
        n_head=opts.n_head,
        use_ref_kv=True,
    )
    model = EdgeLineGPT256RelBCE(cfg)

    print(f'Loading checkpoint from {opts.ckpt_path}')
    checkpoint = torch.load(opts.ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)

    clean_state = {}
    for k, v in state_dict.items():
        if 'attn.mask' in k:
            continue
        nk = k.replace('module.', '').replace('_orig_mod.', '')
        clean_state[nk] = v

    model.load_state_dict(clean_state, strict=False)
    model.cuda().eval()
    return model


def save_pred(prob, gt, mask, out_file, binary_threshold=None):
    pred_np = _tensor_to_np(prob)
    if binary_threshold is not None:
        pred_np = (pred_np >= binary_threshold).astype(np.float32)
    gt_np = _tensor_to_np(gt)
    mask_np = _tensor_to_np(mask)
    merged = pred_np * mask_np + gt_np * (1 - mask_np)

    vis = (np.stack([merged] * 3, axis=-1) * 255).astype(np.uint8)
    cv2.imwrite(out_file, vis[:, :, ::-1])

    return torch.from_numpy(merged).unsqueeze(0).unsqueeze(0).cuda()


def geo_inference(opts):
    """Inference without npz-based alignment; local reference uses previous generated frame."""
    model = build_model(opts)

    dataset = ContinuousEdgeLineDatasetMask(
        opts.imgs_list,
        test_mask_path=opts.mask_list,
        is_train=False,
        image_size=opts.image_size,
        line_path=opts.test_line_list,
    )
    patch_dataset_line_loader_with_dilation(dataset, opts.dilate_line)

    print(f'Inference frames: {len(dataset)} (dilate_line={opts.dilate_line})')

    # ---- Frame 0: use GT as initialization and produce first generated frame ----
    first = dataset[0]
    name0 = os.path.basename(first['name'][0] if isinstance(first['name'], list) else first['name'])
    if not name0.lower().endswith(('.png', '.jpg', '.jpeg')):
        name0 += '.png'

    img0 = _ensure_4d(first['img']).cuda()
    mask0 = _ensure_4d(first['mask']).cuda()
    gt_e0 = _ensure_4d(first['edge']).cuda()
    gt_l0 = _ensure_4d(first['line']).cuda()

    with torch.no_grad():
        edge0, line0, _ = SampleEdgeLineLogitsWithRefExtraction(
            model,
            context=[img0, gt_e0 * (1 - mask0), prepare_gt_line_input(gt_l0, mask0, opts.dilate_line)],
            mask=mask0,
            iterations=opts.iterations,
            extract_ref=False,
        )

    # 保存第一帧预测，并作为后续 global/local 的“生成参考帧”
    pred_edge_prev = save_pred(edge0[0], gt_e0, mask0, os.path.join(opts.save_url, 'edge', name0))
    pred_line_prev = save_pred(
        line0[0],
        gt_l0,
        mask0,
        os.path.join(opts.save_url, 'line', name0),
        binary_threshold=opts.line_binary_thresh if opts.solid_line else None,
    )

    global_edge_gen = pred_edge_prev.clone()
    global_line_gen = pred_line_prev.clone()

    with torch.no_grad():
        global_ref_feat = model.extract_reference_features(
            global_img=None,
            global_edge=dilate_tensor(global_edge_gen, opts.ref_dilate),
            global_line=dilate_tensor(global_line_gen, opts.ref_dilate),
        )

    # ---- Frame 1...N: local/global refs all from generated frames ----
    for i in tqdm(range(1, len(dataset))):
        item = dataset[i]
        name = os.path.basename(item['name'][0] if isinstance(item['name'], list) else item['name'])
        if not name.lower().endswith(('.png', '.jpg', '.jpeg')):
            name += '.png'

        img = _ensure_4d(item['img']).cuda()
        mask = _ensure_4d(item['mask']).cuda()
        gt_e = _ensure_4d(item['edge']).cuda()
        gt_l = _ensure_4d(item['line']).cuda()

        local_edge = pred_edge_prev.clone()
        local_line = pred_line_prev.clone()

        local_edge = dilate_tensor(local_edge, opts.ref_dilate)
        local_line = dilate_tensor(local_line, opts.ref_dilate)

        with torch.no_grad():
            local_ref_feat = model.extract_reference_features(
                global_img=None,
                local_edge=local_edge,
                local_line=local_line,
                local_mask=torch.zeros_like(mask),
            )
            ref_feat = torch.cat([global_ref_feat, local_ref_feat], dim=2)

            edge_pred, line_pred, _ = SampleEdgeLineLogitsWithRefExtraction(
                model,
                context=[img, gt_e * (1 - mask), prepare_gt_line_input(gt_l, mask, opts.dilate_line)],
                mask=mask,
                iterations=opts.iterations,
                ref_feat=ref_feat,
                extract_ref=False,
            )

        pred_edge_prev = save_pred(edge_pred[0], gt_e, mask, os.path.join(opts.save_url, 'edge', name))
        pred_line_prev = save_pred(
            line_pred[0],
            gt_l,
            mask,
            os.path.join(opts.save_url, 'line', name),
            binary_threshold=opts.line_binary_thresh if opts.solid_line else None,
        )


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--imgs_list', type=str, default=None,
                        help='Txt path-list of input GT/source images, one image path per line.')
    parser.add_argument('--mask_list', type=str, default=None,
                        help='Txt path-list of masks, one mask path per line.')
    # Backward-compatible hidden aliases. Prefer --imgs_list/--mask_list in new commands.
    parser.add_argument('--image_url', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--mask_url', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--test_line_list', type=str, default='')

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--save_url', type=str, default='./results_geo')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--ref_dilate', type=int, default=3)
    parser.add_argument('--dilate_line', type=int, default=1, choices=[1, 3],
                        help='Optional GT line dilation size to match training strategy.')
    parser.add_argument('--solid_line', action='store_true',
                        help='If set, binarize generated line output to get solid lines.')
    parser.add_argument('--line_binary_thresh', type=float, default=0.5,
                        help='Threshold used when --solid_line is enabled.')

    # Integrated preprocessing options.
    parser.add_argument('--no_preprocess', dest='auto_preprocess', action='store_false',
                        help='Disable integrated HAWP pkl generation and mask-overlay preprocessing.')
    parser.set_defaults(auto_preprocess=True)
    parser.add_argument('--tmp_root', type=str, default='./tmp',
                        help='Temporary preprocessing root. Default: ./tmp')

    # HAWP preprocessing.
    parser.add_argument('--hawp_ckpt_path', type=str, default='ckpt/best_lsm_hawp.pth',
                        help='Checkpoint path for LSM-HAWP preprocessing.')
    parser.add_argument('--hawp_num_workers', type=int, default=16,
                        help='Number of HAWP worker processes. Default: 16')
    parser.add_argument('--hawp_max_images', type=int, default=0,
                        help='Limit total images for HAWP preprocessing; 0 means all. Default: 0')
    parser.add_argument('--hawp_threshold', type=float, default=0.8,
                        help='LSM-HAWP detection threshold. Default: 0.8')
    parser.add_argument('--hawp_size', type=int, default=512,
                        help='LSM-HAWP input size. Default: 512')
    parser.add_argument('--hawp_batch_size', type=int, default=16,
                        help='HAWP batch size per worker. Default: 16')
    parser.add_argument('--force_hawp', action='store_true',
                        help='Run HAWP preprocessing even if --test_line_list is provided.')
    parser.add_argument('--allow_missing_pkls', action='store_true',
                        help='Write tmp/pkls.txt even if some final pkls are missing.')
    parser.add_argument('--keep_tmp', action='store_true',
                        help='Keep HAWP stage/temp dirs for debugging.')

    # Masked-image preprocessing.
    parser.add_argument('--no_mask_overlay', action='store_true',
                        help='Do not generate tmp/masked_images; use source --imgs_list for inference.')
    parser.add_argument('--overlay_mode', choices=['white', 'fill', 'green'], default='white',
                        help='Mask overlay mode for generated input images. Default: white')
    parser.add_argument('--fill_value', type=int, default=255,
                        help='Fill value for --overlay_mode white/fill. Default: 255')
    parser.add_argument('--mask_alpha', type=float, default=0.3,
                        help='Alpha for --overlay_mode green. Default: 0.3')
    parser.add_argument('--mask_threshold', type=int, default=127,
                        help='Mask binarization threshold. Pixels > threshold are treated as mask. Default: 127')
    parser.add_argument('--no_repeat_masks', action='store_true',
                        help='Disable cyclic mask reuse when mask count is smaller than image count.')

    return parser


if __name__ == '__main__':
    set_seed(42)
    parser = build_argparser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids

    args = preprocess_inputs(args)

    os.makedirs(os.path.join(args.save_url, 'edge'), exist_ok=True)
    os.makedirs(os.path.join(args.save_url, 'line'), exist_ok=True)

    geo_inference(args)
