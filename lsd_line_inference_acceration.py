# -*- coding: utf-8 -*-
"""
LSD line preprocessing: recursive images / image-list -> HAWP-compatible pkl files,
with optional line-map visualization saved as .jpg.

The output pkl format follows src.lsm_hawp.lsm_hawp_model.LSM_HAWP:
    {"lines": [[y1, x1, y2, x2], ...], "scores": [score, ...]}

Examples:
    python lsd_line_inference_acceration.py \
        -i ./tmp/masked_images \
        -o ./tmp/lsd_pkls \
        --save_line ./tmp/lsd_vis \
        --num_workers 16 \
        --min_length 8 \
        --save_list ./tmp/lsd_pkls.txt

If your Geo inference uses --image_size 256 while source images are not 256x256,
add --resize_to 256 so that pkl coordinates match dataset.load_wireframe(size=256).
"""

import argparse
import os
import pickle
import shutil
import multiprocessing as mp
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import ImageFile
from tqdm import tqdm

# 允许加载轻微损坏/截断的图片，和 lsm_hawp_inference_acceration.py 保持一致。
ImageFile.LOAD_TRUNCATED_IMAGES = True

# suffix 已统一 lower，所以这里只放小写。
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def natural_key(path):
    name = Path(path).as_posix()
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def safe_unlink_or_rmtree(path: Path):
    if not path.exists():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def read_path_list(list_path: Path):
    list_path = list_path.expanduser().resolve()
    base_dir = list_path.parent
    paths = []
    with open(list_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip().strip('"').strip("'")
            if not line or line.startswith('#'):
                continue
            p = Path(line).expanduser()
            if not p.is_absolute():
                p = base_dir / p
            if p.is_file() and p.suffix.lower() in EXTS:
                paths.append(p.resolve())
            else:
                raise RuntimeError(f'Invalid or missing image path in list {list_path}: {p}')
    if not paths:
        raise RuntimeError(f'No valid images found in list: {list_path}')
    return paths


def collect_images(input_path: Path):
    """Return (image_paths, base_root). Supports directory or txt path-list."""
    input_path = input_path.expanduser().resolve()
    if input_path.is_dir():
        print(f'[INFO] Scanning images recursively in {input_path} ...')
        imgs = [p for p in input_path.rglob('*') if p.is_file() and p.suffix.lower() in EXTS]
        imgs = sorted(imgs, key=natural_key)
        base_root = input_path
    elif input_path.is_file() and input_path.suffix.lower() == '.txt':
        print(f'[INFO] Reading image list from {input_path} ...')
        imgs = read_path_list(input_path)
        parents = [str(p.parent) for p in imgs]
        base_root = Path(os.path.commonpath(parents)).resolve()
    elif input_path.is_file() and input_path.suffix.lower() in EXTS:
        imgs = [input_path]
        base_root = input_path.parent
    else:
        raise RuntimeError(f'Unsupported input_path: {input_path}')

    if not imgs:
        raise RuntimeError(f'No images found from input_path: {input_path}')
    return imgs, base_root


def build_lsd(args):
    refine_map = {
        0: getattr(cv2, 'LSD_REFINE_NONE', 0),
        1: getattr(cv2, 'LSD_REFINE_STD', 1),
        2: getattr(cv2, 'LSD_REFINE_ADV', 2),
    }
    refine = refine_map.get(int(args.lsd_refine), getattr(cv2, 'LSD_REFINE_STD', 1))

    if not hasattr(cv2, 'createLineSegmentDetector'):
        raise RuntimeError(
            'cv2.createLineSegmentDetector is unavailable in this OpenCV build. '
            'Please install opencv-python/opencv-contrib-python with LSD support.'
        )

    # Different OpenCV wheels expose slightly different signatures. Try full args first,
    # then fall back to refine-only.
    try:
        return cv2.createLineSegmentDetector(
            refine=refine,
            scale=float(args.lsd_scale),
            sigma_scale=float(args.lsd_sigma_scale),
            quant=float(args.lsd_quant),
            ang_th=float(args.lsd_ang_th),
            log_eps=float(args.lsd_log_eps),
            density_th=float(args.lsd_density_th),
            n_bins=int(args.lsd_n_bins),
        )
    except TypeError:
        return cv2.createLineSegmentDetector(refine)


def detect_lsd_lines(img_bgr, args):
    """Return lines in HAWP-compatible order [y1, x1, y2, x2], plus scores."""
    if args.resize_to and int(args.resize_to) > 0:
        size = int(args.resize_to)
        img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if args.equalize_hist:
        gray = cv2.equalizeHist(gray)
    if args.gaussian_blur and int(args.gaussian_blur) > 1:
        k = int(args.gaussian_blur)
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    lsd = build_lsd(args)
    detected = lsd.detect(gray)
    raw_lines = detected[0] if detected is not None else None

    out = []
    scores = []
    if raw_lines is None:
        return out, scores, gray.shape[:2]

    # raw_lines: N x 1 x 4, each line is [x1, y1, x2, y2]
    for line in raw_lines.reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < float(args.min_length):
            continue
        if args.max_length and float(args.max_length) > 0 and length > float(args.max_length):
            continue
        out.append([y1, x1, y2, x2])
        scores.append(1.0)

    if args.max_lines and int(args.max_lines) > 0 and len(out) > int(args.max_lines):
        lengths = [np.hypot(l[3] - l[1], l[2] - l[0]) for l in out]
        idxs = np.argsort(lengths)[::-1][:int(args.max_lines)]
        out = [out[i] for i in idxs]
        scores = [scores[i] for i in idxs]

    return out, scores, gray.shape[:2]


def save_pkl(lines, scores, out_pkl: Path):
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    info = {
        'lines': np.asarray(lines, dtype=np.float32),
        'scores': np.asarray(scores, dtype=np.float32),
    }
    with open(out_pkl, 'wb') as f:
        pickle.dump(info, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_line_vis(lines, hw, out_jpg: Path, thickness=1):
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    h, w = hw
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for y1, x1, y2, x2 in lines:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.line(vis, p1, p2, (255, 255, 255), int(max(1, thickness)), cv2.LINE_AA)
    # Always save .jpg as requested.
    ok = cv2.imwrite(str(out_jpg), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f'Failed to write line visualization: {out_jpg}')


def _process_one(task):
    img_path, base_root, out_root, save_line_root, args_dict = task
    # Rebuild a lightweight argparse-like object inside worker.
    args = argparse.Namespace(**args_dict)
    cv2.setNumThreads(0)

    img_path = Path(img_path).resolve()
    base_root = Path(base_root).resolve()
    out_root = Path(out_root).resolve()
    save_line_root = Path(save_line_root).resolve() if save_line_root else None

    rel = img_path.relative_to(base_root)
    out_pkl = out_root / rel.parent / f'{img_path.stem}.pkl'
    out_jpg = save_line_root / rel.parent / f'{img_path.stem}.jpg' if save_line_root is not None else None

    if (not args.overwrite) and out_pkl.exists() and (out_jpg is None or out_jpg.exists()):
        return str(img_path), str(out_pkl), str(out_jpg) if out_jpg else '', 0, 'SKIP', ''

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return str(img_path), str(out_pkl), str(out_jpg) if out_jpg else '', 0, 'FAIL', 'Failed to read image'

    try:
        lines, scores, hw = detect_lsd_lines(img, args)
        save_pkl(lines, scores, out_pkl)
        if out_jpg is not None:
            save_line_vis(lines, hw, out_jpg, thickness=args.line_thickness)
        return str(img_path), str(out_pkl), str(out_jpg) if out_jpg else '', len(lines), 'DONE', ''
    except Exception as e:
        return str(img_path), str(out_pkl), str(out_jpg) if out_jpg else '', 0, 'FAIL', str(e)


def write_list(path: Path, rows):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(str(Path(r).resolve()) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='LSD preprocessing: recursive images/list -> HAWP-compatible pkl + optional line jpg')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='Input image root, a single image, or a txt path-list.')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help='Output pkl root. Directory structure is preserved relative to input root/common parent.')
    parser.add_argument('--save_line', type=str, default='',
                        help='Optional output root for visualized line images (.jpg). Directory structure is preserved.')
    parser.add_argument('--save_list', type=str, default='',
                        help='Optional txt path to save generated pkl paths in the same order as input images.')
    parser.add_argument('--save_line_list', type=str, default='',
                        help='Optional txt path to save generated visualization jpg paths in the same order as input images.')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='CPU worker processes. 0 means use all available CPU cores. Default: 16')
    parser.add_argument('--max_images', type=int, default=0,
                        help='Limit total images; 0 means all. Default: 0')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing pkl/jpg files.')
    parser.add_argument('--clear_output', action='store_true',
                        help='Remove output_path and save_line directory before processing. Use carefully.')

    # Geometry / filtering.
    parser.add_argument('--resize_to', type=int, default=0,
                        help='Resize image to square size before LSD and save coordinates in this size. 0 keeps original size. Use 256 for Geo image_size=256 if needed.')
    parser.add_argument('--min_length', type=float, default=8.0,
                        help='Drop LSD segments shorter than this length in pixels. Default: 8')
    parser.add_argument('--max_length', type=float, default=0.0,
                        help='Drop segments longer than this length; 0 disables. Default: 0')
    parser.add_argument('--max_lines', type=int, default=0,
                        help='Keep only the longest K lines; 0 disables. Default: 0')
    parser.add_argument('--line_thickness', type=int, default=1,
                        help='Line thickness for visualization jpg. Default: 1')

    # Basic image preprocessing.
    parser.add_argument('--equalize_hist', action='store_true',
                        help='Apply histogram equalization before LSD. Sometimes increases lines but may add texture noise.')
    parser.add_argument('--gaussian_blur', type=int, default=0,
                        help='Optional odd Gaussian kernel size before LSD; 0 disables. Example: 3')

    # LSD parameters. Defaults follow OpenCV defaults where possible.
    parser.add_argument('--lsd_refine', type=int, default=1, choices=[0, 1, 2],
                        help='LSD refine mode: 0=none, 1=std, 2=adv. Default: 1')
    parser.add_argument('--lsd_scale', type=float, default=0.8)
    parser.add_argument('--lsd_sigma_scale', type=float, default=0.6)
    parser.add_argument('--lsd_quant', type=float, default=2.0)
    parser.add_argument('--lsd_ang_th', type=float, default=22.5)
    parser.add_argument('--lsd_log_eps', type=float, default=0.0)
    parser.add_argument('--lsd_density_th', type=float, default=0.7)
    parser.add_argument('--lsd_n_bins', type=int, default=1024)

    return parser.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input_path).resolve()
    out_root = Path(args.output_path).resolve()
    save_line_root = Path(args.save_line).resolve() if args.save_line else None

    imgs, base_root = collect_images(in_path)
    imgs = sorted(imgs, key=natural_key)
    if args.max_images and args.max_images > 0:
        imgs = imgs[:args.max_images]

    if args.clear_output:
        safe_unlink_or_rmtree(out_root)
        if save_line_root is not None:
            safe_unlink_or_rmtree(save_line_root)

    out_root.mkdir(parents=True, exist_ok=True)
    if save_line_root is not None:
        save_line_root.mkdir(parents=True, exist_ok=True)

    total = len(imgs)
    if total == 0:
        raise RuntimeError('No images to process.')

    cpu_count = os.cpu_count() or 1
    if args.num_workers is None or int(args.num_workers) <= 0:
        n_workers = min(cpu_count, total)
    else:
        n_workers = min(max(1, int(args.num_workers)), total)

    print(f'[INFO] Input root/common parent : {base_root}')
    print(f'[INFO] Output pkl root          : {out_root}')
    if save_line_root is not None:
        print(f'[INFO] Save line jpg root      : {save_line_root}')
    print(f'[INFO] Total images            : {total}')
    print(f'[INFO] Workers                 : {n_workers}')
    print(f'[INFO] resize_to               : {args.resize_to}')
    print(f'[INFO] min_length              : {args.min_length}')

    # Keep only serializable simple values for workers.
    args_dict = vars(args).copy()
    tasks = [(str(p), str(base_root), str(out_root), str(save_line_root) if save_line_root else '', args_dict) for p in imgs]

    done = 0
    skipped = 0
    failed = 0
    line_counts = []
    pkl_paths_ordered = [None] * total
    jpg_paths_ordered = [None] * total

    def _handle_result(i, res):
        nonlocal done, skipped, failed
        img_path, out_pkl, out_jpg, n_lines, status, err = res
        pkl_paths_ordered[i] = out_pkl
        jpg_paths_ordered[i] = out_jpg if out_jpg else None
        if status == 'DONE':
            done += 1
            line_counts.append(int(n_lines))
        elif status == 'SKIP':
            skipped += 1
        else:
            failed += 1
            print(f'\n[FAILED] {img_path}\n         {err}\n')

    if n_workers <= 1:
        for i, task in enumerate(tqdm(tasks, desc='LSD', unit='img', dynamic_ncols=True)):
            _handle_result(i, _process_one(task))
    else:
        try:
            ctx = mp.get_context('fork')
        except ValueError:
            ctx = mp.get_context()
        chunksize = max(1, total // (n_workers * 8))
        with ctx.Pool(processes=n_workers, maxtasksperchild=200) as pool:
            for i, res in enumerate(tqdm(pool.imap(_process_one, tasks, chunksize=chunksize),
                                         total=total, desc='LSD', unit='img', dynamic_ncols=True)):
                _handle_result(i, res)

    if args.save_list:
        valid_pkls = [p for p in pkl_paths_ordered if p]
        write_list(Path(args.save_list), valid_pkls)
        print(f'[INFO] Pkl list saved          : {Path(args.save_list).resolve()}')

    if args.save_line_list and save_line_root is not None:
        valid_jpgs = [p for p in jpg_paths_ordered if p]
        write_list(Path(args.save_line_list), valid_jpgs)
        print(f'[INFO] Line jpg list saved     : {Path(args.save_line_list).resolve()}')

    mean_lines = float(np.mean(line_counts)) if line_counts else 0.0
    median_lines = float(np.median(line_counts)) if line_counts else 0.0

    print('\n' + '=' * 40)
    print('Summary')
    print(f'Total images       : {total}')
    print(f'Done               : {done}')
    print(f'Skipped            : {skipped}')
    print(f'Failed             : {failed}')
    print(f'Mean lines/image   : {mean_lines:.2f}')
    print(f'Median lines/image : {median_lines:.2f}')
    print('=' * 40)

    if failed > 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
