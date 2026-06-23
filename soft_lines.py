#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_soft_guidance_line_fields_rgb_keep_roots.py

多核版 reliability-aware soft line 预处理脚本。

在用户原始 prepare_soft_guidance_lines_v3_mp.py 基础上修改：
  1) 保留 --pkl_root 递归读取训练/验证 pkl line 的逻辑；
  2) 保留 --line_root 递归读取测试 jpg/png line 的逻辑；
  3) 二者可共存，输出继续保持相对目录结构；
  4) 默认输出 RGB 三通道 soft field，供 3-channel / dual-branch ProPainter 使用：
       R = center_line
       G = soft_line * soft_weight   （默认沿用原脚本 soft_weight 语义）
       B = reliability
  5) 可通过 --output_mode gray 切回原来的单通道 final_line 输出：
       final_line = center + soft_weight * soft * reliability
  6) 可通过 --rgb_soft_raw 让 G 通道保存未乘 soft_weight 的原始 soft_line；
  7) 可通过 --trusted_response_only 或 --output_mode trusted 只输出 Trusted Response：
       trusted_response = soft_line * reliability

建议：
  - 训练/推理 3ch reliability-aware ProPainter 时使用默认 --output_mode rgb。
  - 若想复现旧单通道 soft_line，用 --output_mode gray。
"""

import argparse
import pickle
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def _shim_numpy_core_for_pickle():
    try:
        import numpy as np
        import numpy.core as ncore
        sys.modules.setdefault('numpy._core', ncore)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        sys.modules.setdefault('numpy._core._multiarray_umath', np.core._multiarray_umath)
        sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    except Exception:
        pass


def load_lines_from_pkl(pkl_path: Path):
    _shim_numpy_core_for_pickle()
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'lines' in data:
        lines = data['lines']
        scores = data.get('scores', None)
    else:
        lines = data
        scores = None
    if hasattr(lines, 'tolist'):
        lines = lines.tolist()
    if hasattr(scores, 'tolist'):
        scores = scores.tolist()
    if not isinstance(lines, (list, tuple)):
        raise ValueError(f'[{pkl_path}] Unsupported lines type: {type(lines)}')
    cleaned, cleaned_scores = [], []
    for i, ln in enumerate(lines):
        if hasattr(ln, 'tolist'):
            ln = ln.tolist()
        if isinstance(ln, (list, tuple)) and len(ln) == 4:
            x1, y1, x2, y2 = ln
            cleaned.append((float(x1), float(y1), float(x2), float(y2)))
            if scores is not None and i < len(scores):
                try:
                    cleaned_scores.append(float(scores[i]))
                except Exception:
                    cleaned_scores.append(1.0)
            else:
                cleaned_scores.append(1.0)
    return cleaned, cleaned_scores


def render_pkl_lines_to_gray(lines, scores=None, size=256, line_width=1.0,
                             swap_xy=True, invert_y=True, use_scores=False, score_power=1.0):
    H = W = int(size)
    canvas = np.zeros((H, W), dtype=np.uint8)
    if not lines:
        return canvas
    max_v = 0.0
    for x1, y1, x2, y2 in lines:
        max_v = max(max_v, abs(x1), abs(y1), abs(x2), abs(y2))
    normalized = max_v <= 1.5
    thickness = max(1, int(round(line_width)))
    if scores is None:
        scores = [1.0] * len(lines)
    for (x1, y1, x2, y2), sc in zip(lines, scores):
        if normalized:
            x1 *= (W - 1); y1 *= (H - 1); x2 *= (W - 1); y2 *= (H - 1)
        if swap_xy:
            px1, py1, px2, py2 = y1, x1, y2, x2
        else:
            px1, py1, px2, py2 = x1, y1, x2, y2
        if invert_y:
            py1 = (H - 1) - py1
            py2 = (H - 1) - py2
        val = int(np.clip((float(sc) ** score_power) * 255.0, 1, 255)) if use_scores else 255
        cv2.line(canvas, (int(round(px1)), int(round(py1))), (int(round(px2)), int(round(py2))),
                 color=val, thickness=thickness, lineType=cv2.LINE_AA)
    return canvas


def read_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return img


def write_gray(path: Path, arr01):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), arr):
        raise RuntimeError(f'Failed to write image: {path}')


def write_rgb(path: Path, rgb01):
    """Write RGB [H,W,3] in [0,1].

    cv2.imwrite expects BGR order, so explicitly convert from RGB to BGR.
    This guarantees that PIL/Image.open(...).convert('RGB') reads channels as:
      R=center, G=soft, B=reliability.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f'RGB output must be [H,W,3], got {rgb.shape}')
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f'Failed to write RGB image: {path}')



HEATMAP_COLORMAPS = {
    'turbo': getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET),
    'jet': cv2.COLORMAP_JET,
    'viridis': getattr(cv2, 'COLORMAP_VIRIDIS', cv2.COLORMAP_JET),
    'plasma': getattr(cv2, 'COLORMAP_PLASMA', cv2.COLORMAP_JET),
    'inferno': getattr(cv2, 'COLORMAP_INFERNO', cv2.COLORMAP_JET),
    'magma': getattr(cv2, 'COLORMAP_MAGMA', cv2.COLORMAP_JET),
}


def write_heatmap(path: Path, arr01, colormap='turbo'):
    """Write a paper-visualization heatmap for a single-channel [0,1] map.

    This is only for visualization figures. The numeric map is converted to an
    8-bit heatmap using OpenCV colormaps. Low responses are dark purple/blue
    and high responses are yellow/red for the default turbo colormap.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)
    cmap = HEATMAP_COLORMAPS.get(str(colormap).lower(), HEATMAP_COLORMAPS['turbo'])
    heat_bgr = cv2.applyColorMap(arr, cmap)
    if not cv2.imwrite(str(path), heat_bgr):
        raise RuntimeError(f'Failed to write heatmap: {path}')

def normalize_soft_line(line_gray, p_low=5, p_high=99):
    x = line_gray.astype(np.float32) / 255.0
    lo, hi = np.percentile(x, p_low), np.percentile(x, p_high)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


def hysteresis_region(soft, strong_thr, weak_thr):
    strong = (soft >= strong_thr).astype(np.uint8)
    weak = (soft >= weak_thr).astype(np.uint8)
    _, labels = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros_like(weak, dtype=np.uint8)
    for lab in set(np.unique(labels[strong > 0]).tolist()):
        if lab != 0:
            keep[labels == lab] = 1
    return strong.astype(np.float32), keep.astype(np.float32)


def thin_centerline(region):
    binary = (region > 0).astype(np.uint8) * 255
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        return (cv2.ximgproc.thinning(binary) > 0).astype(np.float32)
    img, skel = binary.copy(), np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max(img.shape) * 2):
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.float32)


def compute_reliability(soft, strong, sigma=3.0):
    strong_u8 = (strong > 0).astype(np.uint8)
    if strong_u8.max() == 0:
        return np.zeros_like(soft, dtype=np.float32)
    dist = cv2.distanceTransform(1 - strong_u8, cv2.DIST_L2, 3)
    return np.clip(soft * np.exp(-dist / max(float(sigma), 1e-6)), 0.0, 1.0).astype(np.float32)


def make_boundary_band(mask_gray, width):
    mask = (mask_gray > 127).astype(np.uint8)
    if width <= 0:
        return np.zeros_like(mask, dtype=np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return ((cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)) > 0).astype(np.float32)


def suppress_mask_boundary(soft, rel, mask_gray, boundary_width, boundary_decay):
    if mask_gray is None or boundary_width <= 0:
        return soft, rel
    if mask_gray.shape[:2] != soft.shape[:2]:
        mask_gray = cv2.resize(mask_gray, (soft.shape[1], soft.shape[0]), interpolation=cv2.INTER_NEAREST)
    band = make_boundary_band(mask_gray, boundary_width)
    if band.max() > 0:
        factor = 1.0 - band * (1.0 - float(np.clip(boundary_decay, 0.0, 1.0)))
        soft, rel = soft * factor, rel * factor
    return soft, rel


def temporal_smooth(fields, alpha):
    if alpha <= 0 or len(fields) <= 1:
        return fields
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out, n = [], len(fields)
    for i in range(n):
        acc, norm = (1.0 - alpha) * fields[i], (1.0 - alpha)
        if i - 1 >= 0:
            acc += 0.5 * alpha * fields[i - 1]; norm += 0.5 * alpha
        if i + 1 < n:
            acc += 0.5 * alpha * fields[i + 1]; norm += 0.5 * alpha
        out.append(np.clip(acc / max(norm, 1e-6), 0.0, 1.0).astype(np.float32))
    return out


def final_line_map(center, soft, rel, soft_weight):
    return np.clip(center.astype(np.float32) + float(soft_weight) * soft.astype(np.float32) * rel.astype(np.float32), 0, 1)


def trusted_response_map(soft, rel):
    """Trusted response used for visualization: S * R.

    S is the temporally smoothed soft response, and R is the temporally
    smoothed reliability map. This output is intended for paper/debug
    visualization only; the normal 3-channel ProPainter input remains
    [center, soft, reliability].
    """
    return np.clip(soft.astype(np.float32) * rel.astype(np.float32), 0.0, 1.0)


def rgb_line_field(center, soft, rel, soft_weight=0.35, rgb_soft_raw=False):
    """Build RGB reliability-aware line field.

    R: center line / strong linked centerline.
    G: soft response field. By default multiplied by soft_weight to preserve the
       old script's --soft_weight behavior; pass --rgb_soft_raw to keep raw soft.
    B: reliability field.
    """
    center_ch = np.clip(center.astype(np.float32), 0.0, 1.0)
    soft_ch = np.clip(soft.astype(np.float32), 0.0, 1.0)
    if not rgb_soft_raw:
        soft_ch = np.clip(float(soft_weight) * soft_ch, 0.0, 1.0)
    rel_ch = np.clip(rel.astype(np.float32), 0.0, 1.0)
    return np.stack([center_ch, soft_ch, rel_ch], axis=-1).astype(np.float32)


def find_mask(mask_root, rel):
    if mask_root is None:
        return None
    p = mask_root / rel
    if p.exists() and p.is_file():
        return p
    parent = mask_root / rel.parent
    for ext in IMG_EXTS:
        q = parent / f'{rel.stem}{ext}'
        if q.exists() and q.is_file():
            return q
    return None


def relative_pkl_files(root):
    return sorted([p.relative_to(root) for p in root.rglob('*.pkl') if p.is_file()])


def relative_image_files(root):
    return sorted([p.relative_to(root) for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS])


def group_by_parent(rel_paths):
    groups = {}
    for rel in rel_paths:
        groups.setdefault(rel.parent, []).append(rel)
    return {k: sorted(v) for k, v in groups.items()}


def chunk_list(xs, chunk_size):
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


def process_items(items, out_root, mask_root, p):
    rels, softs, centers, rels_map = [], [], [], []
    for rel, raw in items:
        mask_path = find_mask(mask_root, rel)
        mask_gray = read_gray(mask_path) if mask_path is not None else None
        soft = normalize_soft_line(raw, p['p_low'], p['p_high'])
        strong, linked = hysteresis_region(soft, p['strong_thr'], p['weak_thr'])
        center = thin_centerline(linked)
        rel_map = compute_reliability(soft, strong, p['reliability_sigma'])
        soft, rel_map = suppress_mask_boundary(soft, rel_map, mask_gray, p['boundary_width'], p['boundary_decay'])
        rels.append(rel); softs.append(soft); centers.append(center); rels_map.append(rel_map)

    softs = temporal_smooth(softs, p['temporal_alpha'])
    rels_map = temporal_smooth(rels_map, p['temporal_alpha'])

    for rel, soft, center, rel_map in zip(rels, softs, centers, rels_map):
        final = final_line_map(center, soft, rel_map, p['soft_weight'])
        trusted = trusted_response_map(soft, rel_map)
        out_path = out_root / rel.with_suffix('.png')

        if p['output_mode'] == 'gray':
            write_gray(out_path, final)
        elif p['output_mode'] == 'trusted':
            # Numeric Trusted Response S*R as grayscale.
            write_gray(out_path, trusted)
        elif p['output_mode'] == 'trusted_heat':
            # Paper-visualization heatmap of Trusted Response S*R.
            # This is the mode used by --trusted_response_only.
            write_heatmap(out_path, trusted, p['heatmap_colormap'])
        elif p['output_mode'] == 'soft_heat':
            # Paper-visualization heatmap of the soft response S only.
            write_heatmap(out_path, soft, p['heatmap_colormap'])
        else:
            field = rgb_line_field(center, soft, rel_map, p['soft_weight'], p['rgb_soft_raw'])
            write_rgb(out_path, field)
            if p.get('save_final_gray', False):
                write_gray(out_root / 'final_gray' / rel.with_suffix('.png'), final)

        # When --trusted_response_only is enabled, keep the output directory clean:
        # only the trusted response maps are written.
        if p['save_debug'] and not p.get('trusted_response_only', False):
            dbg = out_root / 'line_debug' / rel.parent
            stem = rel.stem
            write_gray(dbg / f'{stem}_soft.png', soft)
            write_gray(dbg / f'{stem}_center.png', center)
            write_gray(dbg / f'{stem}_reliability.png', rel_map)
            write_gray(dbg / f'{stem}_trusted.png', trusted)
            write_gray(dbg / f'{stem}_final.png', final)
            if p['output_mode'] == 'rgb':
                field = rgb_line_field(center, soft, rel_map, p['soft_weight'], p['rgb_soft_raw'])
                write_rgb(dbg / f'{stem}_rgb_field.png', field)


def worker(task):
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    mode, root = task['mode'], Path(task['root'])
    out_root = Path(task['out_root'])
    mask_root = Path(task['mask_root']) if task['mask_root'] else None
    p = task['params']
    items = []
    for rel_s in task['rels']:
        rel = Path(rel_s)
        if mode == 'pkl':
            lines, scores = load_lines_from_pkl(root / rel)
            raw = render_pkl_lines_to_gray(lines, scores, p['size'], p['line_width'], p['swap_xy'],
                                           p['invert_y'], p['use_scores'], p['score_power'])
        else:
            raw = read_gray(root / rel)
        items.append((rel, raw))
    process_items(items, out_root, mask_root, p)
    return mode, len(task['rels']), task['rels'][0] if task['rels'] else ''


def build_tasks(mode, root, rels, out_root, mask_root, params, chunk_single_sequence, chunk_size):
    groups = group_by_parent(rels)
    if len(groups) == 1 and chunk_single_sequence:
        only = next(iter(groups.values()))
        groups = {Path(f'chunk_{i:06d}'): c for i, c in enumerate(chunk_list(only, chunk_size))}
    return [{
        'mode': mode,
        'rels': [str(r) for r in group_rels],
        'root': str(root),
        'out_root': str(out_root),
        'mask_root': str(mask_root) if mask_root is not None else None,
        'params': params,
    } for _, group_rels in sorted(groups.items(), key=lambda kv: str(kv[0]))]


def run_tasks(tasks, num_workers):
    total = 0
    if num_workers <= 1:
        for i, task in enumerate(tasks, 1):
            mode, cnt, first = worker(task)
            total += cnt
            print(f'[INFO] done {i}/{len(tasks)} {mode}: count={cnt}, first={first}')
        return total
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            mode, cnt, first = fut.result()
            total += cnt
            print(f'[INFO] done {i}/{len(tasks)} {mode}: count={cnt}, first={first}')
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_root', default=None)
    parser.add_argument('--line_root', default=None)
    parser.add_argument('--mask_root', default=None)
    parser.add_argument('-o', '--out_root', required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--line_width', type=float, default=1.0)
    parser.add_argument('--swap_xy', action='store_true', default=True)
    parser.add_argument('--no_swap_xy', dest='swap_xy', action='store_false')
    parser.add_argument('--invert_y', action='store_true', default=True)
    parser.add_argument('--no_invert_y', dest='invert_y', action='store_false')
    parser.add_argument('--use_scores', action='store_true')
    parser.add_argument('--score_power', type=float, default=1.0)
    parser.add_argument('--p_low', type=float, default=5.0)
    parser.add_argument('--p_high', type=float, default=99.0)
    parser.add_argument('--strong_thr', type=float, default=0.45)
    parser.add_argument('--weak_thr', type=float, default=0.15)
    parser.add_argument('--soft_weight', type=float, default=0.35)
    parser.add_argument('--reliability_sigma', type=float, default=3.0)
    parser.add_argument('--boundary_width', type=int, default=5)
    parser.add_argument('--boundary_decay', type=float, default=0.20)
    parser.add_argument('--temporal_alpha', type=float, default=0.15)
    parser.add_argument('--save_debug', action='store_true')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--chunk_single_sequence', action='store_true')
    parser.add_argument('--chunk_size', type=int, default=256)
    parser.add_argument('--opencv_threads', type=int, default=1)

    # New options for 3-channel reliability-aware ProPainter.
    parser.add_argument('--output_mode', choices=['rgb', 'gray', 'trusted', 'trusted_heat', 'soft_heat'], default='rgb',
                        help='rgb: output RGB field [center, soft, reliability]; gray: old final_line output; trusted: grayscale S*R; trusted_heat: heatmap S*R; soft_heat: heatmap S.')
    parser.add_argument('--trusted_response_only', action='store_true',
                        help='Shortcut for --output_mode trusted_heat. Write only heatmap-style Trusted Response S*R and suppress debug outputs.')
    parser.add_argument('--heatmap_colormap', choices=['turbo', 'jet', 'viridis', 'plasma', 'inferno', 'magma'], default='turbo',
                        help='Colormap for trusted_heat/soft_heat visualization outputs. Default: turbo.')
    parser.add_argument('--rgb_soft_raw', action='store_true',
                        help='For RGB output, store raw soft line in G channel instead of soft_weight * soft.')
    parser.add_argument('--save_final_gray', action='store_true',
                        help='When output_mode=rgb, also save old final_line gray map under out_root/final_gray/.')
    args = parser.parse_args()

    if args.trusted_response_only:
        args.output_mode = 'trusted_heat'
        args.save_debug = False

    if args.pkl_root is None and args.line_root is None:
        raise RuntimeError('At least one of --pkl_root or --line_root must be provided.')
    if args.chunk_single_sequence and args.temporal_alpha > 0:
        print('[WARN] --chunk_single_sequence with --temporal_alpha > 0: smoothing is only inside chunks.')
    try:
        cv2.setNumThreads(max(1, int(args.opencv_threads)))
    except Exception:
        pass

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    mask_root = Path(args.mask_root) if args.mask_root else None
    params = vars(args).copy()
    params.pop('pkl_root'); params.pop('line_root'); params.pop('mask_root'); params.pop('out_root')
    params.pop('num_workers'); params.pop('chunk_single_sequence'); params.pop('chunk_size'); params.pop('opencv_threads')

    tasks = []
    if args.pkl_root:
        root = Path(args.pkl_root)
        rels = relative_pkl_files(root)
        print(f'[INFO] Found PKL files: {len(rels)} under {root}')
        tasks += build_tasks('pkl', root, rels, out_root, mask_root, params, args.chunk_single_sequence, args.chunk_size)
    if args.line_root:
        root = Path(args.line_root)
        rels = relative_image_files(root)
        print(f'[INFO] Found line image files: {len(rels)} under {root}')
        tasks += build_tasks('line', root, rels, out_root, mask_root, params, args.chunk_single_sequence, args.chunk_size)

    print(f'[INFO] Total tasks: {len(tasks)}')
    print(f'[INFO] num_workers: {args.num_workers}')
    print(f'[INFO] output_mode: {args.output_mode}')
    if args.output_mode == 'rgb':
        print('[INFO] RGB channels: R=center_line, G=soft_line%s, B=reliability' %
              (' (raw)' if args.rgb_soft_raw else f' * soft_weight({args.soft_weight})'))
    elif args.output_mode == 'trusted':
        print('[INFO] Trusted Response output: S * R, saved as grayscale PNG.')
    elif args.output_mode == 'trusted_heat':
        print(f'[INFO] Trusted Response output: heatmap(S * R), colormap={args.heatmap_colormap}.')
    elif args.output_mode == 'soft_heat':
        print(f'[INFO] Soft Response output: heatmap(S), colormap={args.heatmap_colormap}.')
    total = run_tasks(tasks, args.num_workers)
    if args.output_mode == 'rgb':
        kind = 'RGB line fields'
    elif args.output_mode == 'trusted':
        kind = 'trusted response grayscale maps'
    elif args.output_mode == 'trusted_heat':
        kind = 'trusted response heatmaps'
    elif args.output_mode == 'soft_heat':
        kind = 'soft response heatmaps'
    else:
        kind = 'gray final line maps'
    print(f'[DONE] Wrote {total} processed {kind} to: {out_root}')


if __name__ == '__main__':
    main()
