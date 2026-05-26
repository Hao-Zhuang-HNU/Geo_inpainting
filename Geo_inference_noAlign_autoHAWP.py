# -*- coding: utf-8 -*-
"""
Geo no-align inference with internal LSM-HAWP extraction.

Main changes compared with Geo_inference_noAlign.py:
1) Remove --test_line_list from user-facing inference.
2) Build masked RGB inputs internally from --image_url and --mask_url using natural-order matching.
3) Run LSM-HAWP on the masked RGB inputs to generate temporary pkl wireframes.
4) Feed masked RGB + extracted pkl wireframes into ContinuousEdgeLineDatasetMask.

Assumptions:
- mask white / non-zero region is the hole to be completed.
- --image_url and --mask_url can be either txt lists or directories.
- the generated HAWP pkl is used as the observed structure, not GT structure.
"""

import argparse
import hashlib
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

from datasets.dataset_Geo import ContinuousEdgeLineDatasetMask
from src.lsm_hawp.lsm_hawp_model import LSM_HAWP
from src.models.TSR_model_RefKV import EdgeLineGPTConfig, EdgeLineGPT256RelBCE
from src.utils_RefKV import set_seed, SampleEdgeLineLogitsWithRefExtraction

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
PKL_EXT = '.pkl'


def natural_key(s):
    s = str(s)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def read_path_list(path_or_txt, exts=None):
    """Read paths from a txt list or recursively from a directory, naturally sorted."""
    if path_or_txt is None:
        return []
    p = Path(path_or_txt)
    if p.is_file() and p.suffix.lower() == '.txt':
        with open(p, 'r', encoding='utf-8') as f:
            items = [line.strip() for line in f if line.strip()]
        return sorted(items, key=natural_key)
    if p.is_dir():
        exts = exts or IMG_EXTS
        items = [str(x) for x in p.rglob('*') if x.is_file() and x.suffix.lower() in exts]
        return sorted(items, key=natural_key)
    if p.is_file():
        return [str(p)]
    raise FileNotFoundError(f'Path does not exist: {path_or_txt}')


def write_list(paths, txt_path):
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for p in paths:
            f.write(str(p) + '\n')


def make_unique_stem(index, src_path, used):
    """Create a stable filename stem while avoiding duplicate basenames."""
    src = Path(src_path)
    stem = src.stem
    if stem not in used:
        used.add(stem)
        return stem
    h = hashlib.sha1(str(src_path).encode('utf-8')).hexdigest()[:8]
    new_stem = f'{index:06d}_{stem}_{h}'
    used.add(new_stem)
    return new_stem


def load_binary_mask(mask_path, target_hw, threshold=127):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'Failed to read mask: {mask_path}')
    h, w = target_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > threshold).astype(np.uint8)
    return mask


def build_masked_images_and_lists(opts):
    """
    Build masked images from GT/clean images and masks.
    The output masked image list is what the Geo dataset will read as --image_url.
    """
    image_paths = read_path_list(opts.image_url, IMG_EXTS)
    mask_paths = read_path_list(opts.mask_url, IMG_EXTS)
    if len(image_paths) == 0:
        raise RuntimeError('No images found from --image_url.')
    if len(mask_paths) == 0:
        raise RuntimeError('No masks found from --mask_url.')
    if len(mask_paths) < len(image_paths) and not opts.repeat_masks:
        raise RuntimeError(
            f'Mask count ({len(mask_paths)}) is smaller than image count ({len(image_paths)}). '
            f'Use --repeat_masks to reuse masks cyclically.'
        )

    cache_root = Path(opts.save_url) / opts.auto_cache_dir
    masked_dir = cache_root / 'masked_images'
    masked_dir.mkdir(parents=True, exist_ok=True)

    masked_image_paths = []
    used_stems = set()

    print(f'[INFO] Building masked inputs: images={len(image_paths)}, masks={len(mask_paths)}, fill={opts.mask_fill}')
    for i, img_path in enumerate(tqdm(image_paths, desc='Masking RGB', unit='img', dynamic_ncols=True)):
        mask_path = mask_paths[i % len(mask_paths)] if opts.repeat_masks else mask_paths[i]

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'Failed to read image: {img_path}')
        h, w = img.shape[:2]
        mask = load_binary_mask(mask_path, (h, w), threshold=opts.mask_threshold)

        stem = make_unique_stem(i, img_path, used_stems)
        out_path = masked_dir / f'{stem}.png'

        if opts.force_preprocess or not out_path.exists():
            masked = img.copy()
            if opts.mask_fill_mode == 'constant':
                fill_value = int(np.clip(opts.mask_fill, 0, 255))
                masked[mask > 0] = (fill_value, fill_value, fill_value)
            elif opts.mask_fill_mode == 'mean':
                known = img[mask == 0]
                fill = known.mean(axis=0).astype(np.uint8) if known.size else np.array([127, 127, 127], dtype=np.uint8)
                masked[mask > 0] = fill
            else:
                raise ValueError(f'Unknown --mask_fill_mode: {opts.mask_fill_mode}')
            cv2.imwrite(str(out_path), masked)

        masked_image_paths.append(str(out_path))

    masked_list = cache_root / 'masked_images.txt'
    write_list(masked_image_paths, masked_list)
    return image_paths, mask_paths, masked_image_paths, str(masked_list), cache_root


def build_pkl_lookup(pkl_root):
    pkl_root = Path(pkl_root)
    lookup = {}
    all_pkls = sorted(pkl_root.rglob('*.pkl'), key=natural_key)
    for f in all_pkls:
        name = f.name
        stem = f.stem
        keys = {name, stem}
        # Compatible with xxx.png.pkl / xxx.jpg.pkl.
        for ext in IMG_EXTS:
            if stem.lower().endswith(ext):
                keys.add(stem[:-len(ext)])
        for k in keys:
            if k and k not in lookup:
                lookup[k] = f
    return lookup, all_pkls


def find_pkl_for_image(img_path, pkl_lookup):
    p = Path(img_path)
    candidates = [
        f'{p.stem}.pkl',
        p.stem,
        f'{p.name}.pkl',
    ]
    for key in candidates:
        if key in pkl_lookup:
            return str(pkl_lookup[key])
    return None


def run_lsm_hawp_on_masked_images(opts, masked_image_paths, cache_root):
    """Run LSM-HAWP internally and return a txt list of generated pkl paths."""
    pkl_root = Path(cache_root) / 'hawp_pkls'
    pkl_root.mkdir(parents=True, exist_ok=True)

    if opts.force_preprocess and pkl_root.exists():
        shutil.rmtree(pkl_root)
        pkl_root.mkdir(parents=True, exist_ok=True)

    existing_lookup, _ = build_pkl_lookup(pkl_root)
    missing_imgs = [p for p in masked_image_paths if find_pkl_for_image(p, existing_lookup) is None]

    if missing_imgs:
        print(f'[INFO] Running LSM-HAWP: missing pkls={len(missing_imgs)} / {len(masked_image_paths)}')
        torch.backends.cudnn.benchmark = True
        model = LSM_HAWP(threshold=opts.hawp_threshold, size=opts.hawp_size)
        state = torch.load(opts.hawp_ckpt_path, map_location='cpu', weights_only=False)
        if 'model' not in state:
            raise KeyError('HAWP checkpoint does not contain key "model".')
        model.lsm_hawp.load_state_dict(state['model'])
        model.lsm_hawp.cuda()
        model.lsm_hawp.eval()

        batch_size = max(1, opts.hawp_batch_size)
        with torch.no_grad():
            for i in tqdm(range(0, len(missing_imgs), batch_size), desc='HAWP extract', unit='batch', dynamic_ncols=True):
                batch = missing_imgs[i:i + batch_size]
                try:
                    model.wireframe_detect(batch, str(pkl_root))
                except Exception as e:
                    print(f'[WARN] HAWP batch failed, fallback to image-by-image. Error: {e}')
                    for one in batch:
                        model.wireframe_detect([one], str(pkl_root))
    else:
        print('[INFO] Reusing existing HAWP pkls from cache.')

    pkl_lookup, all_pkls = build_pkl_lookup(pkl_root)
    print(f'[INFO] HAWP pkl files found: {len(all_pkls)} under {pkl_root}')

    pkl_paths = []
    missing = []
    for img_path in masked_image_paths:
        pkl = find_pkl_for_image(img_path, pkl_lookup)
        if pkl is None:
            missing.append(img_path)
        else:
            pkl_paths.append(pkl)

    if missing:
        log_path = Path(cache_root) / 'missing_pkl_after_auto_hawp.log'
        with open(log_path, 'w', encoding='utf-8') as f:
            for p in missing:
                f.write(str(p) + '\n')
        raise RuntimeError(
            f'HAWP failed to produce {len(missing)} pkl files. Missing list saved to: {log_path}'
        )

    pkl_list = Path(cache_root) / 'hawp_pkls.txt'
    write_list(pkl_paths, pkl_list)
    return str(pkl_list)


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


def prepare_line_input(line, mask, dilate_line=1):
    """Known-region extracted line input. The hole region is removed by mask."""
    _ = dilate_line
    return line * (1 - mask)


def patch_dataset_line_loader_with_dilation(dataset, dilate_line):
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

    print(f'[INFO] Loading Geo checkpoint from {opts.ckpt_path}')
    checkpoint = torch.load(opts.ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)

    clean_state = {}
    for k, v in state_dict.items():
        if 'attn.mask' in k:
            continue
        nk = k.replace('module.', '').replace('_orig_mod.', '')
        clean_state[nk] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(f'[INFO] Geo checkpoint loaded. missing={len(missing)}, unexpected={len(unexpected)}')
    model.cuda().eval()
    return model


def save_pred(prob, observed, mask, out_file, binary_threshold=None):
    """
    Merge predicted hole with observed known-region structure.
    Here `observed` is extracted from the masked input image, not GT structure.
    """
    pred_np = _tensor_to_np(prob)
    if binary_threshold is not None:
        pred_np = (pred_np >= binary_threshold).astype(np.float32)
    obs_np = _tensor_to_np(observed)
    mask_np = _tensor_to_np(mask)
    merged = pred_np * mask_np + obs_np * (1 - mask_np)

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    vis = (np.stack([merged] * 3, axis=-1) * 255).astype(np.uint8)
    cv2.imwrite(str(out_file), vis[:, :, ::-1])

    return torch.from_numpy(merged).unsqueeze(0).unsqueeze(0).cuda()


def geo_inference(opts):
    """Inference without npz alignment; local/global references use generated structure only."""
    _, _, masked_image_paths, masked_image_list, cache_root = build_masked_images_and_lists(opts)
    auto_line_list = run_lsm_hawp_on_masked_images(opts, masked_image_paths, cache_root)

    model = build_model(opts)

    dataset = ContinuousEdgeLineDatasetMask(
        masked_image_list,
        test_mask_path=opts.mask_url,
        is_train=False,
        image_size=opts.image_size,
        line_path=auto_line_list,
    )
    patch_dataset_line_loader_with_dilation(dataset, opts.dilate_line)

    print(f'[INFO] Inference frames: {len(dataset)} | line_source=auto_hawp | dilate_line={opts.dilate_line}')

    # ---- Frame 0: no GT line; initialize from extracted known-region structure ----
    first = dataset[0]
    name0 = os.path.basename(first['name'][0] if isinstance(first['name'], list) else first['name'])
    if not name0.lower().endswith(('.png', '.jpg', '.jpeg')):
        name0 += '.png'

    img0 = _ensure_4d(first['img']).cuda()
    mask0 = _ensure_4d(first['mask']).cuda()
    obs_e0 = _ensure_4d(first['edge']).cuda()
    obs_l0 = _ensure_4d(first['line']).cuda()

    with torch.no_grad():
        edge0, line0, _ = SampleEdgeLineLogitsWithRefExtraction(
            model,
            context=[img0, obs_e0 * (1 - mask0), prepare_line_input(obs_l0, mask0, opts.dilate_line)],
            mask=mask0,
            iterations=opts.iterations,
            extract_ref=False,
        )

    pred_edge_prev = save_pred(edge0[0], obs_e0, mask0, os.path.join(opts.save_url, 'edge', name0))
    pred_line_prev = save_pred(
        line0[0],
        obs_l0,
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
    for i in tqdm(range(1, len(dataset)), desc='Geo inference', unit='frame', dynamic_ncols=True):
        item = dataset[i]
        name = os.path.basename(item['name'][0] if isinstance(item['name'], list) else item['name'])
        if not name.lower().endswith(('.png', '.jpg', '.jpeg')):
            name += '.png'

        img = _ensure_4d(item['img']).cuda()
        mask = _ensure_4d(item['mask']).cuda()
        obs_e = _ensure_4d(item['edge']).cuda()
        obs_l = _ensure_4d(item['line']).cuda()

        local_edge = dilate_tensor(pred_edge_prev.clone(), opts.ref_dilate)
        local_line = dilate_tensor(pred_line_prev.clone(), opts.ref_dilate)

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
                context=[img, obs_e * (1 - mask), prepare_line_input(obs_l, mask, opts.dilate_line)],
                mask=mask,
                iterations=opts.iterations,
                ref_feat=ref_feat,
                extract_ref=False,
            )

        pred_edge_prev = save_pred(edge_pred[0], obs_e, mask, os.path.join(opts.save_url, 'edge', name))
        pred_line_prev = save_pred(
            line_pred[0],
            obs_l,
            mask,
            os.path.join(opts.save_url, 'line', name),
            binary_threshold=opts.line_binary_thresh if opts.solid_line else None,
        )

    if not opts.keep_auto_cache:
        cache_root = Path(cache_root)
        # Keep the text lists for reproducibility, remove heavy masked images/pkls only when explicitly allowed.
        # By default we keep cache because it is useful for debugging and reruns.
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='Geo no-align inference with internal masked-image LSM-HAWP extraction.'
    )
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Geo model checkpoint path.')
    parser.add_argument('--hawp_ckpt_path', type=str, default='./ckpt/best_lsm_hawp.pth', help='best_lsm_hawp.pth path.')
    parser.add_argument('--image_url', type=str, required=True, help='GT/clean image list or image directory. It will be masked internally.')
    parser.add_argument('--mask_url', type=str, required=True, help='Mask list or mask directory. White/non-zero means hole.')

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--save_url', type=str, default='./results_geo_auto_hawp')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--ref_dilate', type=int, default=3)
    parser.add_argument('--dilate_line', type=int, default=1, choices=[1, 3],
                        help='Optional line dilation size to match training strategy.')
    parser.add_argument('--solid_line', action='store_true',
                        help='If set, binarize generated line output to get solid lines.')
    parser.add_argument('--line_binary_thresh', type=float, default=0.5,
                        help='Threshold used when --solid_line is enabled.')

    # Masked-input construction.
    parser.add_argument('--mask_threshold', type=int, default=127,
                        help='Mask pixels > threshold are treated as holes.')
    parser.add_argument('--mask_fill_mode', type=str, default='constant', choices=['constant', 'mean'],
                        help='How to fill holes before HAWP extraction and Geo RGB input.')
    parser.add_argument('--mask_fill', type=int, default=0,
                        help='Constant fill value for hole region when --mask_fill_mode constant. 255=white, 0=black.')
    parser.add_argument('--repeat_masks', action='store_true',
                        help='If masks are fewer than images, reuse masks cyclically by natural order.')

    # Internal HAWP extraction.
    parser.add_argument('--hawp_threshold', type=float, default=0.8,
                        help='LSM_HAWP line confidence threshold.')
    parser.add_argument('--hawp_size', type=int, default=512,
                        help='LSM_HAWP input size.')
    parser.add_argument('--hawp_batch_size', type=int, default=16,
                        help='Batch size for LSM-HAWP wireframe_detect.')
    parser.add_argument('--auto_cache_dir', type=str, default='__auto_hawp__',
                        help='Cache directory under --save_url for masked images and generated pkls.')
    parser.add_argument('--force_preprocess', action='store_true',
                        help='Regenerate masked images and HAWP pkls even if cache exists.')
    parser.add_argument('--keep_auto_cache', action='store_true', default=True,
                        help='Keep masked-image and HAWP-pkl cache for debugging/reuse. Enabled by default.')

    return parser.parse_args()


if __name__ == '__main__':
    set_seed(42)
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids

    os.makedirs(os.path.join(args.save_url, 'edge'), exist_ok=True)
    os.makedirs(os.path.join(args.save_url, 'line'), exist_ok=True)

    geo_inference(args)
