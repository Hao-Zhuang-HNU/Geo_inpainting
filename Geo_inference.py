# -*- coding: utf-8 -*-
import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.dataset_TSR import ContinuousEdgeLineDatasetMask
from src.models.TSR_model_RefKV import EdgeLineGPTConfig, EdgeLineGPT256RelBCE
from src.utils_RefKV import set_seed, SampleEdgeLineLogitsWithRefExtraction


def load_file_list(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


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
    """Prepare known-region GT line input with optional dilation for train/infer consistency."""
    gt_line_in = dilate_tensor(gt_line, dilate_line) if dilate_line > 1 else gt_line
    return gt_line_in * (1 - mask)


def warp_tensor_from_npz(prev_tensor, npz_path, size=256, device='cuda'):
    if prev_tensor is None:
        return None

    default_res = torch.zeros_like(prev_tensor)
    if (not npz_path) or (not os.path.exists(npz_path)):
        return default_res

    try:
        data = np.load(npz_path)
        if 'valid' in data and not bool(data['valid']):
            return default_res
        if 'homography' not in data:
            return default_res
        H_mat = data['homography']

        prev_np = prev_tensor.squeeze().detach().cpu().numpy()
        warped_np = cv2.warpPerspective(prev_np, H_mat, (size, size), flags=cv2.INTER_LINEAR)
        return torch.from_numpy(warped_np).unsqueeze(0).unsqueeze(0).to(device)
    except Exception as e:
        print(f"[Warp Error] {e} for {npz_path}")
        return default_res


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

    print(f"Loading checkpoint from {opts.ckpt_path}")
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


def save_pred(prob, gt, mask, out_file):
    pred_np = _tensor_to_np(prob)
    gt_np = _tensor_to_np(gt)
    mask_np = _tensor_to_np(mask)
    merged = pred_np * mask_np + gt_np * (1 - mask_np)

    vis = (np.stack([merged] * 3, axis=-1) * 255).astype(np.uint8)
    cv2.imwrite(out_file, vis[:, :, ::-1])

    return torch.from_numpy(merged).unsqueeze(0).unsqueeze(0).cuda()


def geo_inference(opts):
    if not opts.npz_list:
        raise ValueError('--npz_list is required for Geo_inference.')

    model = build_model(opts)
    npz_paths = load_file_list(opts.npz_list)

    dataset = ContinuousEdgeLineDatasetMask(
        opts.image_url,
        test_mask_path=opts.mask_url,
        is_train=False,
        image_size=opts.image_size,
        line_path=opts.test_line_list,
    )

    print(f'Inference frames: {len(dataset)}')

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
    pred_line_prev = save_pred(line0[0], gt_l0, mask0, os.path.join(opts.save_url, 'line', name0))

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

        npz_path = npz_paths[i] if i < len(npz_paths) else ''
        local_edge = warp_tensor_from_npz(pred_edge_prev, npz_path, size=opts.image_size)
        local_line = warp_tensor_from_npz(pred_line_prev, npz_path, size=opts.image_size)

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
        pred_line_prev = save_pred(line_pred[0], gt_l, mask, os.path.join(opts.save_url, 'line', name))


if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--image_url', type=str, default=None)
    parser.add_argument('--mask_url', type=str, default=None)
    parser.add_argument('--test_line_list', type=str, default='')
    parser.add_argument('--npz_list', type=str, required=True)

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--save_url', type=str, default='./results_geo')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--ref_dilate', type=int, default=3)
    parser.add_argument('--dilate_line', type=int, default=1, choices=[1, 3],
                        help='Optional GT line dilation size to match training strategy.')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids

    os.makedirs(os.path.join(args.save_url, 'edge'), exist_ok=True)
    os.makedirs(os.path.join(args.save_url, 'line'), exist_ok=True)

    geo_inference(args)
