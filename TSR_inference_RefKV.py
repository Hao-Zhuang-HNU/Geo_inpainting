# -*- coding: utf-8 -*-
import argparse
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

from datasets.dataset_Geo import ContinuousEdgeLineDatasetMask
# 用于标准模式 (Backup)
from src.models.TSR_model import EdgeLineGPTConfig as StandardConfig
from src.models.TSR_model import EdgeLineGPT256RelBCE as StandardModel
# 用于 RefKV 模式
from src.models.Geo_model import EdgeLineGPTConfig as RefKVConfig
from src.models.Geo_model import EdgeLineGPT256RelBCE as RefKVModel

from src.utils_RefKV import set_seed, SampleEdgeLineLogitsWithRefExtraction, SampleEdgeLineLogits_Standard
from src.eval_metrics import compute_all_metrics

from skimage.morphology import skeletonize

# ==========================================
#  Helper Functions
# ==========================================

def load_file_list(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def dilate_tensor(x, kernel_size=3):
    if kernel_size <= 1:
        return x
    padding = kernel_size // 2
    return F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)

def warp_tensor_from_npz(prev_tensor, npz_path, size=256, device='cuda'):
    if prev_tensor is None:
        return None
    default_res = torch.zeros_like(prev_tensor)
    if not npz_path or not os.path.exists(npz_path):
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
        warped_tensor = torch.from_numpy(warped_np).unsqueeze(0).unsqueeze(0).to(device)
        return warped_tensor
    except Exception as e:
        print(f"[Warp Error] {e} for {npz_path}")
        return default_res

def thin_lines(prob_map, threshold=0.3):
    binary = prob_map > threshold
    thinned = skeletonize(binary)
    return thinned.astype(np.float32)

def _tensor_to_np(t):
    if isinstance(t, torch.Tensor):
        return np.squeeze(t.detach().cpu().numpy())
    return np.squeeze(np.asarray(t))

def _ensure_4d(t):
    if t.dim() == 3: return t.unsqueeze(0)
    if t.dim() == 2: return t.unsqueeze(0).unsqueeze(0)
    return t

def standard_inference(opts):
    """
    完全模仿 TSR_inference_backup.py，使用原始模型类
    """
    print("=" * 60)
    print("Running STANDARD inference (Using Original Model Class)")
    print("=" * 60)
    
    # 1. 使用原始模型配置和类 (从 src.models.TSR_model 导入)
    model_config = StandardConfig(
        embd_pdrop=0.0, resid_pdrop=0.0, n_embd=opts.n_embd, 
        block_size=32, attn_pdrop=0.0, n_layer=opts.n_layer, n_head=opts.n_head
    )
    model = StandardModel(model_config)
    
    print(f"Loading checkpoint from {opts.ckpt_path}...")
    checkpoint = torch.load(opts.ckpt_path)
    
    # 还原权重加载逻辑
    if opts.ckpt_path.endswith('.pt'):
        state_dict = checkpoint
    else:
        state_dict = checkpoint.get('model', checkpoint)
    
    # 移除 DDP 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "").replace("_orig_mod.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict) 
    model.cuda()
    model.eval()
    
    test_dataset = ContinuousEdgeLineDatasetMask(
        opts.image_url, test_mask_path=opts.mask_url, is_train=False,
        image_size=opts.image_size, line_path=opts.test_line_list
    )
    
    edge_metric_sum = defaultdict(float)
    line_metric_sum = defaultdict(float)
    eval_count = 0
    
    for it in tqdm(range(len(test_dataset))):
        items = test_dataset[it]
        
        # --- 修复 OpenCV 写入名 ---
        raw_name = items['name']
        if isinstance(raw_name, list): raw_name = raw_name[0]
        pure_name = os.path.basename(raw_name)
        if not pure_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            pure_name += '.png'
        
        curr_img = _ensure_4d(items['img']).cuda()
        curr_mask = _ensure_4d(items['mask']).cuda()
        curr_gt_edge = _ensure_4d(items['edge']).cuda()
        curr_gt_line = _ensure_4d(items['line']).cuda()
        
        with torch.no_grad():
            # 使用同步自 backup 的采样函数
            edge_prob, line_prob = SampleEdgeLineLogits_Standard(
                model,
                context=[curr_img, curr_gt_edge, curr_gt_line],
                mask=curr_mask,
                iterations=opts.iterations
            )
        
        edge_pred_single = edge_prob[0, ...].cpu()
        line_pred_single = line_prob[0, ...].cpu()
        
        if opts.eval_metrics:
            p_edge, g_edge = _tensor_to_np(edge_pred_single), _tensor_to_np(items['edge'])
            p_line, g_line = _tensor_to_np(line_pred_single), _tensor_to_np(items['line'])
            mask_np = _tensor_to_np(items['mask'])
            e_met = compute_all_metrics(p_edge, g_edge, mask=mask_np, use_reproj=False)
            l_met = compute_all_metrics(p_line, g_line, mask=mask_np, use_reproj=False)
            for k, v in e_met.items(): edge_metric_sum[k] += v
            for k, v in l_met.items(): line_metric_sum[k] += v
            eval_count += 1

        # 混合结果
        edge_output = edge_pred_single * items['mask'] + items['edge'] * (1 - items['mask'])
        line_output = line_pred_single * items['mask'] + items['line'] * (1 - items['mask'])
        
        if opts.thin:
            edge_output = torch.from_numpy(thin_lines(_tensor_to_np(edge_output)))
            line_output = torch.from_numpy(thin_lines(_tensor_to_np(line_output)))

        # 转换并保存
        edge_save = (edge_output.repeat(3, 1, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        line_save = (line_output.repeat(3, 1, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(opts.save_url, 'edge', pure_name), edge_save[:, :, ::-1])
        cv2.imwrite(os.path.join(opts.save_url, 'line', pure_name), line_save[:, :, ::-1])

    if opts.eval_metrics and eval_count > 0:
        print_metrics_report("STANDARD MODE", edge_metric_sum, line_metric_sum, eval_count)

def print_metrics_report(mode_name, e_sum, l_sum, count):
    print("\n" + "#" * 60)
    print(f"  {mode_name} REPORT (Frames: {count})")
    print("#" * 60)
    for title, m_dict in [("EDGE", e_sum), ("LINE", l_sum)]:
        print(f"\n--- {title} METRICS ---")
        for k in sorted(m_dict.keys()):
            print(f"  {k:<15}: {m_dict[k] / count:.6f}")
    print("#" * 60 + "\n")

def refkv_inference(opts):
    """
    Ref-KV 模式推理：
    1. 第一帧提取 Global Reference Features。
    2. 后续帧通过 NPZ 单应性矩阵进行 Warp，提取 Local Reference Features。
    3. 结合 Global + Local 特征进行引导推理。
    """
    print("=" * 60)
    print("Running Ref-KV inference with T-1 Propagation & Geometric Warping")
    if opts.local_GT:
        print("!!! Teacher Forcing Enabled: Using Previous Frame GT as Local Reference !!!")
    else:
        print("!!! Autoregressive Mode: Using Previous Frame Prediction as Local Reference !!!")
    print("=" * 60)
    
    # 0. 加载 Warp 所需的 NPZ 列表
    if not opts.npz_list:
        raise ValueError("Error: --npz_list is required for RefKV inference!")
    npz_paths = load_file_list(opts.npz_list)

    # 1. 模型初始化 (使用 RefKV 专用类)
    model_config = RefKVConfig(
        embd_pdrop=0.0, resid_pdrop=0.0, n_embd=opts.n_embd, 
        block_size=32, attn_pdrop=0.0, n_layer=opts.n_layer, n_head=opts.n_head,
        use_ref_kv=True
    )
    model = RefKVModel(model_config)
    
    print(f"Loading checkpoint from {opts.ckpt_path}...")
    checkpoint = torch.load(opts.ckpt_path)
    state_dict = checkpoint.get('model', checkpoint)
    
    # 清理权重 (RefKV 结构不包含预训练的 attn.mask)
    new_state_dict = {}
    for k, v in state_dict.items():
        if "attn.mask" in k: continue
        name = k.replace("module.", "").replace("_orig_mod.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.cuda()
    model.eval()
    
    # 2. 数据集
    test_dataset = ContinuousEdgeLineDatasetMask(
        opts.image_url, test_mask_path=opts.mask_url, is_train=False,
        image_size=opts.image_size, line_path=opts.test_line_list
    )

    edge_metric_sum = defaultdict(float)
    line_metric_sum = defaultdict(float)
    eval_count = 0    

    # =========================================================================
    # Step 1: 处理第一帧 (提取 Global Ref)
    # =========================================================================
    print("Step 1: Processing Global Reference Frame (Index 0)...")
    ref_idx = 0
    ref_items = test_dataset[ref_idx]
    
    # 获取文件名并修复后缀
    raw_name = ref_items['name']
    if isinstance(raw_name, list): raw_name = raw_name[0]
    pure_name = os.path.basename(raw_name)
    if not pure_name.lower().endswith(('.png', '.jpg')): pure_name += '.png'
    
    img_0 = _ensure_4d(ref_items['img']).cuda()
    mask_0 = _ensure_4d(ref_items['mask']).cuda()
    gt_edge_0 = _ensure_4d(ref_items['edge']).cuda()
    gt_line_0 = _ensure_4d(ref_items['line']).cuda()
    
    # 提取 Global 特征 (使用膨胀后的 GT 边缘)
    g_edge_ref = dilate_tensor(gt_edge_0, opts.ref_dilate)
    g_line_ref = dilate_tensor(gt_line_0, opts.ref_dilate)
    
    with torch.no_grad():
        global_ref_feat = model.extract_reference_features(
            global_img=None, global_edge=g_edge_ref, global_line=g_line_ref
        )
        
        # 第一帧推理 (无参考特征或仅使用刚提取的全局特征)
        edge_prob, line_prob, _ = SampleEdgeLineLogitsWithRefExtraction(
            model, 
            context=[img_0, gt_edge_0 * (1 - mask_0), gt_line_0 * (1 - mask_0)],
            mask=mask_0, 
            iterations=opts.iterations,
            extract_ref=False
        )
    
    # 保存第一帧结果
    def save_and_get_vis(prob, gt, mask, folder, name):
        p_np = _tensor_to_np(prob)
        if opts.thin: p_np = thin_lines(p_np)
        res = p_np * _tensor_to_np(mask) + _tensor_to_np(gt) * (1 - _tensor_to_np(mask))
        vis = (np.stack([res]*3, axis=-1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(opts.save_url, folder, name), vis[:, :, ::-1])
        return torch.from_numpy(res).unsqueeze(0).unsqueeze(0).cuda()

    last_edge = save_and_get_vis(edge_prob[0], gt_edge_0, mask_0, 'edge', pure_name)
    last_line = save_and_get_vis(line_prob[0], gt_line_0, mask_0, 'line', pure_name)
    last_gt_edge, last_gt_line = gt_edge_0, gt_line_0

    # =========================================================================
    # Step 2: 序列推理 (Warp + Local Ref)
    # =========================================================================
    print(f"Step 2: Propagating through {len(test_dataset)-1} frames...")
    
    for it in tqdm(range(1, len(test_dataset))):
        items = test_dataset[it]
        raw_name = items['name']
        if isinstance(raw_name, list): raw_name = raw_name[0]
        pure_name = os.path.basename(raw_name)
        if not pure_name.lower().endswith(('.png', '.jpg')): pure_name += '.png'
        
        curr_img = _ensure_4d(items['img']).cuda()
        curr_mask = _ensure_4d(items['mask']).cuda()
        curr_gt_edge = _ensure_4d(items['edge']).cuda()
        curr_gt_line = _ensure_4d(items['line']).cuda()
        
        # 1. 准备 Local 参考源
        if opts.local_GT:
            src_e, src_l = last_gt_edge, last_gt_line
        else:
            src_e, src_l = last_edge, last_line
            
        # 2. Warp & Dilate
        npz_path = npz_paths[it]
        warped_e = warp_tensor_from_npz(src_e, npz_path, size=opts.image_size)
        warped_l = warp_tensor_from_npz(src_l, npz_path, size=opts.image_size)
        l_edge_ref = dilate_tensor(warped_e, opts.ref_dilate)
        l_line_ref = dilate_tensor(warped_l, opts.ref_dilate)
        
        # 3. 提取 Local 特征并拼接
        with torch.no_grad():
            local_feat = model.extract_reference_features(
                global_img=None, local_edge=l_edge_ref, local_line=l_line_ref, 
                local_mask=torch.zeros_like(curr_mask)
            )
            # 融合特征 [Global, Local]
            full_ref_feat = torch.cat([global_ref_feat, local_feat], dim=2)

            # 4. 推理
            edge_prob, line_prob, _ = SampleEdgeLineLogitsWithRefExtraction(
                model,
                context=[curr_img, curr_gt_edge*(1-curr_mask), curr_gt_line*(1-curr_mask)],
                mask=curr_mask,
                iterations=opts.iterations,
                ref_feat=full_ref_feat,
                extract_ref=False
            )
        
        # 5. 评估
        if opts.eval_metrics:
            p_e, g_e = _tensor_to_np(edge_prob[0]), _tensor_to_np(items['edge'])
            p_l, g_l = _tensor_to_np(line_prob[0]), _tensor_to_np(items['line'])
            m_np = _tensor_to_np(items['mask'])
            e_met = compute_all_metrics(p_e, g_e, mask=m_np)
            l_met = compute_all_metrics(p_l, g_l, mask=m_np)
            for k, v in e_met.items(): edge_metric_sum[k] += v
            for k, v in l_met.items(): line_metric_sum[k] += v
            eval_count += 1
            
        # 6. 保存并更新状态
        last_edge = save_and_get_vis(edge_prob[0], curr_gt_edge, curr_mask, 'edge', pure_name)
        last_line = save_and_get_vis(line_prob[0], curr_gt_line, curr_mask, 'line', pure_name)
        last_gt_edge, last_gt_line = curr_gt_edge, curr_gt_line

    # 7. 打印报告
    if opts.eval_metrics and eval_count > 0:
        print_metrics_report("REFKV MODE", edge_metric_sum, line_metric_sum, eval_count)

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--image_url', type=str, default=None)
    parser.add_argument('--mask_url', type=str, default=None)
    parser.add_argument('--test_line_list', type=str, default='')
    
    # 新增参数
    parser.add_argument('--npz_list', type=str, default='', help="Path to txt file containing npz paths for warping")
    parser.add_argument('--ref_dilate', type=int, default=3, help="Kernel size for reference dilation")
    parser.add_argument('--local_GT', action='store_true', help="If set, use Previous Frame GT as Local Ref (Teacher Forcing)")
    parser.add_argument('--thin', action='store_true', help="If set, apply skeletonization to thin predicted lines")

    
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--save_url', type=str, default='./results')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--test_pred', action='store_true')
    parser.add_argument('--mode', type=str, default='refkv', choices=['standard', 'refkv'])
    parser.add_argument('--eval_metrics', action='store_true')
    parser.add_argument('--sparse_dir', type=str, default=None)

    opts = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.GPU_ids

    os.makedirs(opts.save_url + '/edge', exist_ok=True)
    os.makedirs(opts.save_url + '/line', exist_ok=True)
    os.makedirs(opts.save_url + '/edge_gt', exist_ok=True)

    if opts.mode == 'standard':
        standard_inference(opts)
    elif opts.mode == 'refkv':
        refkv_inference(opts)