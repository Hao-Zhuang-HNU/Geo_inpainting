import argparse
import os
import cv2
import torch
import numpy as np
import pickle
from tqdm import tqdm
import torchvision.transforms as T
from torch.multiprocessing import Pool, set_start_method
from scipy.spatial.distance import cdist

try:
    from kornia.feature import DISK, LightGlue
except ImportError:
    print("Error: Kornia not installed. Please install via: pip install kornia")
    exit(1)

import warnings
warnings.filterwarnings("ignore")

# ================= Global for Workers =================
model_disk = None
model_lightglue = None
device = None
transform = None

def init_worker(gpu_id):
    """
    初始化工作进程：在指定的 GPU 上加载模型，避免重复加载。
    """
    global model_disk, model_lightglue, device, transform
    
    # 简单的轮询或指定 GPU 策略
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0')
    
    # 加载 DISK 和 LightGlue (设置为评估模式)
    try:
        model_disk = DISK.from_pretrained('depth').to(device).eval()
        model_lightglue = LightGlue('disk').to(device).eval()
    except Exception as e:
        print(f"Worker init failed: {e}")
        return

    transform = T.ToTensor()

def get_hawp_lines_proven(pkl_path, img_w, img_h):
    """
    鲁棒地从 Pickle 文件读取 HAWP 线框数据并归一化/反归一化。
    """
    if not os.path.exists(pkl_path): return np.zeros((0, 4))
    try:
        with open(pkl_path, 'rb') as f: data = pickle.load(f)
        # 兼容不同的字典结构
        lines = data.get('lines', data.get('lines_pred', [])) if isinstance(data, dict) else data
        lines = np.array(lines)
        
        if lines.size == 0: return np.zeros((0, 4))
        if lines.ndim != 2: lines = lines.reshape(-1, 4)
        
        lines_cv = lines.copy()
        # HAWP 格式通常是 [y1, x1, y2, x2]，OpenCV 需要 [x1, y1, x2, y2]
        lines_cv[:, [0, 1, 2, 3]] = lines_cv[:, [1, 0, 3, 2]] 
        
        # 如果是归一化坐标 (0-1)，则扩展到图像尺寸
        if lines_cv.max() <= 1.05:
            lines_cv[:, [0, 2]] *= img_w
            lines_cv[:, [1, 3]] *= img_h
        return lines_cv
    except:
        return np.zeros((0, 4))

def warp_lines_local_idw(lines, mkpts0, mkpts1, k=10, dist_thresh=None):
    """
    使用逆距离加权 (IDW) 对线段进行局部 Warp。
    相比 Homography，能更好地处理视差。
    """
    if len(lines) == 0:
        return lines
    if len(mkpts0) == 0:
        return lines 

    # 1. 计算稀疏光流向量
    flow_vectors = mkpts1 - mkpts0 

    # 2. 将线段拆分为端点 (2N, 2)
    line_points = lines.reshape(-1, 2)
    
    # 3. 计算距离矩阵 (CPU上有可能会比较慢，但对于几千个点通常是毫秒级)
    # line_points: (2N, 2), mkpts0: (M, 2)
    dists = cdist(line_points, mkpts0, metric='euclidean')

    # 4. 找到最近的 K 个匹配点
    curr_k = min(k, len(mkpts0))
    
    # 获取最近 K 个点的索引和距离
    idx = np.argpartition(dists, curr_k, axis=1)[:, :curr_k]
    nearest_dists = np.take_along_axis(dists, idx, axis=1)
    
    # 5. 计算 IDW 权重
    epsilon = 1e-6
    weights = 1.0 / (nearest_dists + epsilon)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights_norm = weights / weights_sum

    # 6. 加权平均计算位移
    nearest_flow = flow_vectors[idx] # (2N, k, 2)
    delta = np.sum(weights_norm[:, :, np.newaxis] * nearest_flow, axis=1)

    # 7. 应用位移
    warped_points = line_points + delta

    # (可选) 距离过滤：如果端点离所有匹配点都很远，可能不可靠
    if dist_thresh is not None:
        min_dists = nearest_dists[:, 0]
        mask = min_dists > dist_thresh
        warped_points[mask] = line_points[mask]

    return warped_points.reshape(-1, 4)

def process_pair(args):
    path_curr, pkl_curr, path_prev, pkl_prev, save_path, is_first = args
    
    # 如果已存在，跳过 (断点续传)
    if os.path.exists(save_path): return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 第一帧没有前一帧，无法计算光流/Warp
    if is_first:
        np.savez_compressed(
            save_path, 
            homography=np.eye(3, dtype=np.float32), 
            mkpts0=np.zeros((0, 2), dtype=np.float32),
            mkpts1=np.zeros((0, 2), dtype=np.float32),
            confidence_match=0.0, 
            confidence_iou=0.0, 
            valid=False
        )
        return

    # 1. 读取图像
    img_curr_bgr = cv2.imread(path_curr)
    img_prev_bgr = cv2.imread(path_prev)
    
    if img_curr_bgr is None or img_prev_bgr is None:
        # 读取失败，写入空数据防止报错
        np.savez_compressed(save_path, homography=np.eye(3), valid=False)
        return

    h, w = img_curr_bgr.shape[:2]
    
    # BGR -> RGB -> Tensor
    img_curr_rgb = cv2.cvtColor(img_curr_bgr, cv2.COLOR_BGR2RGB)
    img_prev_rgb = cv2.cvtColor(img_prev_bgr, cv2.COLOR_BGR2RGB)
    
    # 推理时不需要梯度
    t_curr = transform(img_curr_rgb).unsqueeze(0).to(device)
    t_prev = transform(img_prev_rgb).unsqueeze(0).to(device)

    H_mat = np.eye(3, dtype=np.float32)
    mkpts0 = np.zeros((0, 2), dtype=np.float32)
    mkpts1 = np.zeros((0, 2), dtype=np.float32)
    match_score = 0.0
    num_matches = 0
    
    try:
        with torch.no_grad():
            # 2. 特征提取 (DISK)
            feats0 = model_disk(t_prev)
            feats1 = model_disk(t_curr)
            
            kpts0, desc0 = feats0[0].keypoints, feats0[0].descriptors
            kpts1, desc1 = feats1[0].keypoints, feats1[0].descriptors
            
            # 维度调整
            if kpts0.dim() == 2: kpts0 = kpts0.unsqueeze(0)
            if kpts1.dim() == 2: kpts1 = kpts1.unsqueeze(0)
            if desc0.dim() == 2: desc0 = desc0.unsqueeze(0)
            if desc1.dim() == 2: desc1 = desc1.unsqueeze(0)

            # 3. 特征匹配 (LightGlue)
            input_dict = {
                'image0': {'keypoints': kpts0, 'descriptors': desc0, 'image': t_prev, 'image_size': torch.tensor([[w, h]], device=device)},
                'image1': {'keypoints': kpts1, 'descriptors': desc1, 'image': t_curr, 'image_size': torch.tensor([[w, h]], device=device)}
            }
            
            matches = model_lightglue(input_dict)
            
            # 4. 解析匹配点
            if 'matches0' in matches:
                m0 = matches['matches0'][0]
                valid_mask = m0 > -1
                mkpts0 = kpts0[0][valid_mask].cpu().numpy()
                mkpts1 = kpts1[0][m0[valid_mask]].cpu().numpy()
            elif 'matches' in matches:
                ms = matches['matches'][0]
                mkpts0 = kpts0[0][ms[:,0]].cpu().numpy()
                mkpts1 = kpts1[0][ms[:,1]].cpu().numpy()

            num_matches = len(mkpts0)
            if num_matches > 0:
                # 简单的归一化分数，仅供参考
                match_score = min(1.0, num_matches / 200.0)

            # 5. 计算全局单应性 (作为 Fallback 或粗略对齐)
            if num_matches > 4:
                H_computed, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
                if H_computed is not None:
                    H_mat = H_computed.astype(np.float32)

    except Exception as e:
        print(f"Inference error {path_curr}: {e}")

    # 6. 计算 IoU (使用 IDW 局部 Warp)
    iou_score = 0.0
    if num_matches > 10: # IDW 需要一定数量的匹配点才稳定
        lines_prev = get_hawp_lines_proven(pkl_prev, w, h)
        lines_curr = get_hawp_lines_proven(pkl_curr, w, h)
        
        if len(lines_prev) > 0 and len(lines_curr) > 0:
            # === 核心修改：使用 IDW 替代 PerspectiveTransform ===
            # 将上一帧的线根据光流场 warp 到当前帧
            lines_warped = warp_lines_local_idw(lines_prev, mkpts0, mkpts1, k=10)
            
            # 使用栅格化 (Rasterization) 方法快速计算 IoU
            # 缩小画布尺寸以加快计算速度 (精度足够用于过滤)
            scale = 0.25 # 缩放到原图的 1/4
            sh, sw = int(h*scale), int(w*scale)
            
            c_warp = np.zeros((sh, sw), dtype=np.uint8)
            c_curr = np.zeros((sh, sw), dtype=np.uint8)
            
            # 绘制线条 (thickness=1)
            # 注意坐标需要缩放
            for l in lines_warped * scale:
                cv2.line(c_warp, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), 1, 1)
            
            for l in lines_curr * scale:
                cv2.line(c_curr, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), 1, 1)
                
            # 计算 IoU
            inter = np.logical_and(c_warp, c_curr).sum()
            union = np.logical_or(c_warp, c_curr).sum()
            iou_score = inter / union if union > 0 else 0.0

    # 7. 保存结果
    is_valid = (match_score > 0.05) # 稍微放宽阈值，因为 LightGlue 很准
    
    np.savez_compressed(
        save_path,
        homography=H_mat,       # 全局 H (兼容旧代码)
        mkpts0=mkpts0,          # 源匹配点 (新特性，允许后续复现 IDW)
        mkpts1=mkpts1,          # 目标匹配点
        confidence_match=float(match_score),
        confidence_iou=float(iou_score), # 基于 IDW 的 IoU
        valid=is_valid
    )

def main():
    parser = argparse.ArgumentParser(description="Preprocess HAWP dataset with LightGlue + IDW Warp")
    parser.add_argument('--img_list', required=True, help="List of image paths")
    parser.add_argument('--pkl_list', required=True, help="List of HAWP wireframe pkl paths")
    parser.add_argument('-o', '--output_dir', required=True, help="Output directory for .npz files")
    parser.add_argument('--num_workers', type=int, default=16, help="Number of parallel workers")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()

    with open(args.img_list) as f: imgs = [x.strip() for x in f.readlines()]
    with open(args.pkl_list) as f: pkls = [x.strip() for x in f.readlines()]
    
    if len(imgs) != len(pkls):
        print("Error: Length of img_list and pkl_list must match.")
        exit(1)

    tasks = []
    # 尝试计算公共前缀以生成简洁的文件名
    try:
        common = os.path.commonpath(imgs)
    except:
        common = os.path.dirname(imgs[0])

    print(f"Generating tasks for {len(imgs)} images...")
    
    for i in range(len(imgs)):
        path_curr = imgs[i]
        pkl_curr = pkls[i]
        
        # 逻辑：每一帧和它前一帧进行 Warp
        # 如果是序列的第一帧，或者文件夹发生变化（说明换了一个视频片段），则标记为 is_first
        if i > 0:
            path_prev = imgs[i-1]
            pkl_prev = pkls[i-1]
            # 简单判断：如果父目录不同，视为新场景
            if os.path.dirname(path_curr) != os.path.dirname(path_prev):
                is_first = True
                path_prev = "" # 占位
                pkl_prev = ""
            else:
                is_first = False
        else:
            is_first = True
            path_prev = ""
            pkl_prev = ""
        
        # 生成输出路径
        rel = os.path.relpath(path_curr, common)
        # 替换扩展名为 .npz
        save_path = os.path.join(args.output_dir, os.path.splitext(rel)[0] + ".npz")
        
        tasks.append((path_curr, pkl_curr, path_prev, pkl_prev, save_path, is_first))

    print(f"Starting multiprocessing pool with {args.num_workers} workers on GPU {args.gpu_id}...")
    
    # 设置启动方法，spawn 对 CUDA 更安全
    try: 
        set_start_method('spawn')
    except RuntimeError: 
        pass
    
    with Pool(args.num_workers, initializer=init_worker, initargs=(args.gpu_id,)) as p:
        # 使用 imap_unordered 提高效率，tqdm 显示进度
        list(tqdm(p.imap_unordered(process_pair, tasks, chunksize=5), total=len(tasks)))
    
    print("Preprocessing finished.")

if __name__ == '__main__':
    main()