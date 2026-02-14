import os
import numpy as np
import cv2
import torch
import struct
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import average_precision_score

# =========================================================
# Part 1: 基础几何指标 (F1, Chamfer, AP)
# =========================================================

def _to_binary(data, threshold=0.5):
    """将概率图转换为二值图 (numpy, bool)"""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return data > threshold

def compute_pixel_ap(pred, gt, mask=None):
    """
    计算像素级 Average Precision (AP)
    pred: [H, W] range 0-1 (预测概率)
    gt: [H, W] range 0 or 1 (真实标签)
    mask: [H, W] range 0 or 1 (可选)
    """
    # 1. 展平
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # 2. 处理 Mask
    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
        pred_flat = pred_flat[mask_flat]
        gt_flat = gt_flat[mask_flat]
        
    # 3. 边界检查
    if len(gt_flat) == 0:
        return 0.0
        
    # === 关键修复: 强制 GT 二值化 ===
    # sklearn 要求 y_true 为 {0, 1} 或 {-1, 1}
    # 即使 GT 只有 0.0 和 1.0，如果是 float 类型有时候也会报错，最好转 int
    gt_flat = (gt_flat > 0.5).astype(int) 
    
    if gt_flat.sum() == 0:
        return 0.0 # 如果没有正样本，AP 无定义或为 0
        
    return average_precision_score(gt_flat, pred_flat)

def compute_structural_f1(pred, gt, mask=None, threshold=0.5, tolerance=2.0):
    """
    计算带容差的 F1-Score
    """
    pred_bin = _to_binary(pred, threshold)
    gt_bin = _to_binary(gt, 0.5)
    
    if mask is not None:
        mask_bin = _to_binary(mask, 0.5)
        pred_bin = pred_bin & mask_bin
        gt_bin = gt_bin & mask_bin

    if not np.any(pred_bin) and not np.any(gt_bin):
        return 1.0
    if not np.any(pred_bin) or not np.any(gt_bin):
        return 0.0

    # Distance Transform
    # dt_gt: distance from every pixel to nearest GT pixel
    dt_gt = distance_transform_edt(~gt_bin)
    # dt_pred: distance from every pixel to nearest Pred pixel
    dt_pred = distance_transform_edt(~pred_bin)
    
    # Precision: pred points within tolerance of gt
    tp_pred = np.sum((dt_gt[pred_bin] <= tolerance))
    precision = tp_pred / (np.sum(pred_bin) + 1e-6)
    
    # Recall: gt points within tolerance of pred
    tp_gt = np.sum((dt_pred[gt_bin] <= tolerance))
    recall = tp_gt / (np.sum(gt_bin) + 1e-6)
    
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1

def compute_chamfer_distance(pred, gt, mask=None, threshold=0.5):
    """
    计算 Mask 区域内的 Chamfer Distance
    """
    pred_bin = _to_binary(pred, threshold)
    gt_bin = _to_binary(gt, 0.5)
    
    if mask is not None:
        mask_bin = _to_binary(mask, 0.5)
        pred_bin = pred_bin & mask_bin
        gt_bin = gt_bin & mask_bin

    # 如果一方为空，返回一个较大的惩罚值或0（视情况而定）
    if not np.any(gt_bin) and not np.any(pred_bin):
        return 0.0
    if not np.any(gt_bin) or not np.any(pred_bin):
        return 100.0 # Penalty

    dt_gt = distance_transform_edt(~gt_bin)
    dt_pred = distance_transform_edt(~pred_bin)
    
    # Average distance from Pred to nearest GT
    d_p2g = np.mean(dt_gt[pred_bin])
    # Average distance from GT to nearest Pred
    d_g2p = np.mean(dt_pred[gt_bin])
    
    return (d_p2g + d_g2p) / 2.0

# =========================================================
# Part 2: COLMAP 数据读取与重投影一致性
# =========================================================

def read_colmap_bin_array(path):
    """
    读取 COLMAP .bin 格式的稠密矩阵 (depth maps)
    格式: width&height&channels&DATA
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Depth file not found: {path}")

    with open(path, "rb") as fid:
        # 1. 解析文本头：一直读取直到遇到第三个 '&'
        header_bytes = b""
        amp_count = 0
        while True:
            char = fid.read(1)
            if char == b"":
                raise ValueError(f"Unexpected EOF in header: {path}")
            
            header_bytes += char
            if char == b"&":
                amp_count += 1
                if amp_count >= 3:
                    break

        # 2. 解析头信息
        # header_bytes 例如: b'256&256&1&'
        parts = header_bytes.split(b"&")
        width = int(parts[0])
        height = int(parts[1])
        channels = int(parts[2])

        # 3. 读取剩余的 float32 数据
        expected_floats = width * height * channels
        data = np.fromfile(fid, dtype=np.float32)

    # 4. 校验与 Reshape
    if data.size != expected_floats:
        raise ValueError(
            f"Size mismatch in {path}: Header says {width}x{height}x{channels}, "
            f"expected {expected_floats} floats, but found {data.size}."
        )

    # 统一 reshape 为 [H, W, C] 以兼容后续处理
    data = data.reshape((height, width, channels))
    
    return data

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

class ColmapEvaluator:
    def __init__(self, sparse_dir):
        """
        sparse_dir: COLMAP output folder containing 'images.txt', 'cameras.txt'
        """
        self.cameras = {}
        self.images = {}
        self.name_to_img_id = {}
        
        if sparse_dir and os.path.exists(sparse_dir):
            print(f"[Eval] Loading COLMAP data from {sparse_dir}...")
            self._load_cameras(os.path.join(sparse_dir, "cameras.txt"))
            self._load_images(os.path.join(sparse_dir, "images.txt"))
        else:
            print(f"[Eval] Warning: Sparse dir {sparse_dir} not found. Reprojection metric will be skipped.")

    def _load_cameras(self, path):
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"): continue
                els = line.strip().split()
                cam_id = int(els[0])
                model = els[1]
                w, h = int(els[2]), int(els[3])
                params = np.array([float(x) for x in els[4:]])
                self.cameras[cam_id] = {"model": model, "width": w, "height": h, "params": params}

    def _load_images(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith("#"):
                    i += 1
                    continue
                # Image line
                els = line.split()
                img_id = int(els[0])
                qvec = np.array([float(x) for x in els[1:5]])
                tvec = np.array([float(x) for x in els[5:8]])
                cam_id = int(els[8])
                name = els[9]
                
                R = qvec2rotmat(qvec)
                self.images[img_id] = {"id": img_id, "R": R, "t": tvec, "camera_id": cam_id, "name": name}
                self.name_to_img_id[name] = img_id
                
                # Skip points2D line
                i += 2

    def get_intrinsic_matrix(self, cam_id):
        cam = self.cameras[cam_id]
        params = cam["params"]
        K = np.eye(3)
        if cam["model"] == "PINHOLE":
            fx, fy, cx, cy = params
            K[0,0], K[1,1], K[0,2], K[1,2] = fx, fy, cx, cy
        elif cam["model"] == "SIMPLE_RADIAL":
            f, cx, cy, _ = params
            K[0,0], K[1,1], K[0,2], K[1,2] = f, f, cx, cy
        elif cam["model"] == "SIMPLE_PINHOLE":
            f, cx, cy = params
            K[0,0], K[1,1], K[0,2], K[1,2] = f, f, cx, cy
        else:
            # Simplified handling for other models, assuming first param is f
            f = params[0]
            cx = cam["width"] / 2
            cy = cam["height"] / 2
            K[0,0], K[1,1], K[0,2], K[1,2] = f, f, cx, cy
        return K

    def compute_reprojection_consistency(self, pred_edge, ref_edge_gt, cur_name, ref_name, depth_path, mask=None):
        """
        计算重投影一致性 (IoU / F1 of Warped Reference vs Prediction)
        """
        if cur_name not in self.name_to_img_id or ref_name not in self.name_to_img_id:
            return -1.0
        
        if not os.path.exists(depth_path):
            # print(f"Depth missing: {depth_path}")
            return -1.0

        # 1. Get Poses and Intrinsics
        cur_img = self.images[self.name_to_img_id[cur_name]]
        ref_img = self.images[self.name_to_img_id[ref_name]]
        
        K_cur = self.get_intrinsic_matrix(cur_img["camera_id"])
        K_ref = self.get_intrinsic_matrix(ref_img["camera_id"])
        
        R_cur, t_cur = cur_img["R"], cur_img["t"]
        R_ref, t_ref = ref_img["R"], ref_img["t"]

        # 2. Load Reference Depth
        try:
            depth_ref = read_colmap_bin_array(depth_path) # [H, W, 1]
            if depth_ref.ndim == 3: depth_ref = depth_ref[:,:,0]
        except Exception as e:
            print(f"[Colmap Error] Failed to read depth: {depth_path}")
            print(f"[Colmap Error] Detail: {e}")            
            return -1.0

        H, W = pred_edge.shape
        
        # Resize depth if needed (COLMAP depth usually matches image size, but check)
        if depth_ref.shape != (H, W):
            depth_ref = cv2.resize(depth_ref, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # 3. Warp Reference Edge GT to Current Frame
        # Back-project Reference: pixel -> camera -> world
        y_ref, x_ref = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        x_ref = x_ref.flatten()
        y_ref = y_ref.flatten()
        z_ref = depth_ref.flatten()
        
        # Filter valid depth
        valid = z_ref > 1e-3
        x_ref, y_ref, z_ref = x_ref[valid], y_ref[valid], z_ref[valid]
        
        # Pixel to Camera (Ref)
        # P_cam = K_inv * [u, v, 1] * z
        fx, fy = K_ref[0,0], K_ref[1,1]
        cx, cy = K_ref[0,2], K_ref[1,2]
        X_cam = (x_ref - cx) * z_ref / fx
        Y_cam = (y_ref - cy) * z_ref / fy
        Z_cam = z_ref
        
        P_cam_ref = np.vstack((X_cam, Y_cam, Z_cam)) # [3, N]
        
        # Camera to World
        # P_world = R_ref^T * (P_cam - t_ref) ... wait, COLMAP stores R_world2cam
        # P_cam = R * P_world + t  => P_world = R^T * (P_cam - t)
        P_world = R_ref.T @ (P_cam_ref - t_ref[:, None])
        
        # World to Camera (Cur)
        P_cam_cur = R_cur @ P_world + t_cur[:, None]
        
        # Camera to Pixel (Cur)
        X_cur, Y_cur, Z_cur = P_cam_cur[0], P_cam_cur[1], P_cam_cur[2]
        u_cur = (X_cur * K_cur[0,0] / Z_cur) + K_cur[0,2]
        v_cur = (Y_cur * K_cur[1,1] / Z_cur) + K_cur[1,2]
        
        # 4. Create Warped Edge Map
        u_cur = np.round(u_cur).astype(int)
        v_cur = np.round(v_cur).astype(int)
        
        # Filter bounds
        valid_proj = (u_cur >= 0) & (u_cur < W) & (v_cur >= 0) & (v_cur < H) & (Z_cur > 0)
        u_cur, v_cur = u_cur[valid_proj], v_cur[valid_proj]
        
        # Map indices back to original reference edge values
        orig_indices = np.where(valid)[0][valid_proj]
        ref_edge_flat = ref_edge_gt.flatten()
        val_edge = ref_edge_flat[orig_indices]
        
        # Only project edges (value > 0.5)
        edge_indices = val_edge > 0.5
        u_cur_edge = u_cur[edge_indices]
        v_cur_edge = v_cur[edge_indices]
        
        warped_gt = np.zeros((H, W), dtype=np.float32)
        warped_gt[v_cur_edge, u_cur_edge] = 1.0
        
        # Dilate warped GT slightly to handle small registration errors
        kernel = np.ones((3,3), np.uint8)
        warped_gt = cv2.dilate(warped_gt, kernel, iterations=1)
        
        # 5. Compute IoU / F1 between Warped GT and Prediction in Mask
        return compute_structural_f1(pred_edge, warped_gt, mask=mask, tolerance=3.0)

# =========================================================
# Wrapper Function for Inference Script
# =========================================================

# 全局评估器实例
_colmap_evaluator = None

def init_colmap_evaluator(sparse_dir):
    global _colmap_evaluator
    _colmap_evaluator = ColmapEvaluator(sparse_dir)

def compute_all_metrics(pred, gt, mask=None, threshold=0.5, 
                        use_reproj=False, 
                        cur_name=None, ref_name=None, depth_dir=None):
    """
    统一接口计算所有指标
    """
    metrics = {}
    
    # 1. Basic Metrics
    metrics['f1'] = compute_structural_f1(pred, gt, mask, threshold)
    metrics['ap'] = compute_pixel_ap(pred, gt, mask)
    metrics['chamfer'] = compute_chamfer_distance(pred, gt, mask, threshold)
    
    # 2. Reprojection Consistency (Optional)
    if use_reproj and _colmap_evaluator is not None and cur_name and ref_name and depth_dir:
        # 构造深度图路径: frame_00001.png.geometric.bin
        # 假设 colmap depth 命名规则是 图像名 + .geometric.bin
        depth_name = ref_name + ".geometric.bin"
        depth_path = os.path.join(depth_dir, depth_name)
        
        reproj_score = _colmap_evaluator.compute_reprojection_consistency(
            pred, gt, cur_name, ref_name, depth_path, mask
        )
        if reproj_score >= 0:
            metrics['reproj_consist'] = reproj_score
            
    return metrics

def get_colmap_evaluator():
    """返回全局的 evaluator 实例"""
    return _colmap_evaluator