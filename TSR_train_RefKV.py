# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys
import random
import csv
import math
import time
import collections
import queue
import threading

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 定义膨胀函数
def dilate_tensor(x, kernel_size=3):
    """
    使用 MaxPool2d 对二值/概率 Tensor 进行膨胀 (Dilation)。
    x: [B, C, H, W]
    """
    if kernel_size <= 1:
        return x
    padding = kernel_size // 2
    return F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)

# ======== Performance ========
def _dataloader_worker_init(worker_id):
    try:
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass

import torch as _torch_perf_guard
try:
    _torch_perf_guard.set_num_threads(1)
except Exception:
    pass

# ======== DDP Env ========
torch.backends.cudnn.benchmark = True
os.environ.setdefault("NCCL_P2P_DISABLE", "0")
os.environ.setdefault("NCCL_IB_DISABLE", "0")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

from datasets.dataset_Geo import (
    ContinuousEdgeLineDatasetMask,
    ContinuousEdgeLineDatasetMaskFinetune,
)
from src.models.Geo_model import EdgeLineGPT256RelBCE, EdgeLineGPTConfig
from src.utils_RefKV import set_seed

import cv2
import numpy as np
from skimage.color import rgb2gray

# --- Helpers ---
def _get_summary_writer_class():
    try:
        from torch.utils.tensorboard import SummaryWriter as _W
        return _W
    except Exception:
        class _NoOp:
            def __init__(self, *a, **k): pass
            def add_scalar(self, *a, **k): pass
            def add_text(self, *a, **k): pass
            def add_image(self, *a, **k): pass
            def close(self): pass
        return _NoOp

SummaryWriter = _get_summary_writer_class()

# ============================================================
#  New Loss Definitions: Focal + Tversky
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: 解决极度正负样本不平衡，挖掘困难样本。
    alpha: 平衡正负样本权重 (0.75 表示侧重正样本)
    gamma: 聚焦困难样本 (2.0 是标准值)
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # inputs: logits, targets: binary (0 or 1)
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss) # pt 是模型对该类别的预测概率
        
        # Focal Term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Alpha Term
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_term * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class TverskyLoss(nn.Module):
    """
    Tversky Loss: Dice Loss 的泛化版本。
    alpha: 惩罚 FP (误检) 的权重
    beta:  惩罚 FN (漏检) 的权重
    alpah = 0.3, beta = 0.7 —— 漏检更严重，容易乱画
    alpha = 0.7, beta = 0.3 —— 误检更严重，容易不敢画
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha 
        self.beta = beta   
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Flatten [B, C, H, W] -> [B, -1]
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # True Positives, False Positives, False Negatives
        TP = (probs * targets).sum(1)
        FP = ((1 - targets) * probs).sum(1)
        FN = (targets * (1 - probs)).sum(1)
        
        # Tversky Index
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - Tversky.mean()

class EdgeLineLoss(nn.Module):
    # 【修改】新增 alpha 和 beta 参数
    def __init__(self, focal_weight=1.0, tversky_weight=1.0, alpha=0.7, beta=0.3):
        super(EdgeLineLoss, self).__init__()
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        # 初始化 Focal Loss
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        
        # 【修改】使用传入的参数初始化 Tversky
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)

    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets)
        loss_tversky = self.tversky(logits, targets)
        
        return self.focal_weight * loss_focal + self.tversky_weight * loss_tversky

def check_grads_finite(model):
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all(): return False
    return True

def _ensure_4d_float(t: torch.Tensor, device):
    if t.dim() == 3: t = t.unsqueeze(0)
    return t.to(device=device, dtype=torch.float32)

def _build_clean_ref_sample(dataset, idx, image_size, device):
    img_path = dataset.image_id_list[idx]
    img = cv2.imread(img_path)
    if img is None: raise FileNotFoundError(f"Fail to read reference image: {img_path}")
    img = img[:, :, ::-1]
    img = dataset.resize(img, image_size, image_size, center_crop=False)
    img_gray = rgb2gray(img)
    edge = dataset.load_edge(img_gray)
    basename = os.path.splitext(os.path.basename(img_path))[0]
    line = dataset.load_wireframe(basename, image_size)
    img_t = dataset.to_tensor(img, norm=True)
    edge_t = torch.from_numpy(edge).float().unsqueeze(0)
    line_t = torch.from_numpy(line).float().unsqueeze(0)
    return _ensure_4d_float(img_t, device), _ensure_4d_float(edge_t, device), _ensure_4d_float(line_t, device)

def build_logger(ckpt_path):
    logger = logging.getLogger("TSR_RefKV")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    os.makedirs(ckpt_path, exist_ok=True)
    fh = logging.FileHandler(os.path.join(ckpt_path, "train.log"))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger

def append_simple_log(ckpt_path, message):
    with open(os.path.join(ckpt_path, "simplified_log.txt"), "a") as f:
        f.write(message + "\n")

# ============================================================
#  Dataset Components
# ============================================================

class MaPDatasetWrapper(Dataset):
    def __init__(self, dataset, seq_to_indices, npz_path_list=None, logger=None):
        self.dataset = dataset
        self.seq_to_indices = seq_to_indices
        self.npz_path_list = npz_path_list
        self.logger = logger
        
        self.idx_to_seq = {}
        # 预计算信息：{idx: (seq_id, is_first, prev_idx, global_idx)}
        self.idx_info = {} 
        
        count = 0
        for seq_id, idxs in seq_to_indices.items():
            # 1. 强制按文件名/帧号排序
            try:
                paths = [self.dataset.image_id_list[i] for i in idxs]
                sorted_pairs = sorted(zip(paths, idxs), key=lambda x: x[0])
                sorted_idxs = [p[1] for p in sorted_pairs]
            except:
                sorted_idxs = sorted(idxs)

            # 2. 强制序列第一帧为 Global Ref
            global_ref_idx = sorted_idxs[0]

            for i, curr_idx in enumerate(sorted_idxs):
                self.idx_to_seq[curr_idx] = seq_id
                
                # t=0 是第一帧
                is_first = (i == 0)
                # t-1 是上一帧
                prev_idx = sorted_idxs[i-1] if i > 0 else -1
                
                self.idx_info[curr_idx] = {
                    'seq_id': seq_id,
                    'is_first': is_first,
                    'prev_idx': prev_idx,
                    'global_idx': global_ref_idx
                }
            count += 1
            
        if self.logger:
            self.logger.info(f"[Dataset] Configured {count} sequences for Unidirectional Propagation (Geometry Only).")

    def __len__(self):
        return len(self.dataset)

    def _load_raw_geometry(self, idx):
        """
        读取原始几何数据 (Edge, Line, Mask) 用于 Warp。
        不返回 RGB，RGB 只用于生成 Edge。
        """
        img_path = self.dataset.image_id_list[idx]
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            # 防崩溃
            # print(f"[ERROR] Failed to read image: {img_path}")
            img = np.zeros((self.dataset.image_size, self.dataset.image_size, 3), dtype=np.uint8)

        img = self.dataset.resize(img, self.dataset.image_size, self.dataset.image_size, center_crop=False)
        
        # 1. Edge
        img_gray = rgb2gray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        edge = self.dataset.load_edge(img_gray) # Numpy [H, W]
        
        # 2. Line
        basename = os.path.splitext(os.path.basename(img_path))[0]
        line = self.dataset.load_wireframe(basename, self.dataset.image_size) # Numpy [H, W]
        
        # 3. Mask
        mask_tensor = self.dataset[idx]['mask']
        mask = mask_tensor.numpy() if isinstance(mask_tensor, torch.Tensor) else mask_tensor
        if mask.ndim == 3: mask = mask[0]
            
        return {'edge': edge, 'line': line, 'mask': mask}

    def _warp_geometry(self, data_dict, H, shape):
        """使用 Homography 矩阵 Warp 几何数据"""
        h, w = shape
        warped = {}
        # Edge/Line/Mask 都是二值或类别特征，使用 Nearest 插值保持锐利
        warped['edge'] = cv2.warpPerspective(data_dict['edge'], H, (w, h), flags=cv2.INTER_NEAREST)
        warped['line'] = cv2.warpPerspective(data_dict['line'], H, (w, h), flags=cv2.INTER_NEAREST)
        warped['mask'] = cv2.warpPerspective(data_dict['mask'], H, (w, h), flags=cv2.INTER_NEAREST)
        return warped

    def __getitem__(self, idx):
        # 1. 当前帧数据
        curr_item = self.dataset[idx]
        info = self.idx_info[idx]
        current_conf = 0.0
        
        
        # 2. Global Ref 数据
        if idx == info['global_idx']:
            g_edge = curr_item['edge']
            g_line = curr_item['line']
        else:
            g_item = self.dataset[info['global_idx']]
            g_edge = g_item['edge']
            g_line = g_item['line']

        # 3. Local Ref 数据
        l_edge_t, l_line_t, l_mask_t = None, None, None
        valid_local = False
        
        if not info['is_first'] and self.npz_path_list is not None:
            try:
                warp_data = np.load(self.npz_path_list[idx])
                # 获取 IoU
                current_conf = float(warp_data['confidence_iou']) 
                # 检查置信度
                if bool(warp_data['valid']) and float(warp_data['confidence_iou']) > 0.0:
                    prev_geo = self._load_raw_geometry(info['prev_idx'])
                    warped_raw = self._warp_geometry(prev_geo, warp_data['homography'], (self.dataset.image_size, self.dataset.image_size))
                    
                    l_edge_t = torch.from_numpy(warped_raw['edge']).float().unsqueeze(0)
                    l_line_t = torch.from_numpy(warped_raw['line']).float().unsqueeze(0)
                    l_mask_t = torch.from_numpy(warped_raw['mask']).float().unsqueeze(0)
                    valid_local = True
            except Exception:
                pass
        
        # 4. Fallback / Reference Dropout
        import random
        if not valid_local or (self.dataset.is_train and random.random() < 0.2):
            l_edge_t = torch.zeros_like(curr_item['edge'])
            l_line_t = torch.zeros_like(curr_item['line'])
            l_mask_t = torch.ones_like(curr_item['mask'])
            if l_mask_t.dim() == 2: l_mask_t = l_mask_t.unsqueeze(0)

        seq_hash = hash(str(info['seq_id']))

        # 【温和版】模拟推理时的噪声 (Simulated Inference Noise)
        # 仅在训练且 Local 有效时进行
        if self.dataset.is_train and valid_local:
            import torchvision.transforms as T
            import random
            
            # 50% 的概率触发干扰 (保留一半完美数据，保证基础学习能力)
            if random.random() < 0.5:
                # 随机选择一种干扰方式
                
                # 方式 A: 高斯模糊 (模拟预测时的边缘发虚)
                # sigma 设置小一点 (0.5 ~ 1.0)，不要太大
                if random.random() < 0.6:
                    sigma = random.uniform(0.5, 1.0)
                    l_edge_t = T.GaussianBlur(kernel_size=5, sigma=sigma)(l_edge_t)
                    l_line_t = T.GaussianBlur(kernel_size=5, sigma=sigma)(l_line_t)
                
                # 方式 B: 强度衰减 (模拟预测置信度低)
                # 乘以一个 0.6~0.9 的系数
                else:
                    scale = random.uniform(0.6, 0.9)
                    l_edge_t = l_edge_t * scale
                    l_line_t = l_line_t * scale

        # 【性能优化关键点】返回前调用 .contiguous()
        # 这能确保 Tensor 在内存中是连续的，从而触发 PyTorch 的快速 pin_memory 传输
        return {
            "c_img": curr_item['img'].contiguous(),
            "c_edge": curr_item['edge'].contiguous(),
            "c_line": curr_item['line'].contiguous(),
            "c_mask": curr_item['mask'].contiguous(),
            "g_edge": g_edge.contiguous(),
            "g_line": g_line.contiguous(),
            "l_edge": l_edge_t.contiguous(),
            "l_line": l_line_t.contiguous(),
            "l_mask": l_mask_t.contiguous(),
            "conf": torch.tensor(current_conf, dtype=torch.float32), # 新增置信度加权
            "seq_hash": seq_hash, 
            "orig_idx": idx 
        }
        
# Sampler，增加 Padding 逻辑以防止 DDP 死锁
class GroupedSequenceSampler(Sampler):
    def __init__(self, seq_to_indices, batch_size, rank=0, world_size=1, seed=42):
        self.seq_to_indices = seq_to_indices
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.seed = seed
        
        all_seq_keys = sorted(list(seq_to_indices.keys()))
        
        # 1. 计算每个 Rank 分配到的总样本数
        self.rank_lengths = collections.defaultdict(int)
        for i, k in enumerate(all_seq_keys):
            r = i % world_size
            self.rank_lengths[r] += len(seq_to_indices[k])
            
        # 2. 找到最大长度，这将是所有 Rank 的长度（不足的要补齐）
        self.max_length = 0
        if self.rank_lengths:
            self.max_length = max(self.rank_lengths.values())
        
        # 3. 获取属于当前 Rank 的序列 Key
        self.my_seq_keys = [k for i, k in enumerate(all_seq_keys) if i % world_size == rank]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        if not self.my_seq_keys:
             return iter([])

        # Shuffle 序列顺序
        indices = torch.randperm(len(self.my_seq_keys), generator=g).tolist()
        shuffled_keys = [self.my_seq_keys[i] for i in indices]
        
        final_indices = []
        for key in shuffled_keys:
            seq_idxs = self.seq_to_indices[key]
            final_indices.extend(seq_idxs)
            
        # 4. 填充 (Padding) 到 max_length
        # DDP 要求所有进程步数一致。如果本进程样本少，则重复数据直到补齐。
        current_len = len(final_indices)
        if current_len < self.max_length:
            while len(final_indices) < self.max_length:
                needed = self.max_length - len(final_indices)
                # 从头开始重复，保证数据有效
                subset = final_indices[:needed]
                if not subset: break 
                final_indices.extend(subset)
                
        return iter(final_indices)

    def __len__(self):
        # 返回 max_length，欺骗 DataLoader 认为所有 Rank 长度一致
        return self.max_length

def build_datasets_and_loader(opts, logger, train_npz_list=None):
    if not opts.MaP: raise ValueError("Only MaP mode is supported.")
    base_dataset = ContinuousEdgeLineDatasetMaskFinetune(
        pt_dataset=opts.data_path, mask_path=opts.mask_path, test_mask_path=None,
        is_train=True, mask_rates=opts.mask_rates, image_size=opts.image_size,
        line_path=opts.train_wireframes_list,
    )
    
    # 【修改】
    train_wrapper = MaPDatasetWrapper(
        base_dataset, 
        base_dataset.seq_to_indices, 
        npz_path_list=train_npz_list, # 传入
        logger=logger
    )
    
    rank = int(getattr(opts, "rank", 0))
    world_size = int(getattr(opts, "world_size", 1))
    train_sampler = GroupedSequenceSampler(base_dataset.seq_to_indices, opts.seq_batch, rank=rank, world_size=world_size, seed=opts.seed)
    
    train_loader = DataLoader(train_wrapper, batch_size=opts.seq_batch, sampler=train_sampler, num_workers=opts.num_workers,
                              pin_memory=True, persistent_workers=opts.persistent_workers, prefetch_factor=opts.prefetch_factor,
                              worker_init_fn=_dataloader_worker_init, drop_last=False)
    
    val_dataset = ContinuousEdgeLineDatasetMaskFinetune(
        pt_dataset=opts.validation_path, mask_path=opts.valid_mask_path, test_mask_path=opts.valid_mask_path,
        is_train=False, mask_rates=opts.mask_rates, image_size=opts.image_size, line_path=opts.val_wireframes_list,
    )
    return train_loader, val_dataset, train_sampler, base_dataset

def build_model(opts, device, logger):
    cfg = EdgeLineGPTConfig(n_layer=opts.n_layer, n_head=opts.n_head, n_embd=opts.n_embd,
        embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0, block_size=32, use_ref_kv=True, ref_strength=1.0)
    model = EdgeLineGPT256RelBCE(cfg).to(device)
    
    if opts.pretrain_ckpt and os.path.isfile(opts.pretrain_ckpt):
        logger.info(f"[INIT] Loading: {opts.pretrain_ckpt}")
        ckpt = torch.load(opts.pretrain_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        
        # --- 【新增】增强版清洗逻辑 ---
        new_state = {}
        for k, v in state.items():
            # 1. 移除 DDP 和 torch.compile 产生的各种前缀
            name = k.replace("module.", "").replace("_orig_mod.", "")
            
            # 2. 处理特定的层丢弃（如 attention mask）
            if "attn.mask" in name:
                continue
                
            # 3. 【特殊逻辑】处理解码器改名（如果你想尝试强制接力旧解码器）
            # 注意：这要求新旧层结构完全一致。如果不一致，建议让 Decoder 随机初始化
            # if "convt1" in name: 
            #    new_state[name.replace("convt1", "edge_decoder.0")] = v
            #    new_state[name.replace("convt1", "line_decoder.0")] = v
            
            new_state[name] = v

        # 加载并检查匹配情况
        missing_keys, unexpected_keys = model.load_state_dict(new_state, strict=False)
        
        if len(missing_keys) > 0:
            logger.warning(f"[INIT] Missing keys (random initialized): {missing_keys[:5]}... (Total: {len(missing_keys)})")
        if len(unexpected_keys) > 0:
            logger.warning(f"[INIT] Unexpected keys (skipped): {unexpected_keys[:5]}... (Total: {len(unexpected_keys)})")
            
        logger.info("[INIT] Successfully loaded pretrained weights.")
    else:
        logger.info("[INIT] Scratch training.")
        
    return model

def build_seq_ref_feats(model, dataset, device, opts):
    if not hasattr(dataset, "seq_to_indices") or not dataset.seq_to_indices: return {}
    seq_to_ref = {}
    raw_model = model.module if isinstance(model, DDP) else model
    for seq_id, idxs in dataset.seq_to_indices.items():
        if not idxs: continue
        # 这里仅作简单的 Global Ref 预提取演示
        ref_idx = idxs[0] 
        img, edge, line = _build_clean_ref_sample(dataset, ref_idx, opts.image_size, device)
        raw_model.eval()
        with torch.no_grad():
            ref_feat = raw_model.extract_reference_features(global_img=img, global_edge=edge, global_line=line)
        raw_model.train()
        seq_to_ref[seq_id] = (ref_feat.detach(), ref_idx)
    return seq_to_ref

@torch.no_grad()
def _tb_log_seq_images(model, dataset, seq_id, idxs, ref_feat, ref_global_idx, device, writer, gstep, max_frames=3):
    raw_model = model.module if isinstance(model, DDP) else model
    
    short = str(seq_id)[:8]
    picked = [i for i in idxs if i != ref_global_idx][:max_frames]
    if not picked or writer is None: return
    
    for gi in picked:
        meta = dataset[gi]
        img = meta["img"].to(device, dtype=torch.float32, non_blocking=True)
        edge_gt = meta["edge"].to(device, dtype=torch.float32, non_blocking=True)
        line_gt = meta["line"].to(device, dtype=torch.float32, non_blocking=True)
        mask = meta["mask"].to(device, dtype=torch.float32, non_blocking=True)
        
        if img.dim() == 3: img = img.unsqueeze(0)
        if edge_gt.dim() == 3: edge_gt = edge_gt.unsqueeze(0)
        if line_gt.dim() == 3: line_gt = line_gt.unsqueeze(0)
        if mask.dim() == 3: mask = mask.unsqueeze(0)
        
        rf = ref_feat
        if rf.shape[0] == 1 and img.shape[0] > 1: rf = rf.expand(img.shape[0], *rf.shape[1:])
        
        edge_logits, line_logits = raw_model.forward_with_logits(
            img_idx=img, edge_idx=edge_gt, line_idx=line_gt, masks=mask, ref_feat=rf
        )
        
        edge_pr = torch.sigmoid(edge_logits).clamp(0,1)[0,0]
        line_pr = torch.sigmoid(line_logits).clamp(0,1)[0,0]
        
        base_name = os.path.basename(dataset.image_id_list[gi])
        writer.add_image(f"img/{short}_{base_name}/edge_pred", edge_pr.unsqueeze(0), gstep)
        writer.add_image(f"img/{short}_{base_name}/line_pred", line_pr.unsqueeze(0), gstep)
        writer.add_image(f"img/{short}_{base_name}/edge_gt", edge_gt[0,0].unsqueeze(0), gstep)

# ---------------------- Train Loop ----------------------

def train_one_epoch_optimized(model, train_loader, dataset_obj, optimizer, device, scaler, logger, epoch, amp, opts, scheduler, writer=None, global_step=0):
    model.train()
    
    # --- 计算动态课程学习参数 ---
    if opts.use_dilate_curriculum:
        current_gt_dilate = 3 if epoch <= opts.dilate_switch_ep else 1
    else:
        current_gt_dilate = opts.gt_dilate

    # 仅主进程打印当前课程状态
    if dist.get_rank() == 0:
        logger.info(f"[Curriculum] Epoch {epoch}: GT Dilate={current_gt_dilate}")

    criterion = EdgeLineLoss(
        focal_weight=opts.focal_weight, 
        tversky_weight=opts.tversky_weight,
        alpha=opts.line_tv_alpha, 
        beta=opts.line_tv_beta
    )
    
    total_loss, total_samples = 0.0, 0
    raw_model = model.module if isinstance(model, DDP) else model
    idx_to_seq = train_loader.dataset.idx_to_seq
    
    for it, batch in enumerate(train_loader):
        B = batch['c_img'].size(0)
        
        # 1. 搬运数据
        c_img = batch['c_img'].to(device, non_blocking=True)
        c_edge = batch['c_edge'].to(device, non_blocking=True)
        c_line = batch['c_line'].to(device, non_blocking=True)
        c_mask = batch['c_mask'].to(device, non_blocking=True)
        g_edge = batch['g_edge'].to(device, non_blocking=True)
        g_line = batch['g_line'].to(device, non_blocking=True)
        l_edge = batch['l_edge'].to(device, non_blocking=True)
        l_line = batch['l_line'].to(device, non_blocking=True)
        l_mask = batch['l_mask'].to(device, non_blocking=True)

        # 维度修正
        for t in [c_mask, c_edge, c_line, l_mask, l_edge, l_line, g_edge, g_line]:
            if t.dim() == 3: t.unsqueeze_(1)
        
        if opts.ref_dilate > 1:
            g_edge = dilate_tensor(g_edge, opts.ref_dilate)
            g_line = dilate_tensor(g_line, opts.ref_dilate)
            l_edge = dilate_tensor(l_edge, opts.ref_dilate)
            l_line = dilate_tensor(l_line, opts.ref_dilate)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', enabled=amp, dtype=torch.bfloat16):
            # 2. 前向传播
            ref_feat = raw_model.extract_reference_features(
                global_img=None, global_edge=g_edge, global_line=g_line,
                local_img=None, local_edge=l_edge, local_line=l_line, local_mask=l_mask
            )
            
            edge_logits, line_logits = raw_model.forward_with_logits(
                img_idx=c_img, edge_idx=c_edge, line_idx=c_line, masks=c_mask, ref_feat=ref_feat
            )
            
            # 3. 计算 Loss
            target_edge = dilate_tensor(c_edge, kernel_size=current_gt_dilate)
            target_line = dilate_tensor(c_line, kernel_size=current_gt_dilate)
            
            loss_line = criterion(line_logits, target_line)
            
            # 动态加权逻辑 (基于参考帧置信度)
            avg_conf = batch['conf'].to(device).mean().item()
            loss_weight = 1.0 + avg_conf 
            
            if opts.disable_edge:
                loss = loss_line * loss_weight
            else:
                loss_edge = criterion(edge_logits, target_edge)
                loss = (loss_edge + loss_line) * loss_weight
        
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
            
        loss.backward()
        if check_grads_finite(model):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item() * B
        total_samples += B
        
        # TensorBoard 记录
        if opts.tb_images and writer is not None and dist.get_rank() == 0:
            should_log = (it % 100 == 0)
            if should_log:
                try:
                    curr_orig_idx = batch['orig_idx'][0].item()
                    seq_id = idx_to_seq.get(curr_orig_idx, "unknown")
                    seq_name = str(seq_id)[:10]
                except: seq_name = "unknown"

                with torch.no_grad():
                    sample_idx = 0
                    step = global_step + it
                    img_vis = (c_img[sample_idx] * 0.5 + 0.5).clamp(0, 1)
                    writer.add_image(f"train/{seq_name}/input", img_vis, step)
                    line_pred = torch.sigmoid(line_logits[sample_idx:sample_idx+1]).clamp(0, 1)
                    writer.add_image(f"train/{seq_name}/pred_line", line_pred[0], step)
                    
                    line_overlay = torch.cat([c_line[sample_idx], line_pred[0], torch.zeros_like(c_line[sample_idx])], dim=0)
                    writer.add_image(f"train/{seq_name}/overlay_line", line_overlay, step)

                    if not opts.disable_edge:
                        edge_pred = torch.sigmoid(edge_logits[sample_idx:sample_idx+1]).clamp(0, 1)
                        writer.add_image(f"train/{seq_name}/pred_edge", edge_pred[0], step)

        if (it + 1) % 50 == 0 and dist.get_rank() == 0:
            logger.info(f"[SeqTrain] Ep {epoch} It {it+1}/{len(train_loader)} Loss {loss.item():.4f}")

    if scheduler is not None: scheduler.step()
    
    total_loss_tensor = torch.tensor(total_loss, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    
    return total_loss_tensor.item() / max(1, total_samples_tensor.item()), global_step + len(train_loader)

# 修改 helper 增加 npz_list 加载
def load_file_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

# ============================================================
#  Validation Logic (Modified for Sequential Diffusion)
# ============================================================
@torch.no_grad()
@torch.no_grad()
def evaluate_sequence(model, val_dataset, seq_to_ref, device, logger, amp, opts, writer=None, epoch=0, val_npz_list=None):
    model.eval()
    
    if opts.use_dilate_curriculum:
        current_gt_dilate = 3 if epoch <= opts.dilate_switch_ep else 1
    else:
        current_gt_dilate = opts.gt_dilate

    criterion = EdgeLineLoss(
        focal_weight=opts.focal_weight, 
        tversky_weight=opts.tversky_weight,
        alpha=opts.line_tv_alpha, 
        beta=opts.line_tv_beta
    )
    
    raw_model = model.module if isinstance(model, DDP) else model
    
    if opts.dist:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    all_seq_ids = sorted(list(val_dataset.seq_to_indices.keys()))
    my_seq_ids = [s for i, s in enumerate(all_seq_ids) if i % world_size == rank]
    
    local_loss = 0.0
    local_frames = 0
    logged_seqs = 0
    max_log_seqs = opts.tb_max_images if opts.tb_images else 0

    def warp_tensor(tensor_map, H_mat):
        np_map = tensor_map.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        h, w = np_map.shape[:2]
        warped_np = cv2.warpPerspective(np_map, H_mat, (w, h), flags=cv2.INTER_LINEAR)
        if warped_np.ndim == 2: warped_np = warped_np[..., None]
        return torch.from_numpy(warped_np).permute(2, 0, 1).unsqueeze(0).to(device)

    for seq_id in my_seq_ids:
        raw_idxs = val_dataset.seq_to_indices[seq_id]
        try:
            paths = [val_dataset.image_id_list[i] for i in raw_idxs]
            sorted_pairs = sorted(zip(paths, raw_idxs), key=lambda x: x[0])
            sorted_idxs = [p[1] for p in sorted_pairs]
        except:
            sorted_idxs = raw_idxs
            
        if len(sorted_idxs) == 0: continue

        ref_idx_0 = sorted_idxs[0]
        _, g_edge, g_line = _build_clean_ref_sample(val_dataset, ref_idx_0, opts.image_size, device)
        
        if opts.ref_dilate > 1:
            g_edge = dilate_tensor(g_edge, opts.ref_dilate)
            g_line = dilate_tensor(g_line, opts.ref_dilate)
        
        prev_edge = g_edge
        prev_line = g_line
        
        for i, target_idx in enumerate(sorted_idxs):
            meta_tgt = val_dataset[target_idx]
            t_img = _ensure_4d_float(meta_tgt["img"], device)
            t_edge_gt = _ensure_4d_float(meta_tgt["edge"], device)
            t_line_gt = _ensure_4d_float(meta_tgt["line"], device)
            t_mask = _ensure_4d_float(meta_tgt["mask"], device)
            if t_mask.dim() == 3: t_mask = t_mask.unsqueeze(1)

            if i == 0:
                # 第一帧没有可用的前序预测，直接用参考帧监督，避免整段序列长度为1时 val_frames=0
                l_edge = g_edge
                l_line = g_line
                l_mask = torch.zeros_like(t_mask)
            else:
                warp_valid = False
                H_mat = np.eye(3)
                if val_npz_list is not None:
                    try:
                        wd = np.load(val_npz_list[target_idx])
                        if bool(wd['valid']):
                            H_mat = wd['homography']
                            warp_valid = True
                    except:
                        pass

                if warp_valid:
                    l_edge = warp_tensor(prev_edge, H_mat)
                    l_line = warp_tensor(prev_line, H_mat)
                    l_mask = torch.zeros_like(t_mask)
                else:
                    l_edge = torch.zeros_like(t_edge_gt)
                    l_line = torch.zeros_like(t_line_gt)
                    l_mask = torch.ones_like(t_mask)

            if opts.ref_dilate > 1:
                l_edge = dilate_tensor(l_edge, opts.ref_dilate)
                l_line = dilate_tensor(l_line, opts.ref_dilate)

            with torch.amp.autocast('cuda', enabled=amp, dtype=torch.bfloat16):
                ref_feat = raw_model.extract_reference_features(
                    global_img=None, global_edge=g_edge, global_line=g_line,
                    local_img=None, local_edge=l_edge, local_line=l_line, local_mask=l_mask
                )
                
                edge_logits, line_logits = raw_model.forward_with_logits(
                    img_idx=t_img, edge_idx=t_edge_gt, line_idx=t_line_gt, masks=t_mask, ref_feat=ref_feat
                )
                
                target_edge = dilate_tensor(t_edge_gt, kernel_size=current_gt_dilate)
                target_line = dilate_tensor(t_line_gt, kernel_size=current_gt_dilate)
                
                loss_line = criterion(line_logits, target_line)
                if opts.disable_edge:
                    loss = loss_line
                else:
                    loss_edge = criterion(edge_logits, target_edge)
                    loss = loss_edge + loss_line
                    
                local_loss += loss.item()
                local_frames += 1
            
            prev_edge = torch.sigmoid(edge_logits).detach()
            prev_line = torch.sigmoid(line_logits).detach()
            
            if opts.tb_images and writer is not None and rank == 0 and logged_seqs < max_log_seqs and i < 4:
                step = epoch * 1000 + i
                seq_name = str(seq_id)[:8]
                img_vis = (t_img[0] * 0.5 + 0.5).clamp(0, 1)
                writer.add_image(f"val/{seq_name}/{i}_input", img_vis, epoch)
                writer.add_image(f"val/{seq_name}/{i}_pred_line", prev_line[0], epoch)
                line_overlay = torch.cat([t_line_gt[0], prev_line[0], torch.zeros_like(t_line_gt[0])], dim=0)
                writer.add_image(f"val/{seq_name}/{i}_overlay_line", line_overlay, epoch)
        
        if opts.tb_images and writer is not None and rank == 0:
            logged_seqs += 1

    if opts.dist:
        stats = torch.tensor([local_loss, local_frames], dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        avg_loss = stats[0].item() / max(1, stats[1].item())
    else:
        avg_loss = local_loss / max(1, local_frames)
        
    if rank == 0:
        logger.info(f"[Val-Seq] Global Loss: {avg_loss:.4f} (Frames: {int(local_frames)})")
        
    return avg_loss

def main_worker(opts):
    set_seed(42)
    os.makedirs(opts.ckpt_path, exist_ok=True)
    logger = build_logger(opts.ckpt_path)
    
    if opts.dist:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank, rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    opts.rank = rank
    opts.world_size = world_size
    opts.local_rank = local_rank
    is_main = (rank == 0)
    
    if is_main:
        logger.info(f"[SYS] Device: {device} | DDP: {opts.dist} | GPUs: {world_size}")
        logger.info(f"[TB] TensorBoard logging images: {opts.tb_images}")
        with open(os.path.join(opts.ckpt_path, "simplified_log.txt"), "w") as f:
            f.write(f"Training Log for {opts.name}\n==================================\n")


    # 1. 拼接训练集清单
    def combine_lists(path_list):
        if path_list is None: 
            return None
        
        # 【关键修复】如果传入的是单个字符串，将其包装成列表
        if isinstance(path_list, str):
            path_list = [path_list]
            
        combined = []
        for p in path_list:
            # 增加文件存在性检查，防止路径空字符导致的错误
            if not os.path.isfile(p):
                logger.warning(f"[Loader] List file not found: {p}")
                continue
                
            with open(p, 'r') as f:
                content = [line.strip() for line in f.readlines() if line.strip()]
                combined.extend(content)
        return combined

    train_img_all = combine_lists(opts.data_path)
    train_pkl_all = combine_lists(opts.train_wireframes_list)
    train_npz_all = combine_lists(opts.train_npz_list)
    
    # 2. 拼接验证集清单
    val_img_all = combine_lists(opts.validation_path)
    val_pkl_all = combine_lists(opts.val_wireframes_list)
    val_npz_all = combine_lists(opts.val_npz_list)
    train_npz_list = train_npz_all
    val_npz_list = val_npz_all

    # 【重要改动】我们需要把拼接后的列表写回一个临时文件，或者修改 Dataset 接收列表对象
    # 为了不动 dataset_TSR.py 的结构，我们把拼接后的列表存为临时 txt
    temp_dir = os.path.join(opts.ckpt_path, "temp_lists")
    os.makedirs(temp_dir, exist_ok=True)
    
    def save_temp_list(data, name):
        p = os.path.join(temp_dir, name)
        with open(p, 'w') as f:
            f.write('\n'.join(data))
        return p

    opts.data_path = save_temp_list(train_img_all, "train_img.txt")
    opts.train_wireframes_list = save_temp_list(train_pkl_all, "train_pkl.txt")
    # train_npz_all 直接传给 Wrapper 即可，不需要存 txt
    
    opts.validation_path = save_temp_list(val_img_all, "val_img.txt")
    opts.val_wireframes_list = save_temp_list(val_pkl_all, "val_pkl.txt")

    # 构建 Loader (传入拼接后的 train_npz_all)
    train_loader, val_dataset, train_sampler, train_base_ds = build_datasets_and_loader(
        opts, logger, train_npz_list=train_npz_all
    )
    
    # 构建模型
    model = build_model(opts, device, logger)

    # 【性能优化 A】启用 PyTorch 2.0 编译
    # 这会合并大量零碎算子，大幅提升 GPU 利用率
    # 在 DDP 包装之前进行 compile
    if torch.__version__ >= "2.0.0":
        try:
            logger.info("[Opt] Compiling model with torch.compile...")
            # 'reduce-overhead' 模式适合小 batch 且显存充足的情况
            # 如果编译报错，可以改回 'default'
            model = torch.compile(model, mode='default') 
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    if opts.dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            logger.info("[DDP] Model wrapped with DistributedDataParallel")
    
    # 构建优化器
    # 【性能优化 B】启用 Fused Adam
    # 这能减少优化器步骤的 CPU-GPU 交互
    optim_kwargs = {'lr': opts.lr, 'betas': (0.9, 0.95), 'weight_decay': 0.0}
    if torch.cuda.is_available() and hasattr(torch.optim, 'Adam') and 'fused' in torch.optim.Adam.__init__.__code__.co_varnames:
        optim_kwargs['fused'] = True
        if is_main: logger.info("[Opt] Enabled Fused Adam.")
        
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], **optim_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.train_epoch, eta_min=1e-6)
    
    best_val = 1e9
    best_path = os.path.join(opts.ckpt_path, "best.pth")
    latest_path = os.path.join(opts.ckpt_path, "latest.pth")
    
    writer = None
    if is_main:
        writer = SummaryWriter(log_dir=opts.tb_logdir or os.path.join(opts.ckpt_path, "tb"))
        logger.info(f"[TB] TensorBoard directory: {writer.log_dir}")
    
    seq_to_ref = {}
    if opts.MaP: 
        seq_to_ref = build_seq_ref_feats(model, val_dataset, device, opts)

    global_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=opts.AMP)
    
    for epoch in range(1, opts.train_epoch + 1):
        if opts.dist: train_sampler.set_epoch(epoch)
        
        train_loss, global_step = train_one_epoch_optimized(
            model, train_loader, train_base_ds, optimizer, device, None, 
            logger, epoch, opts.AMP, opts, scheduler, 
            writer=writer, global_step=global_step
        )

        if opts.dist: dist.barrier()

        if is_main:
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": best_val}, latest_path)

        if ((epoch % opts.val_freq == 0) or (epoch == opts.train_epoch)):
            val_loss = evaluate_sequence(
                model, val_dataset, seq_to_ref, device, logger, 
                opts.AMP, opts, writer=writer, epoch=epoch,
                val_npz_list=val_npz_list
            )
            
            if is_main:
                log_msg = f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                logger.info(log_msg)
                append_simple_log(opts.ckpt_path, log_msg)
                writer.add_scalar("epoch/train_loss", train_loss, epoch)
                writer.add_scalar("epoch/val_loss", val_loss, epoch)
                
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"model": model.state_dict(), "best_val": best_val, "epoch": epoch}, best_path)
                    best_msg = f" [CKPT] New best ({best_val:.4f}) saved."
                    logger.info(best_msg)
                    append_simple_log(opts.ckpt_path, best_msg)
        else:
             if is_main: logger.info(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

    if writer is not None:
        writer.close()
    if is_main: 
        logger.info("Finish Ref-KV training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="RefKV_train_seq")
    parser.add_argument("--GPU_ids", type=str, default="0")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt")
    parser.add_argument("--data_path", type=str, required=True, nargs='+', help='Image lists')
    parser.add_argument("--validation_path", type=str, required=True, nargs='+', help='Val image lists')
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--valid_mask_path", type=str, required=True)
    parser.add_argument("--train_wireframes_list", type=str, required=True, nargs='+', help='PKL lists')
    parser.add_argument("--val_wireframes_list", type=str, required=True, nargs='+', help='Val PKL lists')
    parser.add_argument("--mask_rates", type=float, nargs="+", default=[0.4, 0.8, 1.0])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=16)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_epoch", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seq_batch", type=int, default=8)
    parser.add_argument("--AMP", action="store_true")
    parser.add_argument("--MaP", action="store_true")
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--ref_topk", type=int, default=10)
    parser.add_argument("--ref_w_vis", type=float, default=0.7)
    parser.add_argument("--ref_w_sharp", type=float, default=0.3)
    parser.add_argument("--tb_logdir", type=str, default=None)
    parser.add_argument("--tb_images", action="store_true")
    parser.add_argument("--tb_max_images", type=int, default=3)
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_freq", type=int, default=2, help="Validation frequency")
    parser.add_argument("--focal_weight", type=float, default=1.0)
    parser.add_argument("--tversky_weight", type=float, default=1.0)
    parser.add_argument("--gt_dilate", type=int, default=3, help="Kernel size for dilating GT lines")
    parser.add_argument("--ref_dilate", type=int, default=1, help="Kernel size for dilating Reference lines (Global/Local)")
    parser.add_argument("--disable_edge", action="store_true", help="If set, disable loss calculation for Edge")
    parser.add_argument("--train_npz_list", type=str, default=None, nargs='+', help='.npz lists')
    parser.add_argument("--val_npz_list", type=str, default=None, nargs='+', help='Val .npz lists')

    parser.add_argument("--line_tv_alpha", type=float, default=0.3, help="Line FP penalty (越小越敢画)")
    parser.add_argument("--line_tv_beta", type=float, default=0.7, help="Line FN penalty (越大越怕断)")
    parser.add_argument("--edge_tv_alpha", type=float, default=0.7)
    parser.add_argument("--edge_tv_beta", type=float, default=0.3)

    parser.add_argument("--use_dilate_curriculum", action="store_true", help="是否开启GT加粗渐进式学习")
    parser.add_argument("--dilate_switch_ep", type=int, default=3, help="在哪个Epoch将GT Dilation从3降到1")

    
    opts = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.GPU_ids
    opts.ckpt_path = os.path.join(opts.ckpt_path, opts.name)
    os.makedirs(opts.ckpt_path, exist_ok=True)
    main_worker(opts)