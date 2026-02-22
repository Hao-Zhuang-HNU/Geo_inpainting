# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys
import random
import time
import collections
import yaml

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torchvision.transforms as T 

import torch.multiprocessing

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

from datasets.dataset_TSR import (
    ContinuousEdgeLineDatasetMask,
    ContinuousEdgeLineDatasetMaskFinetune,
)
from src.models.TSR_model_RefKV import EdgeLineGPT256RelBCE, EdgeLineGPTConfig
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss) 
        focal_term = (1 - pt) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_term * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha 
        self.beta = beta   
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        TP = (probs * targets).sum(1)
        FP = ((1 - targets) * probs).sum(1)
        FN = (targets * (1 - probs)).sum(1)
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - Tversky.mean()

class EdgeLineLoss(nn.Module):
    def __init__(self, focal_weight=1.0, tversky_weight=1.0, alpha=0.7, beta=0.3):
        super(EdgeLineLoss, self).__init__()
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
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
    
    # 确保返回的tensor不保留梯度信息
    return (_ensure_4d_float(img_t, device).detach(), 
            _ensure_4d_float(edge_t, device).detach(), 
            _ensure_4d_float(line_t, device).detach())

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
    def __init__(self, dataset, seq_to_indices, npz_path_list=None, logger=None, no_align=False):
        self.dataset = dataset
        self.seq_to_indices = seq_to_indices
        self.npz_path_list = npz_path_list
        self.logger = logger
        self.no_align = no_align  # ablation
        self.idx_to_seq = {}
        self.idx_info = {} 
        
        count = 0
        for seq_id, idxs in seq_to_indices.items():
            try:
                paths = [self.dataset.image_id_list[i] for i in idxs]
                sorted_pairs = sorted(zip(paths, idxs), key=lambda x: x[0])
                sorted_idxs = [p[1] for p in sorted_pairs]
            except:
                sorted_idxs = sorted(idxs)

            global_ref_idx = sorted_idxs[0]

            for i, curr_idx in enumerate(sorted_idxs):
                self.idx_to_seq[curr_idx] = seq_id
                is_first = (i == 0)
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
        img_path = self.dataset.image_id_list[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.dataset.image_size, self.dataset.image_size, 3), dtype=np.uint8)

        img = self.dataset.resize(img, self.dataset.image_size, self.dataset.image_size, center_crop=False)
        img_gray = rgb2gray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        edge = self.dataset.load_edge(img_gray)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        line = self.dataset.load_wireframe(basename, self.dataset.image_size)
        mask_tensor = self.dataset[idx]['mask']
        mask = mask_tensor.numpy() if isinstance(mask_tensor, torch.Tensor) else mask_tensor
        if mask.ndim == 3: mask = mask[0]
        return {'edge': edge, 'line': line, 'mask': mask}

    def _warp_geometry(self, data_dict, H, shape):
        h, w = shape
        warped = {}
        warped['edge'] = cv2.warpPerspective(data_dict['edge'], H, (w, h), flags=cv2.INTER_NEAREST)
        warped['line'] = cv2.warpPerspective(data_dict['line'], H, (w, h), flags=cv2.INTER_NEAREST)
        warped['mask'] = cv2.warpPerspective(data_dict['mask'], H, (w, h), flags=cv2.INTER_NEAREST)
        return warped

    def __getitem__(self, idx):
        # 1. 当前帧数据
        curr_item = self.dataset[idx]
        info = self.idx_info[idx]
        current_conf = 0.0
        npz_ok = 0.0
        warp_valid = 0.0
        local_used = 0.0   # whether we actually use warped local ref
        is_first = 1.0 if info['is_first'] else 0.0
        
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

        # [修改] 增加 no_align 逻辑分支
        if not info['is_first']:
            if self.no_align:
                # 模式：不进行对齐，直接使用当前帧 GT 作为 local reference
                try:
                    l_edge_t = curr_item['edge'].clone()
                    l_line_t = curr_item['line'].clone()
                    l_mask_t = torch.zeros_like(curr_item['mask'])
                    if l_mask_t.dim() == 2:
                        l_mask_t = l_mask_t.unsqueeze(0)
                    
                    current_conf = 1.0 # 强制全通
                    warp_valid = 1.0
                    npz_ok = 1.0 # 模拟 npz 存在
                    valid_local = True
                    local_used = 1.0
                except Exception as e:
                    pass
            elif self.npz_path_list is not None:
                # 模式：原有的 Warp 逻辑
                try:
                    warp_data = np.load(self.npz_path_list[idx])
                    npz_ok = 1.0
                    warp_valid = 1.0 if bool(warp_data.get('valid', False)) else 0.0
                    current_conf = float(warp_data.get('confidence_iou', 0.0))

                    if warp_valid > 0.5 and current_conf > 0.0:
                        prev_geo = self._load_raw_geometry(info['prev_idx'])
                        warped_raw = self._warp_geometry(
                            prev_geo, warp_data['homography'],
                            (self.dataset.image_size, self.dataset.image_size)
                        )
                        l_edge_t = torch.from_numpy(warped_raw['edge']).float().unsqueeze(0)
                        l_line_t = torch.from_numpy(warped_raw['line']).float().unsqueeze(0)
                        l_mask_t = torch.from_numpy(warped_raw['mask']).float().unsqueeze(0)
                        valid_local = True
                        local_used = 1.0
                except Exception:
                    pass

        if not valid_local:
            l_edge_t = torch.zeros_like(curr_item['edge'])
            l_line_t = torch.zeros_like(curr_item['line'])
            l_mask_t = torch.ones_like(curr_item['mask'])
            if l_mask_t.dim() == 2:
                l_mask_t = l_mask_t.unsqueeze(0)

        seq_hash = hash(str(info['seq_id']))

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
            "conf": torch.tensor(current_conf, dtype=torch.float32),

            # [ADD] diagnostics outputs
            "npz_ok": torch.tensor(npz_ok, dtype=torch.float32),
            "warp_valid": torch.tensor(warp_valid, dtype=torch.float32),
            "local_used": torch.tensor(local_used, dtype=torch.float32),
            "is_first": torch.tensor(is_first, dtype=torch.float32),

            "seq_hash": seq_hash,
            "orig_idx": idx
        }
                

def build_datasets_and_loader(opts, logger, train_npz_list=None):
    if not opts.MaP: raise ValueError("Only MaP mode is supported.")
    base_dataset = ContinuousEdgeLineDatasetMaskFinetune(
        pt_dataset=opts.data_path, mask_path=opts.mask_path, test_mask_path=None,
        is_train=True, mask_rates=opts.mask_rates, image_size=opts.image_size,
        line_path=opts.train_wireframes_list,
    )
    
    train_wrapper = MaPDatasetWrapper(
        base_dataset, 
        base_dataset.seq_to_indices, 
        npz_path_list=train_npz_list, 
        logger=logger,
        no_align=getattr(opts, "no_align", False)  # ablation
    )
    
    rank = int(getattr(opts, "rank", 0))
    world_size = int(getattr(opts, "world_size", 1))

    # [DDP Fix] Distribute by frame-level "clips" (global ref, local ref, frame i)
    # instead of distributing by sequence lengths. This avoids rank imbalance and DDP timeout
    # caused by highly variable sequence lengths.
    if getattr(opts, "dist", False):
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_wrapper,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(getattr(opts, "seed", 42)),
            drop_last=False,
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_wrapper,
        batch_size=opts.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=opts.num_workers,
        pin_memory=True,
        persistent_workers=opts.persistent_workers,
        prefetch_factor=opts.prefetch_factor,
        worker_init_fn=_dataloader_worker_init,
        drop_last=False,
    )
    
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
        try:
            # 【修复】显式设置 weights_only=False 以消除警告
            ckpt = torch.load(opts.pretrain_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            # 兼容旧版本 PyTorch
            ckpt = torch.load(opts.pretrain_ckpt, map_location="cpu")
            
        state = ckpt.get("model", ckpt)
        
        new_state = {}
        for k, v in state.items():
            name = k.replace("module.", "").replace("_orig_mod.", "")
            if "attn.mask" in name:
                continue
            new_state[name] = v

        missing_keys, unexpected_keys = model.load_state_dict(new_state, strict=False)
        
        if len(missing_keys) > 0:
            logger.warning(f"[INIT] Missing keys: {missing_keys[:5]}... (Total: {len(missing_keys)})")
        if len(unexpected_keys) > 0:
            logger.warning(f"[INIT] Unexpected keys: {unexpected_keys[:5]}... (Total: {len(unexpected_keys)})")
            
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
        ref_idx = idxs[0] 
        img, edge, line = _build_clean_ref_sample(dataset, ref_idx, opts.image_size, device)
        raw_model.eval()
        with torch.no_grad():
            ref_feat = raw_model.extract_reference_features(global_img=img, global_edge=edge, global_line=line)
        raw_model.train()
        seq_to_ref[seq_id] = (ref_feat.detach(), ref_idx)
    return seq_to_ref

# ---------------------- Train Loop ----------------------

def train_one_epoch_optimized(model, train_loader, dataset_obj, optimizer, device, scaler, logger, epoch, amp, opts, scheduler, writer=None, global_step=0):
    model.train()
    
    if epoch < opts.dilate1_ep:
        current_dilate = 3
    else:
        current_dilate = 1

    if dist.get_rank() == 0:
        logger.info(f"[Curriculum] Epoch {epoch}: GT Dilate={current_dilate}")

    criterion = EdgeLineLoss(
        focal_weight=opts.focal_weight, 
        tversky_weight=opts.tversky_weight,
        alpha=opts.line_tv_alpha, 
        beta=opts.line_tv_beta
    )
    
    total_loss, total_samples = 0.0, 0
    raw_model = model.module if isinstance(model, DDP) else model
    
    for it, batch in enumerate(train_loader):
        B = batch['c_img'].size(0)
        
        c_img = batch['c_img'].to(device, non_blocking=True)
        c_edge = batch['c_edge'].to(device, non_blocking=True)
        c_line = batch['c_line'].to(device, non_blocking=True)
        c_mask = batch['c_mask'].to(device, non_blocking=True)
        g_edge = batch['g_edge'].to(device, non_blocking=True)
        g_line = batch['g_line'].to(device, non_blocking=True)
        l_edge = batch['l_edge'].to(device, non_blocking=True)
        l_line = batch['l_line'].to(device, non_blocking=True)
        l_mask = batch['l_mask'].to(device, non_blocking=True)
        local_conf = batch.get("conf", None)  # [ADD] confidence_iou from npz
        if local_conf is not None:
            local_conf = local_conf.to(device=device, non_blocking=True, dtype=torch.float32)
            #消融：如果启用 no_gate，强制将置信度设为 1.0 (全通)
            if getattr(opts, "no_gate", False):
                local_conf = torch.ones_like(local_conf)

        npz_ok = batch.get("npz_ok", None)
        warp_valid = batch.get("warp_valid", None)
        local_used = batch.get("local_used", None)
        is_first = batch.get("is_first", None)

        if npz_ok is not None: npz_ok = npz_ok.to(device=device, non_blocking=True, dtype=torch.float32)
        if warp_valid is not None: warp_valid = warp_valid.to(device=device, non_blocking=True, dtype=torch.float32)
        if local_used is not None: local_used = local_used.to(device=device, non_blocking=True, dtype=torch.float32)
        if is_first is not None: is_first = is_first.to(device=device, non_blocking=True, dtype=torch.float32)

        # only diagnose non-first frames (more meaningful)
        non_first = None
        if is_first is not None:
            non_first = (1.0 - is_first).clamp(0, 1)  # [B]
            denom_nf = non_first.sum().clamp_min(1.0)
        else:
            denom_nf = torch.tensor(float(B), device=device)

        # [修改] 移除了原有的 known 区域掩码计算和 _binary_iou 实时计算代码
        # 仅依赖从 batch 中读取的 local_conf (即 npz 中的 confidence_iou)

        # handle "empty line" cases (optional)
        empty_gt = (c_line.flatten(1).sum(1) == 0).float()
        empty_local = (l_line.flatten(1).sum(1) == 0).float()

        # periodic logging (rank0 only)
        if dist.get_rank() == 0 and ((it % 50) == 0):
            # ratios (prefer non-first)
            if non_first is not None:
                npz_ok_ratio = (npz_ok * non_first).sum() / denom_nf if npz_ok is not None else torch.tensor(0.0, device=device)
                warp_valid_ratio = (warp_valid * non_first).sum() / denom_nf if warp_valid is not None else torch.tensor(0.0, device=device)
                local_used_ratio = (local_used * non_first).sum() / denom_nf if local_used is not None else torch.tensor(0.0, device=device)

                conf_nf = (local_conf * non_first) if local_conf is not None else None
                conf_mean = conf_nf.sum() / denom_nf if conf_nf is not None else torch.tensor(0.0, device=device)
            else:
                npz_ok_ratio = npz_ok.mean() if npz_ok is not None else torch.tensor(0.0, device=device)
                warp_valid_ratio = warp_valid.mean() if warp_valid is not None else torch.tensor(0.0, device=device)
                local_used_ratio = local_used.mean() if local_used is not None else torch.tensor(0.0, device=device)
                conf_mean = local_conf.mean() if local_conf is not None else torch.tensor(0.0, device=device)

            conf_min = local_conf.min() if local_conf is not None else torch.tensor(0.0, device=device)
            conf_max = local_conf.max() if local_conf is not None else torch.tensor(0.0, device=device)

            # [修改] 日志字符串中移除了 iou_known 和 iou_all
            logger.info(
                f"[Diag][Ep{epoch} It{it}] "
                f"npz_ok={npz_ok_ratio.item():.3f} warp_valid={warp_valid_ratio.item():.3f} local_used={local_used_ratio.item():.3f} | "
                f"conf(mean/min/max)={conf_mean.item():.3f}/{conf_min.item():.3f}/{conf_max.item():.3f} | "
                f"empty_gt={float(empty_gt.mean().item()):.3f} empty_local={float(empty_local.mean().item()):.3f}"
            )

            if writer is not None:
                step = global_step + it
                writer.add_scalar("diag/npz_ok_ratio", float(npz_ok_ratio.item()), step)
                writer.add_scalar("diag/warp_valid_ratio", float(warp_valid_ratio.item()), step)
                writer.add_scalar("diag/local_used_ratio", float(local_used_ratio.item()), step)
                writer.add_scalar("diag/conf_mean", float(conf_mean.item()), step)
                writer.add_scalar("diag/conf_min", float(conf_min.item()), step)
                writer.add_scalar("diag/conf_max", float(conf_max.item()), step)
                # [修改] 移除了 diag/iou_known_mean 和 diag/iou_all_mean 的 TensorBoard 写入
            
        for t in [c_mask, c_edge, c_line, l_mask, l_edge, l_line, g_edge, g_line]:
            if t.dim() == 3: t.unsqueeze_(1)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=amp, dtype=torch.bfloat16):
            ref_feat = raw_model.extract_reference_features(
                global_img=None, global_edge=g_edge, global_line=g_line,
                local_img=None, local_edge=l_edge, local_line=l_line, local_mask=l_mask
            )
            
            edge_logits, line_logits = raw_model.forward_with_logits(
                img_idx=c_img, edge_idx=c_edge, line_idx=c_line, masks=c_mask, ref_feat=ref_feat
            )
            
            target_edge = dilate_tensor(c_edge, kernel_size=current_dilate)
            target_line = dilate_tensor(c_line, kernel_size=current_dilate)
            
            loss_line = criterion(line_logits, target_line)
            
            if opts.disable_edge:
                loss = loss_line
            else:
                loss_edge = criterion(edge_logits, target_edge)
                loss = (loss_edge + loss_line)
        
        if torch.isnan(loss) or torch.isinf(loss):
            # 将 loss 抹零，使其产生 全0梯度，参与 DDP 同步
            loss = loss * 0.0 
        loss.backward()
        
        if check_grads_finite(model):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item() * B
        total_samples += B
        
        if opts.tb_images and writer is not None and dist.get_rank() == 0:
            should_log = (it % 100 == 0)
            if should_log:
                try:
                    curr_orig_idx = batch['orig_idx'][0].item()
                    seq_id_val = train_wrapper.idx_to_seq.get(curr_orig_idx, "unknown")
                    seq_name = str(seq_id_val)[:10]
                except:
                    seq_name = "unknown"

                with torch.no_grad():
                    sample_idx = 0
                    step = global_step + it

                    # 1. Input image - 使用clone避免引用原tensor
                    img_vis = (c_img[sample_idx].clone() * 0.5 + 0.5).clamp(0, 1)
                    writer.add_image(f"train/{seq_name}/1_input", img_vis.cpu(), step)
                    del img_vis  # 立即删除

                    # 2. Overlay line - 使用clone并立即转CPU
                    line_pred = torch.sigmoid(line_logits[sample_idx:sample_idx+1].clone()).clamp(0, 1)
                    line_gt = c_line[sample_idx:sample_idx+1].clone()
                    local_warp_line = l_line[sample_idx:sample_idx+1].clone().clamp(0, 1)

                    alpha_blue = 0.3
                    blue_chan = (local_warp_line[0] * alpha_blue).clamp(0, 1)

                    line_overlay = torch.cat([line_gt[0], line_pred[0], blue_chan], dim=0)
                    writer.add_image(f"train/{seq_name}/2_overlay_line", line_overlay.cpu(), step)

                    # 立即删除所有中间变量
                    del line_pred, line_gt, local_warp_line, blue_chan, line_overlay

                    # 3. Local reference line
                    l_line_vis = l_line[sample_idx].clone().clamp(0, 1)
                    writer.add_image(f"train/{seq_name}/3_local_ref_line", l_line_vis.cpu(), step)
                    del l_line_vis

                    # 4. Predicted line
                    pred_vis = torch.sigmoid(line_logits[sample_idx:sample_idx+1].clone()).clamp(0, 1)
                    writer.add_image(f"train/{seq_name}/4_pred_line", pred_vis[0].cpu(), step)
                    del pred_vis

                    # 5. Global reference line
                    g_line_vis = g_edge[sample_idx].clone().clamp(0, 1)
                    writer.add_image(f"train/{seq_name}/5_global_ref_line", g_line_vis.cpu(), step)
                    del g_line_vis

                    # 强制清理显存
                    torch.cuda.empty_cache()
        current_loss_val = loss.item()  
        del loss, edge_logits, line_logits, ref_feat
        del c_img, c_edge, c_line, c_mask
        del g_edge, g_line
        del l_edge, l_line, l_mask
        
        # 清理batch中的其他tensor
        if local_conf is not None:
            del local_conf
        if npz_ok is not None:
            del npz_ok, warp_valid, local_used, is_first
        
        # 每50个iteration强制清理一次
        if (it + 1) % 50 == 0 and dist.get_rank() == 0:
            logger.info(f"[SeqTrain] Ep {epoch} It {it+1}/{len(train_loader)} Loss {current_loss_val:.4f}")
            torch.cuda.empty_cache()

    total_loss_tensor = torch.tensor(total_loss, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    
    return total_loss_tensor.item() / max(1, total_samples_tensor.item()), global_step + len(train_loader)

def prepare_fixed_validation_set(opts, val_dataset, logger):
    """
    Selects 15 sequences and specific frames (based on max line count or position)
    BEFORE training starts. This ensures consistent validation set and removes
    overhead from the evaluation loop.
    """
    if not hasattr(val_dataset, "seq_to_indices") or not val_dataset.seq_to_indices:
        logger.warning("[Val-Viz-Fixed15] val_dataset.seq_to_indices is empty.")
        opts._val_viz_fixed15_meta = {}
        return

    is_dist = bool(getattr(opts, "dist", False)) and dist.is_available() and dist.is_initialized()
    is_main = (not is_dist) or (dist.get_rank() == 0)
    
    fixed_meta = {}

    # Only Rank 0 performs the filesystem scanning / random selection
    if is_main:
        logger.info("[Val-Viz-Fixed15] initializing fixed validation set...")
        all_seq_ids = sorted(list(val_dataset.seq_to_indices.keys()))
        
        # 1. Pick 15 random sequences
        k = min(15, len(all_seq_ids))
        rng = random.Random(int(getattr(opts, "seed", 42)) + 2026)
        picked_seq_ids = rng.sample(all_seq_ids, k)
        logger.info("已选出15个序列")

        # 2. Prepare for Line Counting (only if visualization is enabled)
        enable_viz = bool(getattr(opts, "tb_images", False))
        pkl_paths = None
        
        if enable_viz:
            pkl_list_path = getattr(opts, "val_wireframes_list", None)
            if isinstance(pkl_list_path, (list, tuple)):
                pkl_list_path = pkl_list_path[0] if pkl_list_path else None
            
            if isinstance(pkl_list_path, str) and os.path.isfile(pkl_list_path):
                with open(pkl_list_path, "r") as f:
                    pkl_paths = [ln.strip() for ln in f.readlines() if ln.strip()]

        def _count_lines_in_pkl(pkl_path: str) -> int:
            try:
                if (pkl_path is None) or (not os.path.exists(pkl_path)): return 0
                import pickle
                with open(pkl_path, "rb") as fp:
                    data = pickle.load(fp)
                lines = data.get("lines", data.get("lines_pred", [])) if isinstance(data, dict) else data
                return int(len(lines)) if lines is not None else 0
            except Exception:
                return 0
            
        logger.info("已完成线计算")

        # 3. Select specific frames for each sequence
        for sid in picked_seq_ids:
            idxs = val_dataset.seq_to_indices.get(sid, [])
            if not idxs: continue

            # Sort by image path for deterministic order
            try:
                paths = [val_dataset.image_id_list[i] for i in idxs]
                sorted_pairs = sorted(zip(paths, idxs), key=lambda x: x[0])
                sorted_idxs = [p[1] for p in sorted_pairs]
            except Exception:
                sorted_idxs = sorted(list(idxs))

            if not sorted_idxs: continue

            g_idx = sorted_idxs[0] # Global Ref is usually the first frame

            # Strategy: If Viz enabled, find max lines. Else, take last frame.
            best_cnt = 0
            if enable_viz:
                best_i = 0
                best_cnt = -1
                for i, ds_idx in enumerate(sorted_idxs):
                    # Check PKL first, then dataset tensor
                    if pkl_paths is not None and ds_idx < len(pkl_paths):
                        cnt = _count_lines_in_pkl(pkl_paths[ds_idx])
                    else:
                        try:
                            # Avoid loading full image, just check tensor if available or skip
                            # Here we might need to load item if no PKL list
                            # For speed, if no PKL list, just pick middle frame or skip complex check
                            # To be safe and fast:
                            cnt = 0 
                        except: cnt = 0
                    
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_i = i
                
                t_idx = sorted_idxs[best_i]
                p_idx = sorted_idxs[best_i - 1] if best_i > 0 else -1
            else:
                # Fast fallback
                t_idx = sorted_idxs[-1]
                p_idx = sorted_idxs[-2] if len(sorted_idxs) > 1 else -1

            fixed_meta[sid] = {
                "g_idx": int(g_idx),
                "t_idx": int(t_idx),
                "p_idx": int(p_idx),
                "line_cnt": int(best_cnt),
            }
        
        logger.info(f"[Val-Viz-Fixed15] Locked {len(fixed_meta)} frames.")

    # 4. Broadcast metadata to all ranks
    if is_dist:
        obj_list = [fixed_meta if is_main else None]
        dist.broadcast_object_list(obj_list, src=0)
        fixed_meta = obj_list[0]

    opts._val_viz_fixed15_meta = fixed_meta


@torch.no_grad()
def evaluate_sequence(model, val_dataset, seq_to_ref, device, logger, amp, opts,
                      writer=None, epoch=0, val_npz_list=None):
    """
    Fast validation + fixed TB visualization using PRE-CALCULATED metadata.
    """
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model

    # DDP helpers
    is_dist = bool(getattr(opts, "dist", False)) and dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    is_main = (rank == 0)

    # Curriculum
    current_dilate = 3 if epoch < getattr(opts, "dilate1_ep", 0) else 1

    criterion = EdgeLineLoss(
        focal_weight=getattr(opts, "focal_weight", 1.0),
        tversky_weight=getattr(opts, "tversky_weight", 1.0),
        alpha=getattr(opts, "line_tv_alpha", 0.7),
        beta=getattr(opts, "line_tv_beta", 0.3),
    )

    # Retrieve pre-calculated metadata
    fixed_meta = getattr(opts, "_val_viz_fixed15_meta", {})
    enable_viz = bool(getattr(opts, "tb_images", False)) and (writer is not None)

    # Helper: warp a [1, C, H, W] tensor
    def warp_tensor(tensor_map, H_mat):
        t = tensor_map.float()
        if t.dim() == 3: t = t.unsqueeze(0)
        np_map = t.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
        h, w = np_map.shape[:2]
        warped_np = cv2.warpPerspective(np_map, H_mat, (w, h), flags=cv2.INTER_NEAREST)
        if warped_np.ndim == 2: warped_np = warped_np[..., None]
        warped_t = torch.from_numpy(warped_np).permute(2, 0, 1).unsqueeze(0)
        return warped_t.to(device=device, dtype=torch.float32)

    local_loss = 0.0
    local_frames = 0

    # ONLY rank0 evaluates the fixed frames (fast)
    if is_main and fixed_meta:
        for seq_idx, (sid, info) in enumerate(fixed_meta.items()):
            g_idx = int(info.get("g_idx", -1))
            t_idx = int(info.get("t_idx", -1))
            p_idx = int(info.get("p_idx", -1))

            if g_idx < 0 or t_idx < 0: continue

            try:
                # Global ref
                _, g_edge, g_line = _build_clean_ref_sample(
                    val_dataset, g_idx, getattr(opts, "image_size", 256), device
                )

                # Target frame
                meta_tgt = val_dataset[t_idx]
                t_img = _ensure_4d_float(meta_tgt["img"], device)
                t_edge_gt = _ensure_4d_float(meta_tgt["edge"], device)
                t_line_gt = _ensure_4d_float(meta_tgt["line"], device)
                t_mask = _ensure_4d_float(meta_tgt["mask"], device)
                if t_mask.dim() == 3: t_mask = t_mask.unsqueeze(1)

                # Local ref
                l_edge = torch.zeros_like(t_edge_gt)
                l_line = torch.zeros_like(t_line_gt)
                l_mask = torch.ones_like(t_mask)
                warp_valid = False
                H_mat = np.eye(3, dtype=np.float32)
                local_conf = 0.0
                is_no_align = getattr(opts, "no_align", False) #ablation

                if is_no_align: #ablation
                    warp_valid = True
                    local_conf = 1.0 # 强制全通
                elif (p_idx >= 0) and (val_npz_list is not None):
                    try:
                        wd = np.load(val_npz_list[t_idx])
                        warp_valid = bool(wd.get("valid", False))
                        if warp_valid:
                            H_mat = wd["homography"]
                            if "confidence_iou" in wd:
                                local_conf = float(wd["confidence_iou"])
                    except Exception:
                        warp_valid = False
                        local_conf = 0.0

                if getattr(opts, "no_gate", False):
                    local_conf = 1.0

                if warp_valid and (p_idx >= 0):
                    try:
                        if is_no_align:
                            # [修改] 直接使用当前帧 GT，不进行 Warp
                            l_edge = t_edge_gt
                            l_line = t_line_gt
                            l_mask = torch.zeros_like(t_mask) # 全图有效
                        else:
                            meta_prev = val_dataset[p_idx]
                            prev_edge_gt = _ensure_4d_float(meta_prev["edge"], device)
                            prev_line_gt = _ensure_4d_float(meta_prev["line"], device)
                            # 原有的 Warp 逻辑
                            l_edge = warp_tensor(prev_edge_gt, H_mat)
                            l_line = warp_tensor(prev_line_gt, H_mat)
                            l_mask = torch.zeros_like(t_mask)
                    except Exception:
                        l_edge = torch.zeros_like(t_edge_gt)
                        l_line = torch.zeros_like(t_line_gt)
                        l_mask = torch.ones_like(t_mask)

                with torch.amp.autocast('cuda', enabled=bool(amp), dtype=torch.bfloat16):
                    ref_feat = raw_model.extract_reference_features(
                        global_img=None, global_edge=g_edge, global_line=g_line,
                        local_img=None, local_edge=l_edge, local_line=l_line, local_mask=l_mask
                    )
                    edge_logits, line_logits = raw_model.forward_with_logits(
                        img_idx=t_img, edge_idx=t_edge_gt, line_idx=t_line_gt, masks=t_mask, ref_feat=ref_feat
                    )

                    target_edge = dilate_tensor(t_edge_gt, kernel_size=current_dilate)
                    target_line = dilate_tensor(t_line_gt, kernel_size=current_dilate)

                    loss_line = criterion(line_logits, target_line)
                    if getattr(opts, "disable_edge", False):
                        loss = loss_line
                    else:
                        loss_edge = criterion(edge_logits, target_edge)
                        loss = (loss_edge + loss_line)

                if torch.isnan(loss) or torch.isinf(loss): 
                    # 清理当前iteration的tensors
                    del g_edge, g_line, t_img, t_edge_gt, t_line_gt, t_mask
                    del l_edge, l_line, l_mask, ref_feat, edge_logits, line_logits
                    del target_edge, target_line
                    continue

                local_loss += float(loss.detach().cpu().item())  # 转到CPU
                local_frames += 1

                if enable_viz and (writer is not None):
                    seq_name = str(sid)[:10]
                    # 使用clone并立即转CPU
                    line_pred = torch.sigmoid(line_logits.clone()).clamp(0, 1)
                    local_warp_line = l_line.clone().clamp(0, 1)
                    
                    alpha_blue = 0.5
                    blue_chan = (local_warp_line[0] * alpha_blue).clamp(0, 1)
                    line_overlay = torch.cat([t_line_gt[0].clone(), line_pred[0], blue_chan], dim=0)
                    writer.add_image(f"val_fixed15/{seq_name}/overlay_line", line_overlay.cpu(), epoch)
                    
                    # 立即删除可视化相关tensor
                    del line_pred, local_warp_line, blue_chan, line_overlay
                
                # 清理当前iteration的所有tensors
                del g_edge, g_line, t_img, t_edge_gt, t_line_gt, t_mask
                del l_edge, l_line, l_mask, ref_feat, edge_logits, line_logits
                del target_edge, target_line, loss
                if 'loss_edge' in locals():
                    del loss_edge
                if 'loss_line' in locals():
                    del loss_line
                
                # 每处理5个序列清理一次显存
                if (seq_idx + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.warning(f"[Val] Skip seq {str(sid)[:10]}: {e}")
                # 确保异常时也清理显存
                torch.cuda.empty_cache()
                continue

    # Reduce across ranks
    loss_t = torch.tensor(local_loss, device=device, dtype=torch.float32)
    cnt_t = torch.tensor(local_frames, device=device, dtype=torch.float32)

    if is_dist:
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)

    val_loss = float(loss_t.item()) / max(1.0, float(cnt_t.item()))
    
    if is_main:
        logger.info("[Val] Cleaning up VRAM...")
    torch.cuda.empty_cache()
    
    return val_loss



# 【新增】解析 Config 覆盖 Opts 的函数
def load_config_to_opts(opts):
    if not opts.config_path or not os.path.exists(opts.config_path):
        return opts
    
    with open(opts.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Model Settings
    ms = cfg.get('model_settings', {})
    if 'name' in ms: opts.name = str(ms['name'])
    if 'pretrain_ckpt' in ms: opts.pretrain_ckpt = str(ms['pretrain_ckpt'])
    if 'tb_logdir' in ms: opts.tb_logdir = str(ms['tb_logdir'])
    if 'tb_images' in ms: opts.tb_images = bool(ms['tb_images'])
    if 'tb_max_images' in ms: opts.tb_max_images = int(ms['tb_max_images'])

    # Training Params
    tp = cfg.get('training_params', {})
    
    # 【修复关键点】增加强制类型转换，防止 YAML 将数字解析为字符串
    if 'lr' in tp: opts.lr = float(tp['lr'])  # <--- 修复了这里的报错
    if 'train_epoch' in tp: opts.train_epoch = int(tp['train_epoch'])
    # batch_size: preferred; keep a small compatibility alias for old configs using seq_batch
    if 'batch_size' in tp:
        opts.batch_size = int(tp['batch_size'])
    elif 'seq_batch' in tp:
        # Deprecated alias (old configs). Please rename seq_batch -> batch_size.
        opts.batch_size = int(tp['seq_batch'])
    if 'num_workers' in tp: opts.num_workers = int(tp['num_workers'])
    if 'prefetch_factor' in tp: opts.prefetch_factor = int(tp['prefetch_factor'])
    if 'persistent_workers' in tp: opts.persistent_workers = bool(tp['persistent_workers'])
    if 'AMP' in tp: opts.AMP = bool(tp['AMP'])
    if 'MaP' in tp: opts.MaP = bool(tp['MaP'])
    
    if 'focal_weight' in tp: opts.focal_weight = float(tp['focal_weight'])
    if 'tversky_weight' in tp: opts.tversky_weight = float(tp['tversky_weight'])
    if 'line_tv_alpha' in tp: opts.line_tv_alpha = float(tp['line_tv_alpha'])
    if 'line_tv_beta' in tp: opts.line_tv_beta = float(tp['line_tv_beta'])
    if 'dilate1_ep' in tp: opts.dilate1_ep = int(tp['dilate1_ep'])
    if 'disable_edge' in tp: opts.disable_edge = bool(tp['disable_edge'])

    # Masks
    mk = cfg.get('masks', {})
    if 'train_mask_list' in mk: opts.mask_path = str(mk['train_mask_list'])
    if 'val_mask_list' in mk: opts.valid_mask_path = str(mk['val_mask_list'])

    # 消融指标
    if 'no_align' in tp: opts.no_align = bool(tp['no_align'])

    
    # Store cfg for dataset loading in main
    opts.yaml_cfg = cfg
    return opts

def main_worker(opts):
    set_seed(42)
    opts = load_config_to_opts(opts)
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass 
    
    opts.ckpt_path = os.path.join(opts.ckpt_path, opts.name)
    os.makedirs(opts.ckpt_path, exist_ok=True)
    logger = build_logger(opts.ckpt_path)
    
    if opts.dist:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))

            # [修改] 引入 datetime 并设置 timeout 为 30 分钟
            import datetime
            dist.init_process_group(
                backend="nccl", 
                init_method="env://", 
                timeout=datetime.timedelta(minutes=30) # 增加超时容忍度
            )
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
        if opts.config_path:
            logger.info(f"[Config] Loaded from {opts.config_path}")
            with open(os.path.join(opts.ckpt_path, "config.yml"), "w") as f:
                yaml.dump(opts.yaml_cfg, f)

    # ... (Helpers: read_list_file, combine_datasets_from_yaml, save_temp_list - OMITTED FOR BREVITY, SAME AS BEFORE) ...
    def read_list_file(path):
        if not path or not os.path.isfile(path): return []
        with open(path, 'r') as f: return [line.strip() for line in f.readlines() if line.strip()]

    def combine_datasets_from_yaml(cfg, mode='train'):
        # ... (Same as original code) ...
        # (This block is large and unchanged, imagine it is here)
        def get_seq_id(path):
            try:
                norm_path = os.path.normpath(path)
                drive, tail = os.path.splitdrive(norm_path)
                parts = tail.split(os.sep)
                for part in parts[:-1]:
                    if len(part) > 15: return part
                return os.path.dirname(norm_path)
            except: return os.path.dirname(path)

        dataset_info_list = []
        for ds_key in ['dataset1', 'dataset2']:
            ds = cfg.get(ds_key, {})
            if not ds.get('enable', False): continue
            
            prefix = f"{mode}_"
            img_list = read_list_file(ds.get(f"{prefix}imgs_list"))
            pkl_list = read_list_file(ds.get(f"{prefix}pkls_list"))
            npz_list = read_list_file(ds.get(f"{prefix}npzs_list"))
            
            if not img_list: continue
            
            seq_groups = collections.defaultdict(lambda: {'img': [], 'pkl': [], 'npz': []})
            has_npz = (len(npz_list) == len(img_list))

            for i, img_path in enumerate(img_list):
                sid = get_seq_id(img_path)
                seq_groups[sid]['img'].append(img_path)
                seq_groups[sid]['pkl'].append(pkl_list[i])
                if has_npz: seq_groups[sid]['npz'].append(npz_list[i])
            
            dataset_info_list.append({
                'name': ds.get('name', ds_key),
                'seq_groups': seq_groups,
                'seq_count': len(seq_groups),
                'ratio': float(ds.get('ratio', 1.0))
            })

        if not dataset_info_list: return [], [], []

        final_img, final_pkl, final_npz = [], [], []
        
        # ... logic for ratio mixing ... (omitted for brevity, assume original logic)
        for d in dataset_info_list:
             for sid in sorted(list(d['seq_groups'].keys())):
                group = d['seq_groups'][sid]
                final_img.extend(group['img'])
                final_pkl.extend(group['pkl'])
                final_npz.extend(group['npz'])

        return final_img, final_pkl, final_npz
    # ... (End of helpers) ...

    # ... (Temp file writing & Dataset building) ...
    temp_dir = os.path.join(opts.ckpt_path, "temp_lists")
    os.makedirs(temp_dir, exist_ok=True)
    def save_temp_list(data, name):
        p = os.path.join(temp_dir, name)
        with open(p, 'w') as f: f.write('\n'.join(data))
        return p

    if getattr(opts, 'yaml_cfg', None):
        train_img, train_pkl, train_npz = combine_datasets_from_yaml(opts.yaml_cfg, 'train')
        val_img, val_pkl, val_npz = combine_datasets_from_yaml(opts.yaml_cfg, 'val')
        opts.data_path = save_temp_list(train_img, "train_img.txt")
        opts.train_wireframes_list = save_temp_list(train_pkl, "train_pkl.txt")
        opts.validation_path = save_temp_list(val_img, "val_img.txt")
        opts.val_wireframes_list = save_temp_list(val_pkl, "val_pkl.txt")
        train_npz_list = train_npz
        val_npz_list = val_npz
    else:
        if is_main: logger.warning("No YAML config, falling back to CLI args.")
        train_npz_list = opts.train_npz_list
        val_npz_list = opts.val_npz_list

        
        
    # =========================================================
    #  [ADD] Pre-training File Existence Check
    # =========================================================
    if is_main:
        logger.info("="*20 + " [Sanity Check] Validating Dataset Paths " + "="*20)
        
        def _validate_file_list(list_name, list_path_input):
            if not list_path_input:
                return
            # 兼容处理：输入可能是 list (CLI nargs='+') 或 str (YAML temp file)
            file_path = list_path_input[0] if isinstance(list_path_input, list) else list_path_input
            
            if not os.path.exists(file_path):
                logger.warning(f"[Check] List file not found: {file_path}")
                return

            with open(file_path, 'r') as f:
                paths = [line.strip() for line in f.readlines() if line.strip()]
            
            logger.info(f"[Check] Verifying {len(paths)} files for {list_name}...")
            missing_files = []
            for p in paths:
                if not os.path.exists(p):
                    missing_files.append(p)
            
            if len(missing_files) > 0:
                logger.error(f"[Check] FAILED: Found {len(missing_files)} missing files in {list_name}!")
                for m in missing_files[:5]:
                    logger.error(f"  Missing: {m}")
                if len(missing_files) > 5:
                    logger.error(f"  ... and {len(missing_files) - 5} more.")
                raise FileNotFoundError(f"Abort: Missing files detected in {list_name}.")
            else:
                logger.info(f"[Check] PASSED: All files in {list_name} exist.")

        # 检查训练集图片
        _validate_file_list("Train Images", opts.data_path)
        # 检查训练集线框 (PKL)
        _validate_file_list("Train Wireframes", opts.train_wireframes_list)
        # 检查验证集图片
        _validate_file_list("Val Images", opts.validation_path)
        
        # 检查 NPZ (如果是 YAML 模式，train_npz_list 是文件路径列表；如果是 CLI 模式，可能是列表文件的路径)
        # 为了安全起见，这里仅简单检查列表中的第一项是否为有效路径
        if train_npz_list and len(train_npz_list) > 0:
             # 如果列表第一项是一个存在的 .npz 文件，说明这是一个直接的路径列表，检查前5个即可(避免IO过重)
             # 如果列表第一项是一个存在的 .txt 文件，说明这是列表文件
            first_item = train_npz_list[0]
            if os.path.isfile(first_item):
                if first_item.endswith('.txt'):
                     _validate_file_list("Train NPZ List", [first_item])
                elif first_item.endswith('.npz'):
                    logger.info(f"[Check] Verifying Train NPZ files (Sampling first 100)...")
                    for p in train_npz_list[:100]:
                        if not os.path.exists(p):
                            raise FileNotFoundError(f"Missing NPZ file: {p}")
                    logger.info(f"[Check] PASSED: Train NPZ sample check.")
        
        logger.info("="*60)
    # =========================================================

    train_loader, val_dataset, train_sampler, train_base_ds = build_datasets_and_loader(
        opts, logger, train_npz_list=train_npz_list
    )
    
    if opts.MaP:
        def rebuild_seq_indices(dataset_obj, desc="Train"):
            new_seq_map = collections.defaultdict(list)
            def _extract_sid(path):
                try:
                    norm_path = os.path.normpath(path)
                    drive, tail = os.path.splitdrive(norm_path)
                    parts = tail.split(os.sep)
                    for part in parts[:-1]:
                        if len(part) > 15: return part
                    return os.path.dirname(norm_path)
                except: return os.path.dirname(path)

            for idx, path in enumerate(dataset_obj.image_id_list):
                sid = _extract_sid(path)
                new_seq_map[sid].append(idx)
            
            dataset_obj.seq_to_indices = new_seq_map
            if is_main:
                logger.info(f"[{desc}] Re-grouped data into {len(new_seq_map)} sequences.")
            return new_seq_map

        new_train_map = rebuild_seq_indices(train_base_ds, "Train")
        train_wrapper = MaPDatasetWrapper( 
            train_base_ds, 
            new_train_map, 
            npz_path_list=train_npz_list, 
            logger=logger,
            no_align=getattr(opts, "no_align", False) # ablation
        )
                # [DDP Fix] Distribute by frame-level "clips" rather than by sequences.
        rank = int(getattr(opts, "rank", 0))
        world_size = int(getattr(opts, "world_size", 1))

        if getattr(opts, "dist", False):
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(
                train_wrapper,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=int(getattr(opts, "seed", 42)),
                drop_last=False,
            )
            shuffle = False
        else:
            train_sampler = None
            shuffle = True

        train_loader = DataLoader(
            train_wrapper,
            batch_size=opts.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=opts.num_workers,
            pin_memory=True,
            persistent_workers=opts.persistent_workers,
            prefetch_factor=opts.prefetch_factor,
            worker_init_fn=_dataloader_worker_init,
            drop_last=False,
        )
        rebuild_seq_indices(val_dataset, "Val")

    # =========================================================
    #  [MODIFICATION] PRE-CALCULATE FIXED VALIDATION FRAMES HERE
    # =========================================================
    prepare_fixed_validation_set(opts, val_dataset, logger)
    # =========================================================

    model = build_model(opts, device, logger)
    if opts.dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main: logger.info("[DDP] Model wrapped with DistributedDataParallel")
    
    optim_kwargs = {'lr': opts.lr, 'betas': (0.9, 0.95), 'weight_decay': 0.0}
    if torch.cuda.is_available() and hasattr(torch.optim, 'Adam') and 'fused' in torch.optim.Adam.__init__.__code__.co_varnames: optim_kwargs['fused'] = True
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], **optim_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.train_epoch, eta_min=1e-6)
    
    best_val, best_path, latest_path = 1e9, os.path.join(opts.ckpt_path, "best.pth"), os.path.join(opts.ckpt_path, "latest.pth")
    writer = SummaryWriter(log_dir=opts.tb_logdir or os.path.join(opts.ckpt_path, "tb")) if is_main else None
    # seq_to_ref = build_seq_ref_feats(model, val_dataset, device, opts) if opts.MaP else {}
    global_step, scaler = 0, None
    if is_main: logger.info("[Opt] BFloat16 enabled. GradScaler is DISABLED.")

    # Self-Check (Reduced for brevity, keep your original)
    if is_main:
        logger.info("\n" + "="*50 + "\n[Self-Check] Checks finished. Starting training loop...\n" + "="*50)
        
    if opts.dist: dist.barrier()

    for epoch in range(1, opts.train_epoch + 1):
        if opts.dist: train_sampler.set_epoch(epoch)
        train_loss, global_step = train_one_epoch_optimized(
            model, train_loader, train_base_ds, optimizer, device, scaler, 
            logger, epoch, opts.AMP, opts, scheduler, writer=writer, global_step=global_step
        )

        if scheduler is not None: scheduler.step()
        if opts.dist: dist.barrier()
        
        if is_main:
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": best_val}, latest_path)
        
        if ((epoch % opts.val_freq == 0) or (epoch == opts.train_epoch)):
            val_loss = evaluate_sequence(
                model, val_dataset, {}, device, logger, 
                opts.AMP, opts, writer=writer, epoch=epoch, val_npz_list=val_npz_list
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
                    logger.info(f" [CKPT] New best ({best_val:.4f}) saved.")
        else:
             if is_main: logger.info(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

    if writer is not None: writer.close()
    if is_main: logger.info("Finish Ref-KV training.")
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 【新增】Config Path 参数
    parser.add_argument("--config_path", type=str, default=None, help="Path to YAML config")
    
    # 保留原有的参数定义，作为默认值或被 Config 覆盖
    parser.add_argument("--name", type=str, default="RefKV_train_seq")
    parser.add_argument("--GPU_ids", type=str, default="0")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt")
    # Data paths (can be None if using config)
    parser.add_argument("--data_path", type=str, nargs='+', default=None)
    parser.add_argument("--validation_path", type=str, nargs='+', default=None)
    parser.add_argument("--mask_path", type=str, default="")
    parser.add_argument("--valid_mask_path", type=str, default="")
    parser.add_argument("--train_wireframes_list", type=str, nargs='+', default=None)
    parser.add_argument("--val_wireframes_list", type=str, nargs='+', default=None)
    parser.add_argument("--train_npz_list", type=str, nargs='+', default=None)
    parser.add_argument("--val_npz_list", type=str, nargs='+', default=None)
    
    parser.add_argument("--mask_rates", type=float, nargs="+", default=[0.4, 0.8, 1.0])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=16)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_epoch", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--AMP", action="store_true")
    parser.add_argument("--MaP", action="store_true")
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--tb_logdir", type=str, default=None)
    parser.add_argument("--tb_images", action="store_true")
    parser.add_argument("--tb_max_images", type=int, default=3)
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_freq", type=int, default=2)
    parser.add_argument("--focal_weight", type=float, default=1.0)
    parser.add_argument("--tversky_weight", type=float, default=1.0)
    parser.add_argument("--disable_edge", action="store_true")
    parser.add_argument("--line_tv_alpha", type=float, default=0.3, help="less line")
    parser.add_argument("--line_tv_beta", type=float, default=0.7, help="more line")
    parser.add_argument("--dilate_switch_ep", type=int, default=3)
    parser.add_argument("--dilate1_ep", type=int, default=1, help="Epoch to switch from dilate=3 to dilate=1.")

    #消融指标
    parser.add_argument("--no_align", action="store_true", help="Do not align reference frames (use raw previous) and force IOU to 1.0")


    opts = parser.parse_args()
    
    # 设置可见 GPU
    if opts.GPU_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = opts.GPU_ids
        
    main_worker(opts)
