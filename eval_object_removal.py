#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_object_removal_mv_metrics.py

Object-removal / multi-view inpainting evaluation focused on:
  LPIPS ↓
  VFID ↓
  DINO-A ↑
  DINO-M ↑
  Ewarp_hole ↓
  Line-F1 ↑
  Chamfer Distance ↓

This script is adapted from the user's eval_video.py style: natural-order pairing,
optional resizing, mask-index alignment, VFID proxy with torchvision video models,
Farneback-based Ewarp, and JSON/CSV/debug outputs.

Expected directory inputs:
  --pre_path       predicted/restored frames
  --gt_path        clean background / GT frames
  --mask_path      object/hole masks, white=hole, black=known
  --pred_line_path optional predicted/restored line maps; if omitted, lines are extracted from --pre_path images
  --gt_line_path   optional GT line maps; if omitted, lines are extracted from --gt_path images

Notes:
  1) DINO-A/DINO-M are computed as pairwise cosine similarities among per-frame
     masked-region DINO descriptors. For each frame, DINO patch tokens inside the
     hole mask are averaged and L2-normalized. DINO-A is the average pairwise
     similarity; DINO-M is the minimum pairwise similarity.
  2) Line-F1/Chamfer are computed on binary line maps. If line map directories are
     not supplied, Canny edges are extracted from images as an approximation.
  3) Ewarp_hole uses optical flow between consecutive GT frames by default and
     measures the temporal warp error of predicted frames inside the hole mask.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from scipy import linalg
from scipy.ndimage import binary_dilation, distance_transform_edt

try:
    import cv2
except Exception:
    cv2 = None

try:
    import lpips  # pip install lpips
except Exception:
    lpips = None

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ----------------------- basic utilities -----------------------
def natural_key(p: Path) -> List:
    s = p.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=natural_key)
    return files


def pil_to_np_rgb01(pil_img: Image.Image, size: Optional[int] = None) -> np.ndarray:
    img = pil_img.convert("RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def load_mask(mask_path: Path, size: Optional[int] = None) -> np.ndarray:
    m = Image.open(mask_path).convert("L")
    if size is not None and m.size != (size, size):
        m = m.resize((size, size), resample=Image.NEAREST)
    arr = np.asarray(m).astype(np.float32) / 255.0
    return (arr > 0.5).astype(np.float32)  # HW, 1=hole


def load_binary_line(line_path: Path, size: Optional[int] = None, thresh: int = 127) -> np.ndarray:
    im = Image.open(line_path).convert("L")
    if size is not None and im.size != (size, size):
        im = im.resize((size, size), resample=Image.NEAREST)
    arr = np.asarray(im).astype(np.uint8)
    # Support both white-line-on-black and black-line-on-white conventions.
    # If the image is mostly white, assume black pixels are lines.
    if float(np.mean(arr > thresh)) > 0.70:
        line = (arr < thresh).astype(np.uint8)
    else:
        line = (arr > thresh).astype(np.uint8)
    return line


def extract_canny_line_from_image(img_rgb01: np.ndarray, low: int = 80, high: int = 160) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for Canny line extraction. Install opencv-python or provide line maps.")
    u8 = (np.clip(img_rgb01, 0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return (edges > 0).astype(np.uint8)


def collect_masks_index_aligned(mask_root: Path, seq_name: str) -> List[Path]:
    cand_dir = mask_root / seq_name
    if cand_dir.exists():
        return list_images(cand_dir)
    allm = list_images(mask_root)
    filt = [p for p in allm if seq_name in p.as_posix().split("/")]
    if len(filt) > 0:
        filt.sort(key=natural_key)
        return filt
    return allm


# ----------------------- LPIPS -----------------------
class LPIPSMetric:
    def __init__(self, device: torch.device, net: str = "alex"):
        if lpips is None:
            raise RuntimeError("lpips is not installed. Install with: pip install lpips")
        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred_rgb01: np.ndarray, gt_rgb01: np.ndarray) -> float:
        p = torch.from_numpy(pred_rgb01).permute(2, 0, 1).unsqueeze(0).to(self.device)
        g = torch.from_numpy(gt_rgb01).permute(2, 0, 1).unsqueeze(0).to(self.device)
        p = p * 2.0 - 1.0
        g = g * 2.0 - 1.0
        return float(self.model(p, g).item())


# ----------------------- Frechet distance / VFID proxy -----------------------
def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


class VideoBackbone(nn.Module):
    def __init__(self, name: str, device: torch.device):
        super().__init__()
        if name == "r3d_18":
            weights = torchvision.models.video.R3D_18_Weights.DEFAULT
            self.net = torchvision.models.video.r3d_18(weights=weights)
        elif name == "mc3_18":
            weights = torchvision.models.video.MC3_18_Weights.DEFAULT
            self.net = torchvision.models.video.mc3_18(weights=weights)
        elif name == "r2plus1d_18":
            weights = torchvision.models.video.R2Plus1D_18_Weights.DEFAULT
            self.net = torchvision.models.video.r2plus1d_18(weights=weights)
        else:
            raise ValueError(f"Unknown VFID backbone: {name}")
        self.net.fc = nn.Identity()
        self.net.eval().to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


def make_clips(n_frames: int, clip_len: int, stride: int) -> List[List[int]]:
    if n_frames < clip_len:
        return []
    return [list(range(s, s + clip_len)) for s in range(0, n_frames - clip_len + 1, stride)]


def read_clip_uint8_T_H_W_C(paths: List[Path], pix: Optional[int]) -> np.ndarray:
    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if pix is not None and img.size != (pix, pix):
            img = img.resize((pix, pix), Image.BILINEAR)
        frames.append(np.asarray(img).astype(np.uint8))
    return np.stack(frames, axis=0)


def preprocess_video_batch_uint8(bthwc: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = bthwc.to(device).float() / 255.0  # B,T,H,W,C
    x = x.permute(0, 4, 1, 2, 3).contiguous()  # B,C,T,H,W
    B, C, T, H, W = x.shape
    x2 = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    x2 = F.interpolate(x2, size=(112, 112), mode="bilinear", align_corners=False)
    x = x2.reshape(B, T, C, 112, 112).permute(0, 2, 1, 3, 4).contiguous()
    x = (x - KINETICS_MEAN.to(device)) / KINETICS_STD.to(device)
    return x


@torch.no_grad()
def compute_vfid_proxy(
    pred_frames: List[Path],
    gt_frames: List[Path],
    vb: VideoBackbone,
    device: torch.device,
    pix: Optional[int],
    clip_len: int,
    stride: int,
    batch_size: int,
) -> float:
    clips = make_clips(min(len(pred_frames), len(gt_frames)), clip_len, stride)
    if len(clips) < 2:
        return float("nan")

    pred_feats, gt_feats = [], []
    for i in tqdm(range(0, len(clips), batch_size), desc="VFID clips"):
        batch = clips[i:i + batch_size]
        pred_batch, gt_batch = [], []
        for inds in batch:
            pred_batch.append(read_clip_uint8_T_H_W_C([pred_frames[j] for j in inds], pix))
            gt_batch.append(read_clip_uint8_T_H_W_C([gt_frames[j] for j in inds], pix))
        pred_x = preprocess_video_batch_uint8(torch.from_numpy(np.stack(pred_batch, axis=0)), device)
        gt_x = preprocess_video_batch_uint8(torch.from_numpy(np.stack(gt_batch, axis=0)), device)
        pred_feats.append(vb(pred_x).cpu().numpy())
        gt_feats.append(vb(gt_x).cpu().numpy())

    P = np.concatenate(pred_feats, axis=0)
    G = np.concatenate(gt_feats, axis=0)
    return compute_frechet_distance(np.mean(P, axis=0), np.cov(P, rowvar=False),
                                    np.mean(G, axis=0), np.cov(G, rowvar=False))


# ----------------------- DINO-A / DINO-M -----------------------
class DINOv2MaskedDescriptor:
    def __init__(self, device: torch.device, model_name: str = "dinov2_vits14", repo: str = "facebookresearch/dinov2", size: int = 224):
        self.device = device
        self.size = int(size)
        if self.size % 14 != 0:
            raise ValueError("--dino_size must be divisible by 14 for DINOv2 patch tokens.")
        self.model = torch.hub.load(repo, model_name)
        self.model.eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.grid = self.size // 14

    @torch.no_grad()
    def descriptor(self, img_path: Path, mask_path: Optional[Path]) -> Optional[np.ndarray]:
        img = Image.open(img_path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = (x - self.mean) / self.std

        out = self.model.forward_features(x)
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            toks = out["x_norm_patchtokens"]  # 1,N,D
        elif isinstance(out, dict) and "patch_tokens" in out:
            toks = out["patch_tokens"]
        else:
            raise RuntimeError("Cannot find DINO patch tokens from forward_features output.")
        toks = toks.squeeze(0)  # N,D

        if mask_path is not None and Path(mask_path).exists():
            m = Image.open(mask_path).convert("L").resize((self.grid, self.grid), Image.NEAREST)
            m = (np.asarray(m).astype(np.float32) / 255.0 > 0.5).reshape(-1)
            if int(m.sum()) == 0:
                m = np.ones_like(m, dtype=bool)
        else:
            m = np.ones((self.grid * self.grid,), dtype=bool)

        mtoks = toks[torch.from_numpy(m).to(self.device)]
        if mtoks.numel() == 0:
            return None
        desc = mtoks.mean(dim=0)
        desc = F.normalize(desc, dim=0)
        return desc.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def compute_dino_a_m(
    pred_frames: List[Path],
    mask_frames: List[Optional[Path]],
    dino: DINOv2MaskedDescriptor,
) -> Tuple[float, float, int]:
    descs = []
    for p, m in tqdm(list(zip(pred_frames, mask_frames)), desc="DINO descriptors"):
        d = dino.descriptor(p, m)
        if d is not None:
            descs.append(d)
    if len(descs) < 2:
        return float("nan"), float("nan"), len(descs)
    D = np.stack(descs, axis=0)
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    sims = []
    for i in range(D.shape[0]):
        for j in range(i + 1, D.shape[0]):
            sims.append(float(np.dot(D[i], D[j])))
    return float(np.mean(sims)), float(np.min(sims)), len(descs)


# ----------------------- Optical flow + Ewarp_hole -----------------------
def farneback_flow(im1_rgb01: np.ndarray, im2_rgb01: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for Ewarp. Install opencv-python.")
    g1 = cv2.cvtColor((im1_rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor((im2_rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    return flow.astype(np.float32)


def warp_with_flow(img_rgb01: np.ndarray, flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for Ewarp. Install opencv-python.")
    h, w = img_rgb01.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xs + flow[..., 0]).astype(np.float32)
    map_y = (ys + flow[..., 1]).astype(np.float32)
    valid = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    warped = cv2.remap((img_rgb01 * 255).astype(np.uint8), map_x, map_y,
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0).astype(np.float32) / 255.0
    return warped, valid.astype(np.float32)


def compute_ewarp_hole(
    pred_frames: List[Path],
    gt_frames: List[Path],
    mask_frames: List[Optional[Path]],
    pix: Optional[int],
    flow_source: str = "gt",
) -> float:
    vals = []
    n = min(len(pred_frames), len(gt_frames), len(mask_frames))
    if n < 2:
        return float("nan")
    for t in tqdm(range(n - 1), desc="Ewarp_hole"):
        pred_t = pil_to_np_rgb01(Image.open(pred_frames[t]), size=pix)
        pred_tp1 = pil_to_np_rgb01(Image.open(pred_frames[t + 1]), size=pix)
        gt_t = pil_to_np_rgb01(Image.open(gt_frames[t]), size=pix)
        gt_tp1 = pil_to_np_rgb01(Image.open(gt_frames[t + 1]), size=pix)

        im1, im2 = (gt_t, gt_tp1) if flow_source == "gt" else (pred_t, pred_tp1)
        flow = farneback_flow(im1, im2)
        warped_pred_tp1, valid = warp_with_flow(pred_tp1, flow)
        diff2 = np.mean((pred_t - warped_pred_tp1) ** 2, axis=2)

        mpath = mask_frames[t]
        if mpath is None or not Path(mpath).exists():
            continue
        mask = load_mask(Path(mpath), size=pix)
        denom = float(np.sum(valid * mask))
        if denom > 1:
            vals.append(float(np.sum(diff2 * valid * mask) / denom))
    return float(np.mean(vals)) if len(vals) else float("nan")


# ----------------------- Line-F1 / Chamfer -----------------------
def line_metrics_one(
    pred_line: np.ndarray,
    gt_line: np.ndarray,
    region: Optional[np.ndarray],
    tol: int = 2,
) -> Tuple[float, float, float, float, int, int]:
    pred = (pred_line > 0).astype(bool)
    gt = (gt_line > 0).astype(bool)
    if region is not None:
        r = region.astype(bool)
        pred = np.logical_and(pred, r)
        gt = np.logical_and(gt, r)

    n_pred = int(pred.sum())
    n_gt = int(gt.sum())
    if n_pred == 0 and n_gt == 0:
        return 1.0, 1.0, 1.0, 0.0, n_pred, n_gt
    if n_pred == 0 or n_gt == 0:
        return 0.0, 0.0, 0.0, float("nan"), n_pred, n_gt

    dist_to_gt = distance_transform_edt(~gt)
    dist_to_pred = distance_transform_edt(~pred)

    precision = float(np.mean(dist_to_gt[pred] <= tol))
    recall = float(np.mean(dist_to_pred[gt] <= tol))
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    cd_pred_to_gt = float(np.mean(dist_to_gt[pred]))
    cd_gt_to_pred = float(np.mean(dist_to_pred[gt]))
    chamfer = 0.5 * (cd_pred_to_gt + cd_gt_to_pred)
    return precision, recall, float(f1), float(chamfer), n_pred, n_gt


def compute_line_metrics(
    pred_frames: List[Path],
    gt_frames: List[Path],
    mask_frames: List[Optional[Path]],
    pred_line_frames: Optional[List[Path]],
    gt_line_frames: Optional[List[Path]],
    pix: Optional[int],
    tol: int,
    region_mode: str,
    dilate_iter: int,
    line_thresh: int,
    canny_low: int,
    canny_high: int,
) -> Tuple[Dict[str, float], List[Dict]]:
    records = []
    precs, recs, f1s, cds = [], [], [], []
    n = min(len(pred_frames), len(gt_frames), len(mask_frames))
    if pred_line_frames is not None:
        n = min(n, len(pred_line_frames))
    if gt_line_frames is not None:
        n = min(n, len(gt_line_frames))

    for i in tqdm(range(n), desc="Line metrics"):
        if pred_line_frames is not None:
            pline = load_binary_line(pred_line_frames[i], size=pix, thresh=line_thresh)
        else:
            pimg = pil_to_np_rgb01(Image.open(pred_frames[i]), size=pix)
            pline = extract_canny_line_from_image(pimg, low=canny_low, high=canny_high)

        if gt_line_frames is not None:
            gline = load_binary_line(gt_line_frames[i], size=pix, thresh=line_thresh)
        else:
            gimg = pil_to_np_rgb01(Image.open(gt_frames[i]), size=pix)
            gline = extract_canny_line_from_image(gimg, low=canny_low, high=canny_high)

        region = None
        mpath = mask_frames[i]
        if region_mode != "all" and mpath is not None and Path(mpath).exists():
            mask = load_mask(Path(mpath), size=pix).astype(bool)
            if region_mode == "hole":
                region = mask.astype(np.uint8)
            elif region_mode == "dilate_hole":
                region = binary_dilation(mask, iterations=dilate_iter).astype(np.uint8)
            elif region_mode == "ring":
                dil = binary_dilation(mask, iterations=dilate_iter)
                region = np.logical_and(dil, ~mask).astype(np.uint8)
            else:
                raise ValueError(f"Unknown region_mode: {region_mode}")

        p, r, f1, cd, npred, ngt = line_metrics_one(pline, gline, region, tol=tol)
        precs.append(p)
        recs.append(r)
        f1s.append(f1)
        cds.append(cd)
        records.append({
            "eval_index": i,
            "pred_frame": str(pred_frames[i]),
            "gt_frame": str(gt_frames[i]),
            "mask": str(mask_frames[i]) if mask_frames[i] is not None else None,
            "pred_line": str(pred_line_frames[i]) if pred_line_frames is not None else "Canny(pred_frame)",
            "gt_line": str(gt_line_frames[i]) if gt_line_frames is not None else "Canny(gt_frame)",
            "Line_Precision": p,
            "Line_Recall": r,
            "Line_F1": f1,
            "Chamfer_Distance": cd,
            "pred_line_pixels": npred,
            "gt_line_pixels": ngt,
        })

    metrics = {
        "Line_Precision": float(np.nanmean(precs)) if len(precs) else float("nan"),
        "Line_Recall": float(np.nanmean(recs)) if len(recs) else float("nan"),
        "Line_F1": float(np.nanmean(f1s)) if len(f1s) else float("nan"),
        "Chamfer_Distance": float(np.nanmean(cds)) if len(cds) else float("nan"),
    }
    return metrics, records


# ----------------------- main -----------------------
def apply_temporal_jump(
    pred_frames: List[Path],
    gt_frames: List[Path],
    mask_frames: List[Optional[Path]],
    pred_line_frames: Optional[List[Path]],
    gt_line_frames: Optional[List[Path]],
    jump_image: int,
    jump_mask: int,
) -> Tuple[List[Path], List[Path], List[Optional[Path]], Optional[List[Path]], Optional[List[Path]]]:
    if jump_image < 0 or jump_mask < 0:
        raise ValueError("--jump_image and --jump_mask must be non-negative")
    pred_start = jump_image
    gt_start = jump_image + jump_mask
    mask_start = jump_mask

    lengths = [len(pred_frames) - pred_start, len(gt_frames) - gt_start, len(mask_frames) - mask_start]
    if pred_line_frames is not None:
        lengths.append(len(pred_line_frames) - pred_start)
    if gt_line_frames is not None:
        lengths.append(len(gt_line_frames) - gt_start)
    n = min(lengths)
    if n <= 0:
        raise RuntimeError("No valid pairs remain after temporal jump.")

    pred_frames = pred_frames[pred_start:pred_start + n]
    gt_frames = gt_frames[gt_start:gt_start + n]
    mask_frames = mask_frames[mask_start:mask_start + n]
    if pred_line_frames is not None:
        pred_line_frames = pred_line_frames[pred_start:pred_start + n]
    if gt_line_frames is not None:
        gt_line_frames = gt_line_frames[gt_start:gt_start + n]
    return pred_frames, gt_frames, mask_frames, pred_line_frames, gt_line_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_path", type=str, required=True, help="predicted/restored frame directory")
    parser.add_argument("--gt_path", type=str, required=True, help="clean background / GT frame directory")
    parser.add_argument("--mask_path", type=str, required=True, help="hole/object mask directory, white=hole")
    parser.add_argument("--pred_line_path", type=str, default="", help="optional predicted/restored line map directory")
    parser.add_argument("--gt_line_path", type=str, default="", help="optional GT line map directory")
    parser.add_argument("--seq_name", type=str, default="", help="optional mask subdirectory name; default=gt_path basename")

    parser.add_argument("--pix", type=int, default=256)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="object_removal_metrics_out")

    parser.add_argument("--jump_image", type=int, default=0,
                        help="skip first N pred/gt images: pred[N]/gt[N] <-> mask[0]")
    parser.add_argument("--jump_mask", type=int, default=0,
                        help="skip first N gt/mask frames: pred[0] <-> gt[N] <-> mask[N]")

    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--skip_lpips", action="store_true")

    parser.add_argument("--vfid_backbone", type=str, default="r3d_18", choices=["r3d_18", "mc3_18", "r2plus1d_18"])
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--clip_stride", type=int, default=8)
    parser.add_argument("--vfid_batch_size", type=int, default=8)
    parser.add_argument("--skip_vfid", action="store_true")

    parser.add_argument("--dino_repo", type=str, default="facebookresearch/dinov2")
    parser.add_argument("--dino_model", type=str, default="dinov2_vits14")
    parser.add_argument("--dino_size", type=int, default=224)
    parser.add_argument("--skip_dino", action="store_true")

    parser.add_argument("--flow_source", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--skip_ewarp", action="store_true")

    parser.add_argument("--line_tol", type=int, default=2, help="pixel tolerance for line precision/recall")
    parser.add_argument("--line_region", type=str, default="dilate_hole", choices=["all", "hole", "dilate_hole", "ring"])
    parser.add_argument("--line_region_dilate", type=int, default=5)
    parser.add_argument("--line_thresh", type=int, default=127)
    parser.add_argument("--canny_low", type=int, default=80)
    parser.add_argument("--canny_high", type=int, default=160)
    parser.add_argument("--skip_line", action="store_true")

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pred_root = Path(args.pre_path)
    gt_root = Path(args.gt_path)
    mask_root = Path(args.mask_path)
    assert pred_root.exists(), f"pre_path not found: {pred_root}"
    assert gt_root.exists(), f"gt_path not found: {gt_root}"
    assert mask_root.exists(), f"mask_path not found: {mask_root}"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    pix = args.pix if args.resize else None

    pred_frames = list_images(pred_root)
    gt_frames = list_images(gt_root)
    if len(pred_frames) == 0 or len(gt_frames) == 0:
        raise RuntimeError("No images found under --pre_path or --gt_path")

    n0 = min(len(pred_frames), len(gt_frames))
    if len(pred_frames) != len(gt_frames):
        print(f"[WARN] pred={len(pred_frames)} gt={len(gt_frames)} -> index-align first N={n0}")
    pred_frames = pred_frames[:n0]
    gt_frames = gt_frames[:n0]

    seq_name = args.seq_name.strip() if args.seq_name.strip() else gt_root.name
    masks = collect_masks_index_aligned(mask_root, seq_name)
    if len(masks) == 0:
        raise RuntimeError("No masks found under --mask_path")
    if len(masks) < n0:
        print(f"[WARN] masks={len(masks)} < frames={n0}; missing masks will be ignored by truncation")
    if len(masks) > n0:
        print(f"[INFO] masks={len(masks)} > frames={n0}; take first N masks by natural order")
    mask_frames: List[Optional[Path]] = [m for m in masks[:min(n0, len(masks))]]

    # Keep all streams at the same initial length before temporal jump.
    n0 = min(len(pred_frames), len(gt_frames), len(mask_frames))
    pred_frames = pred_frames[:n0]
    gt_frames = gt_frames[:n0]
    mask_frames = mask_frames[:n0]

    pred_line_frames = list_images(Path(args.pred_line_path)) if args.pred_line_path else None
    gt_line_frames = list_images(Path(args.gt_line_path)) if args.gt_line_path else None
    if pred_line_frames is not None and len(pred_line_frames) == 0:
        raise RuntimeError("--pred_line_path was provided but no line images were found")
    if gt_line_frames is not None and len(gt_line_frames) == 0:
        raise RuntimeError("--gt_line_path was provided but no line images were found")

    pred_frames, gt_frames, mask_frames, pred_line_frames, gt_line_frames = apply_temporal_jump(
        pred_frames, gt_frames, mask_frames, pred_line_frames, gt_line_frames,
        args.jump_image, args.jump_mask
    )

    n = min(len(pred_frames), len(gt_frames), len(mask_frames))
    pred_frames = pred_frames[:n]
    gt_frames = gt_frames[:n]
    mask_frames = mask_frames[:n]
    print(f"[INFO] Evaluation frames: {n}")
    print(f"[INFO] Pair rule after jump: pred[i+{args.jump_image}] <-> gt[i+{args.jump_image + args.jump_mask}] <-> mask[i+{args.jump_mask}]")

    debug_records = []
    if args.debug:
        for i in range(n):
            debug_records.append({
                "eval_index": i,
                "pred_path": str(pred_frames[i]),
                "gt_path": str(gt_frames[i]),
                "mask_path": str(mask_frames[i]),
                "pred_name": pred_frames[i].name,
                "gt_name": gt_frames[i].name,
                "mask_name": Path(mask_frames[i]).name if mask_frames[i] is not None else None,
            })

    # LPIPS
    lpips_value = float("nan")
    if not args.skip_lpips:
        try:
            metric = LPIPSMetric(device=device, net=args.lpips_net)
            vals = []
            for p, g in tqdm(list(zip(pred_frames, gt_frames)), desc="LPIPS"):
                pred = pil_to_np_rgb01(Image.open(p), size=pix)
                gt = pil_to_np_rgb01(Image.open(g), size=pix)
                vals.append(metric(pred, gt))
            lpips_value = float(np.mean(vals)) if len(vals) else float("nan")
        except Exception as e:
            print(f"[WARN] LPIPS failed: {e}")

    # VFID
    vfid_value = float("nan")
    if not args.skip_vfid:
        try:
            vb = VideoBackbone(args.vfid_backbone, device=device)
            vfid_value = compute_vfid_proxy(pred_frames, gt_frames, vb, device, pix,
                                            args.clip_len, args.clip_stride, args.vfid_batch_size)
        except Exception as e:
            print(f"[WARN] VFID failed: {e}")

    # DINO-A / DINO-M
    dino_a, dino_m, dino_count = float("nan"), float("nan"), 0
    if not args.skip_dino:
        try:
            dino = DINOv2MaskedDescriptor(device=device, model_name=args.dino_model,
                                          repo=args.dino_repo, size=args.dino_size)
            dino_a, dino_m, dino_count = compute_dino_a_m(pred_frames, mask_frames, dino)
        except Exception as e:
            print(f"[WARN] DINO-A/DINO-M failed: {e}")

    # Ewarp_hole
    ewarp_hole = float("nan")
    if not args.skip_ewarp:
        try:
            ewarp_hole = compute_ewarp_hole(pred_frames, gt_frames, mask_frames, pix, flow_source=args.flow_source)
        except Exception as e:
            print(f"[WARN] Ewarp_hole failed: {e}")

    # Line-F1 / Chamfer
    line_metrics = {
        "Line_Precision": float("nan"),
        "Line_Recall": float("nan"),
        "Line_F1": float("nan"),
        "Chamfer_Distance": float("nan"),
    }
    line_records: List[Dict] = []
    if not args.skip_line:
        try:
            line_metrics, line_records = compute_line_metrics(
                pred_frames, gt_frames, mask_frames, pred_line_frames, gt_line_frames, pix,
                tol=args.line_tol, region_mode=args.line_region, dilate_iter=args.line_region_dilate,
                line_thresh=args.line_thresh, canny_low=args.canny_low, canny_high=args.canny_high,
            )
        except Exception as e:
            print(f"[WARN] Line metrics failed: {e}")

    results = {
        "count_frames": int(n),
        "LPIPS": float(lpips_value),
        "VFID": float(vfid_value),
        "VFID_backbone": args.vfid_backbone,
        "DINO_A": float(dino_a),
        "DINO_M": float(dino_m),
        "DINO_descriptor_frames": int(dino_count),
        "Ewarp_hole": float(ewarp_hole),
        **line_metrics,
        "line_region": args.line_region,
        "line_tol": int(args.line_tol),
        "flow_source": args.flow_source,
        "resize": bool(args.resize),
        "pix": int(args.pix) if args.resize else None,
        "jump_image": int(args.jump_image),
        "jump_mask": int(args.jump_mask),
    }

    print("\n========== Object Removal Metrics ==========")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k:>24s}: {v:.6f}")
        else:
            print(f"{k:>24s}: {v}")
    print("===========================================\n")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)

    if args.debug:
        with open(out_dir / "debug_alignment.json", "w", encoding="utf-8") as f:
            json.dump(debug_records, f, indent=2, ensure_ascii=False)
        if len(debug_records) > 0:
            with open(out_dir / "debug_alignment.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(debug_records[0].keys()))
                writer.writeheader()
                writer.writerows(debug_records)

    if len(line_records) > 0:
        with open(out_dir / "line_metrics_per_frame.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(line_records[0].keys()))
            writer.writeheader()
            writer.writerows(line_records)

    print(f"Saved: {out_dir / 'metrics.json'}")
    print(f"Saved: {out_dir / 'metrics.csv'}")
    if len(line_records) > 0:
        print(f"Saved: {out_dir / 'line_metrics_per_frame.csv'}")


if __name__ == "__main__":
    main()
