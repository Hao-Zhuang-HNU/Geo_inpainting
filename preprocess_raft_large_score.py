#!/usr/bin/env python3
# preprocess_raft.py
# High-throughput RAFT preprocess with torchvision RAFT-small / RAFT-large + version-safe forward call + auto micro-batching.

import os
import argparse
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from contextlib import nullcontext

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F


# -----------------------------
# Small LRU cache (dynamic size)
# -----------------------------
class LRUCache:
    def __init__(self, maxsize: int):
        self.maxsize = max(0, int(maxsize))
        self.od = OrderedDict()

    def get(self, key):
        if self.maxsize <= 0:
            return None
        v = self.od.get(key, None)
        if v is None:
            return None
        self.od.move_to_end(key)
        return v

    def put(self, key, value):
        if self.maxsize <= 0:
            return
        if key in self.od:
            self.od.move_to_end(key)
        self.od[key] = value
        if len(self.od) > self.maxsize:
            self.od.popitem(last=False)


# -----------------------------
# IO
# -----------------------------
def cv2_read_rgb(path: str) -> Optional[np.ndarray]:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_tensor_chw_01(rgb: np.ndarray) -> torch.Tensor:
    # uint8 HWC -> float32 CHW in [0,1]
    t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return t.float().div_(255.0)


def normalize_minus1_1(img_chw_01: torch.Tensor) -> torch.Tensor:
    # [0,1] -> [-1,1]
    return (img_chw_01 - 0.5) / 0.5


def ceil_to_divisor(x: int, d: int) -> int:
    return int((x + d - 1) // d) * d


def pad_to_hw_bchw(img_bchw: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, h, w = img_bchw.shape
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h == 0 and pad_w == 0:
        return img_bchw
    return F.pad(img_bchw, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


# -----------------------------
# Lines / IoU
# -----------------------------
def get_hawp_lines_proven(pkl_path: str, img_w: int, img_h: int) -> np.ndarray:
    """
    Load lines from pkl and convert to OpenCV xyxy pixel coordinates.
    Supports dict with 'lines' or 'lines_pred'.
    HAWP often stores [y1,x1,y2,x2] -> [x1,y1,x2,y2].
    If normalized (<=1), scale to pixels.
    """
    if not pkl_path:
        return np.zeros((0, 4), dtype=np.float32)
    p = Path(pkl_path)
    if not p.exists():
        return np.zeros((0, 4), dtype=np.float32)

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            lines = data.get("lines", data.get("lines_pred", []))
        else:
            lines = data
        lines = np.asarray(lines)
        if lines.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        if lines.ndim != 2:
            lines = lines.reshape(-1, 4)
        lines = lines.astype(np.float32)

        lines_cv = lines.copy()
        lines_cv[:, [0, 1, 2, 3]] = lines_cv[:, [1, 0, 3, 2]]  # yx -> xy

        if float(lines_cv.max()) <= 1.05:
            lines_cv[:, [0, 2]] *= float(img_w)
            lines_cv[:, [1, 3]] *= float(img_h)

        return lines_cv
    except Exception:
        return np.zeros((0, 4), dtype=np.float32)


def bilinear_sample_flow(flow_hw2: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    H, W, _ = flow_hw2.shape
    x = pts_xy[:, 0].astype(np.float32)
    y = pts_xy[:, 1].astype(np.float32)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    Ia = flow_hw2[y0, x0]
    Ib = flow_hw2[y1, x0]
    Ic = flow_hw2[y0, x1]
    Id = flow_hw2[y1, x1]

    wa = (x1.astype(np.float32) - x) * (y1.astype(np.float32) - y)
    wb = (x1.astype(np.float32) - x) * (y - y0.astype(np.float32))
    wc = (x - x0.astype(np.float32)) * (y1.astype(np.float32) - y)
    wd = (x - x0.astype(np.float32)) * (y - y0.astype(np.float32))

    wa = wa[:, None]
    wb = wb[:, None]
    wc = wc[:, None]
    wd = wd[:, None]

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out.astype(np.float32)


def warp_lines_by_flow(lines_xyxy: np.ndarray, flow_hw2: np.ndarray) -> np.ndarray:
    if lines_xyxy.size == 0:
        return lines_xyxy.astype(np.float32)
    pts = lines_xyxy.reshape(-1, 2).astype(np.float32)
    d = bilinear_sample_flow(flow_hw2, pts)
    pts_w = pts + d
    return pts_w.reshape(-1, 4).astype(np.float32)


def raster_mask(lines_xyxy: np.ndarray, w: int, h: int, scale: float) -> np.ndarray:
    sh = max(1, int(round(h * scale)))
    sw = max(1, int(round(w * scale)))
    canvas = np.zeros((sh, sw), dtype=np.uint8)
    if lines_xyxy.size == 0:
        return canvas
    L = (lines_xyxy * scale).astype(np.float32)
    for l in L:
        cv2.line(canvas, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), 1, 1)
    return canvas


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0



def mask_precision_recall(mask_warp: np.ndarray, mask_curr: np.ndarray) -> Tuple[float, float]:
    """Precision/Recall between two binary masks (uint8 0/1)."""
    inter = int(np.logical_and(mask_warp, mask_curr).sum())
    a = int(mask_warp.sum())
    b = int(mask_curr.sum())
    precision = float(inter) / float(a) if a > 0 else 0.0
    recall = float(inter) / float(b) if b > 0 else 0.0
    return precision, recall


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def _distance_transform_to_lines(lines_xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    """Distance (in pixels) to the nearest line pixel. Returns float32 (h,w)."""
    if lines_xyxy.size == 0:
        return np.full((h, w), 1e6, dtype=np.float32)
    img = np.full((h, w), 255, dtype=np.uint8)  # non-zero background
    L = lines_xyxy.astype(np.float32)
    for l in L:
        x1 = int(np.clip(round(float(l[0])), 0, w - 1))
        y1 = int(np.clip(round(float(l[1])), 0, h - 1))
        x2 = int(np.clip(round(float(l[2])), 0, w - 1))
        y2 = int(np.clip(round(float(l[3])), 0, h - 1))
        cv2.line(img, (x1, y1), (x2, y2), 0, 1)  # zero pixels are "targets"
    dt = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    return dt.astype(np.float32, copy=False)


def _mean_dt_at_endpoints(lines_xyxy: np.ndarray, dt_hw: np.ndarray) -> float:
    if lines_xyxy.size == 0:
        return 1e6
    h, w = dt_hw.shape[:2]
    pts = lines_xyxy.reshape(-1, 2).astype(np.float32)
    xs = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, h - 1)
    vals = dt_hw[ys, xs]
    return float(vals.mean()) if vals.size > 0 else 1e6


def endpoint_sym_error(prev_warp_xyxy: np.ndarray, curr_lines_xyxy: np.ndarray, w: int, h: int) -> float:
    """Symmetric endpoint distance using distance transforms (cheap & robust)."""
    if (prev_warp_xyxy.size == 0) or (curr_lines_xyxy.size == 0):
        return 1e6
    dt_curr = _distance_transform_to_lines(curr_lines_xyxy, w, h)
    dt_prev = _distance_transform_to_lines(prev_warp_xyxy, w, h)
    e_prev2curr = _mean_dt_at_endpoints(prev_warp_xyxy, dt_curr)
    e_curr2prev = _mean_dt_at_endpoints(curr_lines_xyxy, dt_prev)
    return 0.5 * (float(e_prev2curr) + float(e_curr2prev))


def composite_score_scheme_a(precision: float, recall: float, endpoint_err: float,
                             p0: float = 0.5, slope: float = 0.1, tau: float = 2.5) -> float:
    """
    Scheme-A reliability score:
      S = sigmoid((P-p0)/slope) * sigmoid((R-p0)/slope) * exp(-endpoint_err/tau)
    """
    p = float(np.clip(precision, 0.0, 1.0))
    r = float(np.clip(recall, 0.0, 1.0))
    e = max(0.0, float(endpoint_err))
    sp = _sigmoid((p - p0) / slope)
    sr = _sigmoid((r - p0) / slope)
    return float(sp * sr * np.exp(-e / tau))

# -----------------------------
# RAFT
# -----------------------------

def load_raft_model(device: torch.device, raft_model: str = "small", weights_name: str = "DEFAULT"):
    """
    Load torchvision RAFT model (small/large) in a version-tolerant way.

    Notes
    -----
    * We normalize manually; do NOT call weights.transforms() because torchvision's OpticalFlow preset expects (img1, img2).
    * weights_name: DEFAULT / other enum names / NONE
    """
    raft_model = str(raft_model).lower().strip()

    # Try to import both small and large. Older torchvision may only provide raft_small.
    try:
        from torchvision.models.optical_flow import raft_small, raft_large, Raft_Small_Weights, Raft_Large_Weights
    except Exception:
        try:
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            raft_large = None
            Raft_Large_Weights = None
        except Exception as e:
            raise RuntimeError(
                "torchvision optical_flow RAFT is not available. "
                "Please ensure torchvision is installed with optical_flow models."
            ) from e

    if raft_model in ("small", "raft_small", "raft-small"):
        ctor = raft_small
        weights_enum = Raft_Small_Weights
        model_name = "raft_small"
    elif raft_model in ("large", "raft_large", "raft-large"):
        if raft_large is None:
            raise RuntimeError("Your torchvision build does not include raft_large. Please upgrade torchvision.")
        ctor = raft_large
        weights_enum = Raft_Large_Weights
        model_name = "raft_large"
    else:
        raise ValueError(f"Unknown raft_model={raft_model!r}. Use 'small' or 'large'.")

    weights = None
    if (weights_name is not None) and (str(weights_name).upper() != "NONE"):
        wn = str(weights_name).upper()
        if weights_enum is not None:
            weights = getattr(weights_enum, wn, None)
            if weights is None:
                weights = getattr(weights_enum, "DEFAULT", None)

    model = ctor(weights=weights, progress=True).to(device).eval()
    try:
        model._refkv_raft_name = model_name
    except Exception:
        pass
    return model



def autocast_context(device: torch.device, amp: bool):
    if not amp:
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    # New API: torch.amp.autocast("cuda", ...)
    try:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    except Exception:
        # Old fallback
        return torch.cuda.amp.autocast(dtype=torch.float16)


@torch.inference_mode()
def raft_forward_safe(model, img1_bchw: torch.Tensor, img2_bchw: torch.Tensor, iters: int) -> torch.Tensor:
    """
    Version-safe RAFT forward call:
    - try iters=...
    - try num_flow_updates=...
    - try positional third arg
    - fallback to no arg
    Returns last flow tensor (B,2,H,W).
    """
    # try keyword iters
    try:
        out = model(img1_bchw, img2_bchw, iters=iters)
        return out[-1] if isinstance(out, (list, tuple)) else out
    except TypeError:
        pass

    # try keyword num_flow_updates
    try:
        out = model(img1_bchw, img2_bchw, num_flow_updates=iters)
        return out[-1] if isinstance(out, (list, tuple)) else out
    except TypeError:
        pass

    # try positional third arg
    try:
        out = model(img1_bchw, img2_bchw, iters)
        return out[-1] if isinstance(out, (list, tuple)) else out
    except TypeError:
        pass

    # fallback
    out = model(img1_bchw, img2_bchw)
    return out[-1] if isinstance(out, (list, tuple)) else out


@torch.inference_mode()
def raft_infer_batch(model, img1_bchw: torch.Tensor, img2_bchw: torch.Tensor,
                     iters: int = 8, amp: bool = True, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    with autocast_context(device, amp):
        return raft_forward_safe(model, img1_bchw, img2_bchw, iters)


def flow_2hw_to_hw2(flow_2hw: torch.Tensor) -> np.ndarray:
    f = flow_2hw.detach().float().cpu().numpy()
    return np.transpose(f, (1, 2, 0)).astype(np.float32)


def downsample_flow_hw2(flow_hw2: np.ndarray, ds: int, out_dtype=None):
    """
    Downsample flow to (H/ds, W/ds) and scale vectors by 1/ds.
    IMPORTANT: OpenCV resize may NOT support float16 with INTER_AREA, so we resize in float32.
    """
    if ds <= 1:
        flow_out = flow_hw2
        if out_dtype is not None and flow_out.dtype != out_dtype:
            flow_out = flow_out.astype(out_dtype, copy=False)
        return flow_out, np.array([1, 1], dtype=np.int32)

    H, W, _ = flow_hw2.shape
    h2 = max(1, H // ds)
    w2 = max(1, W // ds)

    # resize in float32 for compatibility
    fx = cv2.resize(flow_hw2[..., 0].astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA) / float(ds)
    fy = cv2.resize(flow_hw2[..., 1].astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA) / float(ds)

    flow_ds = np.stack([fx, fy], axis=-1)

    if out_dtype is not None:
        flow_ds = flow_ds.astype(out_dtype, copy=False)
    return flow_ds, np.array([ds, ds], dtype=np.int32)



# -----------------------------
# Streaming scan (avoid loading 67w lines into RAM)
# -----------------------------
def scan_common_root_and_count(img_list_path: str) -> Tuple[str, int]:
    common = None
    n = 0
    with open(img_list_path, "r") as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            if common is None:
                common = p
            else:
                common = os.path.commonpath([common, p])
            n += 1
    if common is None or n == 0:
        raise ValueError("img_list is empty.")
    return common, n


def count_nonempty_lines(path: str) -> int:
    n = 0
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("RAFT preprocess (max throughput)")

    parser.add_argument("--img_list", required=True)
    parser.add_argument("--pkl_list", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("--common_root", default="", help="Optional; auto computed if empty")

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=190, help="Try big; auto micro-batch will split on OOM")
    parser.add_argument("--iters", type=int, default=12)

    parser.add_argument("--raft_model", type=str, default="large", choices=["small", "large"],
                        help="RAFT backbone: small (fast) or large (higher accuracy).")

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)

    parser.add_argument("--weights", type=str, default="DEFAULT", help="DEFAULT / C_T_V2 / C_T_V1 / NONE")
    parser.add_argument("--compile", action="store_true", help="torch.compile for a bit more speed (stable shapes best)")

    # IO & saving
    parser.add_argument("--io_threads", type=int, default=16)
    parser.add_argument("--writer_threads", type=int, default=1)
    parser.add_argument("--no_compress", action="store_true", help="Use np.savez instead of np.savez_compressed (MUCH faster)")

    # iou/caches
    parser.add_argument("--no_iou", action="store_true")
    parser.add_argument("--iou_scale", type=float, default=0.25)
    parser.add_argument("--iou_thr", type=float, default=0.05)
    parser.add_argument("--pkl_cache", type=int, default=16384)
    parser.add_argument("--mask_cache", type=int, default=32768)

    # store flow (optional; huge disk if full res)
    parser.add_argument("--store_flow", type=str, default="none", choices=["none", "fp16", "fp32"])
    parser.add_argument("--flow_downscale", type=int, default=1)

    args = parser.parse_args()

    # OpenCV threads off (we control concurrency)
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    # device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    # perf settings
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # common root + totals
    if args.common_root:
        common_root = args.common_root
        total = count_nonempty_lines(args.img_list)
    else:
        common_root, total = scan_common_root_and_count(args.img_list)

    total_pkl = count_nonempty_lines(args.pkl_list)
    if total_pkl != total:
        raise ValueError(f"img_list lines={total}, pkl_list lines={total_pkl} (must match).")

    os.makedirs(args.output_dir, exist_ok=True)

    # model
    model = load_raft_model(device=device, raft_model=args.raft_model, weights_name=args.weights)
    if args.compile:
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception:
            pass

    # caches
    lines_cache = LRUCache(args.pkl_cache)
    mask_cache = LRUCache(args.mask_cache)
    scale_int = int(round(args.iou_scale * 1000))

    def load_lines(pkl_path: str, w: int, h: int) -> np.ndarray:
        key = (pkl_path, w, h)
        v = lines_cache.get(key)
        if v is not None:
            return v
        arr = get_hawp_lines_proven(pkl_path, w, h)
        lines_cache.put(key, arr)
        return arr

    def load_curr_mask(pkl_path: str, w: int, h: int) -> np.ndarray:
        key = (pkl_path, w, h, scale_int)
        v = mask_cache.get(key)
        if v is not None:
            return v
        lines = load_lines(pkl_path, w, h)
        m = raster_mask(lines, w, h, args.iou_scale)
        mask_cache.put(key, m)
        return m

    # async writer
    write_q: "queue.Queue[Optional[Tuple[str, Dict[str, Any]]]]" = queue.Queue(maxsize=512)
    save_fn = np.savez if args.no_compress else np.savez_compressed

    def writer_worker():
        while True:
            item = write_q.get()
            if item is None:
                write_q.task_done()
                break
            save_path, payload = item
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_fn(save_path, **payload)
            write_q.task_done()

    writers = []
    for _ in range(max(1, args.writer_threads)):
        th = threading.Thread(target=writer_worker, daemon=True)
        th.start()
        writers.append(th)

    # IO pool
    io_pool = ThreadPoolExecutor(max_workers=max(1, args.io_threads))

    def read_pair(prev_path: str, curr_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return cv2_read_rgb(prev_path), cv2_read_rgb(curr_path)

    # micro-batch safe runner
    def infer_with_auto_split(prev_list: List[torch.Tensor], curr_list: List[torch.Tensor]) -> torch.Tensor:
        """
        prev_list/curr_list are CHW tensors normalized to [-1,1] on CPU.
        Returns flow (B,2,H,W) on GPU.
        Auto-splits on OOM and retries on CUDNN_STATUS_NOT_SUPPORTED (non-contiguous).
        """
        B = len(prev_list)
        assert B == len(curr_list) and B > 0

        prev_b = torch.stack(prev_list, dim=0).contiguous()
        curr_b = torch.stack(curr_list, dim=0).contiguous()

        Hm, Wm = prev_b.shape[-2], prev_b.shape[-1]
        Hp = ceil_to_divisor(Hm, 8)
        Wp = ceil_to_divisor(Wm, 8)
        if (Hp, Wp) != (Hm, Wm):
            prev_b = pad_to_hw_bchw(prev_b, Hp, Wp).contiguous()
            curr_b = pad_to_hw_bchw(curr_b, Hp, Wp).contiguous()

        if device.type == "cuda":
            prev_b = prev_b.pin_memory()
            curr_b = curr_b.pin_memory()

        prev_b = prev_b.to(device, non_blocking=True).contiguous()
        curr_b = curr_b.to(device, non_blocking=True).contiguous()

        def _run(x1, x2, amp_flag: bool):
            return raft_infer_batch(model, x1, x2, iters=args.iters, amp=amp_flag, device=device)

        try:
            return _run(prev_b, curr_b, args.amp)

        except RuntimeError as e:
            msg = str(e).lower()

            # 1) cuDNN non-contiguous / not supported: force contiguous and retry once
            if ("cudnn_status_not_supported" in msg) or ("non-contiguous" in msg):
                prev_b2 = prev_b.contiguous()
                curr_b2 = curr_b.contiguous()
                try:
                    return _run(prev_b2, curr_b2, args.amp)
                except RuntimeError as e2:
                    # If AMP triggers cuDNN unsupported issues, retry in FP32 for stability.
                    msg2 = str(e2).lower()
                    if args.amp and (("cudnn_status_not_supported" in msg2) or ("non-contiguous" in msg2)):
                        return _run(prev_b2, curr_b2, False)
                    raise

            # 2) OOM / illegal access: split batch
            if device.type == "cuda":
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()

            if ("out of memory" in msg) or ("cuda error" in msg) or ("illegal memory access" in msg):
                if B == 1:
                    raise
                mid = B // 2
                left = infer_with_auto_split(prev_list[:mid], curr_list[:mid])
                right = infer_with_auto_split(prev_list[mid:], curr_list[mid:])
                return torch.cat([left, right], dim=0)

            raise


    # flush batch (compute + enqueue saves)
    def flush_batch(tasks: List[Tuple[str, str, str, str, str]]):
        """
        tasks: list of (prev_img, curr_img, prev_pkl, curr_pkl, save_path)
        """
        if not tasks:
            return

        futures = [io_pool.submit(read_pair, t[0], t[1]) for t in tasks]
        loaded = [f.result() for f in futures]

        prev_t_list: List[torch.Tensor] = []
        curr_t_list: List[torch.Tensor] = []
        metas = []  # (H,W, prev_pkl, curr_pkl, save_path)

        for (task, (rgb_prev, rgb_curr)) in zip(tasks, loaded):
            prev_img, curr_img, prev_pkl, curr_pkl, save_path = task

            if rgb_prev is None or rgb_curr is None:
                payload = dict(
                    homography=np.eye(3, dtype=np.float32),
                    mkpts0=np.zeros((0, 2), dtype=np.float32),
                    mkpts1=np.zeros((0, 2), dtype=np.float32),
                    confidence_match=0.0,
                    confidence_iou=0.0,
                    valid=False,
                )
                if args.store_flow != "none":
                    dt = np.float16 if args.store_flow == "fp16" else np.float32
                    payload["flow"] = np.zeros((0, 0, 2), dtype=dt)
                    payload["flow_ds"] = np.array([1, 1], dtype=np.int32)
                write_q.put((save_path, payload))
                continue

            h1, w1 = rgb_prev.shape[:2]
            h2, w2 = rgb_curr.shape[:2]
            H = min(h1, h2)
            W = min(w1, w2)
            if (h1 != H) or (w1 != W):
                rgb_prev = cv2.resize(rgb_prev, (W, H), interpolation=cv2.INTER_LINEAR)
            if (h2 != H) or (w2 != W):
                rgb_curr = cv2.resize(rgb_curr, (W, H), interpolation=cv2.INTER_LINEAR)

            # 256x256 fast path: no resize; normalize only
            t_prev = normalize_minus1_1(to_tensor_chw_01(rgb_prev))
            t_curr = normalize_minus1_1(to_tensor_chw_01(rgb_curr))

            prev_t_list.append(t_prev)
            curr_t_list.append(t_curr)
            metas.append((H, W, prev_pkl, curr_pkl, save_path))

        if not metas:
            return

        # run RAFT with auto split if needed
        flow_b2hw = infer_with_auto_split(prev_t_list, curr_t_list)

        # save per sample
        for i in range(len(metas)):
            H, W, prev_pkl, curr_pkl, save_path = metas[i]

            flow_2hw = flow_b2hw[i, :, :H, :W]
            flow_hw2 = flow_2hw_to_hw2(flow_2hw)

            # Reliability score (Scheme A): precision/recall overlap + symmetric endpoint distance
            if args.no_iou:
                score = 0.0
                iou_raw = 0.0
                pr_p = 0.0
                pr_r = 0.0
                endpoint_err = 1e6
                is_valid = True
            else:
                curr_mask = load_curr_mask(curr_pkl, W, H)
                prev_lines = load_lines(prev_pkl, W, H)
                curr_lines = load_lines(curr_pkl, W, H)
            
                if (prev_lines.size == 0) or (curr_lines.size == 0):
                    pr_p, pr_r = 0.0, 0.0
                    endpoint_err = 1e6
                    iou_raw = 0.0
                    score = 0.0
                else:
                    prev_warp = warp_lines_by_flow(prev_lines, flow_hw2)
                    prev_mask = raster_mask(prev_warp, W, H, args.iou_scale)
                    iou_raw = mask_iou(prev_mask, curr_mask)
                    pr_p, pr_r = mask_precision_recall(prev_mask, curr_mask)
                    endpoint_err = endpoint_sym_error(prev_warp, curr_lines, W, H)
                    score = composite_score_scheme_a(pr_p, pr_r, endpoint_err)
            
                is_valid = (score > float(args.iou_thr))

            payload = dict(
                homography=np.eye(3, dtype=np.float32),
                mkpts0=np.zeros((0, 2), dtype=np.float32),
                mkpts1=np.zeros((0, 2), dtype=np.float32),
                confidence_match=1.0,
                confidence_iou=float(score),
                iou_raw=float(iou_raw),
                pr_precision=float(pr_p),
                pr_recall=float(pr_r),
                endpoint_err=float(endpoint_err),
                valid=bool(is_valid),
                flow_hw=np.array([H, W], dtype=np.int32),
            )

            # optional flow saving (disk heavy)
            if args.store_flow != "none":
                dt = np.float16 if args.store_flow == "fp16" else np.float32
                # ALWAYS downscale in float32, then cast to dt for storage
                flow_f32 = flow_hw2.astype(np.float32, copy=False)
                flow_store, flow_meta = downsample_flow_hw2(flow_f32, args.flow_downscale, out_dtype=dt)
                payload["flow"] = flow_store
                payload["flow_ds"] = flow_meta

            write_q.put((save_path, payload))

        # cleanup GPU tensors
        del flow_b2hw
        if device.type == "cuda":
            torch.cuda.synchronize()

    # main loop (stream)
    pbar = tqdm(total=total, desc="RAFT preprocess", ncols=110)

    prev_img = ""
    prev_pkl = ""
    prev_dir = ""
    batch_tasks: List[Tuple[str, str, str, str, str]] = []

    with open(args.img_list, "r") as f_img, open(args.pkl_list, "r") as f_pkl:
        for img_line, pkl_line in zip(f_img, f_pkl):
            img_path = img_line.strip()
            pkl_path = pkl_line.strip()
            if not img_path or not pkl_path:
                continue

            curr_dir = os.path.dirname(img_path)
            is_first = (prev_img == "") or (curr_dir != prev_dir)

            rel = os.path.relpath(img_path, common_root)
            save_path = os.path.join(args.output_dir, os.path.splitext(rel)[0] + ".npz")

            if is_first:
                # write empty for first frame (if missing)
                if not os.path.exists(save_path):
                    payload = dict(
                        homography=np.eye(3, dtype=np.float32),
                        mkpts0=np.zeros((0, 2), dtype=np.float32),
                        mkpts1=np.zeros((0, 2), dtype=np.float32),
                        confidence_match=0.0,
                        confidence_iou=0.0,
                        valid=False,
                    )
                    if args.store_flow != "none":
                        dt = np.float16 if args.store_flow == "fp16" else np.float32
                        payload["flow"] = np.zeros((0, 0, 2), dtype=dt)
                        payload["flow_ds"] = np.array([1, 1], dtype=np.int32)
                    write_q.put((save_path, payload))

                prev_img = img_path
                prev_pkl = pkl_path
                prev_dir = curr_dir
                pbar.update(1)
                continue

            # non-first: schedule compute (prev -> curr) if missing
            if not os.path.exists(save_path):
                batch_tasks.append((prev_img, img_path, prev_pkl, pkl_path, save_path))
                if len(batch_tasks) >= max(1, args.batch_size):
                    flush_batch(batch_tasks)
                    batch_tasks.clear()

            prev_img = img_path
            prev_pkl = pkl_path
            prev_dir = curr_dir
            pbar.update(1)

    # flush tail
    if batch_tasks:
        flush_batch(batch_tasks)
        batch_tasks.clear()

    # finalize writers
    write_q.join()
    for _ in writers:
        write_q.put(None)
    write_q.join()
    for th in writers:
        th.join()

    io_pool.shutdown(wait=True)
    pbar.close()
    print("[Done] RAFT preprocess finished.")


if __name__ == "__main__":
    main()