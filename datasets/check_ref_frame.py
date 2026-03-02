import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


def _tensor_to_bin_np(x: torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor/map, got shape={x.shape}")
    return (x > 0.5).astype(np.uint8)


def _chamfer_distance_binary(a_bin: np.ndarray, b_bin: np.ndarray) -> float:
    """Symmetric Chamfer Distance between two binary maps.
    Uses distance transform for nearest-neighbor distance.
    """
    a_pts = int(a_bin.sum())
    b_pts = int(b_bin.sum())
    if a_pts == 0 and b_pts == 0:
        return 0.0
    if a_pts == 0 or b_pts == 0:
        return float("inf")

    # Distance-to-B map: B pixels are 0, others 1.
    dist_to_b = cv2.distanceTransform((1 - b_bin).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_a = cv2.distanceTransform((1 - a_bin).astype(np.uint8), cv2.DIST_L2, 3)

    d_ab = float(dist_to_b[a_bin > 0].mean())
    d_ba = float(dist_to_a[b_bin > 0].mean())
    return 0.5 * (d_ab + d_ba)


def _tensor_to_gray_u8(x: torch.Tensor) -> np.ndarray:
    """Convert [1,H,W] / [H,W] tensor in [0,1] to uint8 gray image."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor/map, got shape={x.shape}")
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _gray_to_bgr(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected [H,W] gray image, got shape={x.shape}")
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)


def _build_wireframe_panel(item: Dict[str, torch.Tensor], text_lines: List[str]) -> np.ndarray:
    """Build a 3x3 panel from tensors actually consumed by the model.

    Columns: current / global ref / local ref
    Rows: line / edge / mask
    """
    c_line = _tensor_to_gray_u8(item["c_line"])
    g_line = _tensor_to_gray_u8(item["g_line"])
    l_line = _tensor_to_gray_u8(item["l_line"])

    c_edge = _tensor_to_gray_u8(item["c_edge"])
    g_edge = _tensor_to_gray_u8(item["g_edge"])
    l_edge = _tensor_to_gray_u8(item["l_edge"])

    c_mask = _tensor_to_gray_u8(item["c_mask"])
    l_mask = _tensor_to_gray_u8(item["l_mask"])
    g_mask = np.zeros_like(c_mask)  # global branch does not take global mask

    row1 = cv2.hconcat([_gray_to_bgr(c_line), _gray_to_bgr(g_line), _gray_to_bgr(l_line)])
    row2 = cv2.hconcat([_gray_to_bgr(c_edge), _gray_to_bgr(g_edge), _gray_to_bgr(l_edge)])
    row3 = cv2.hconcat([_gray_to_bgr(c_mask), _gray_to_bgr(g_mask), _gray_to_bgr(l_mask)])
    vis = cv2.vconcat([row1, row2, row3])

    y = 24
    for t in text_lines:
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        y += 24
    cv2.putText(vis, "Cols: Current | GlobalRef | LocalRef", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
    y += 24
    cv2.putText(vis, "Rows: Line | Edge | Mask", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
    return vis


def _save_model_input_npz(item: Dict[str, torch.Tensor], out_path: str) -> None:
    """Save concatenated geometry inputs used by model to local npz."""
    cat = torch.cat(
        [
            item["c_edge"], item["c_line"], item["c_mask"],
            item["g_edge"], item["g_line"],
            item["l_edge"], item["l_line"], item["l_mask"],
        ],
        dim=0,
    )
    np.savez_compressed(
        out_path,
        model_input_cat=cat.detach().cpu().float().numpy(),
        c_edge=item["c_edge"].detach().cpu().float().numpy(),
        c_line=item["c_line"].detach().cpu().float().numpy(),
        c_mask=item["c_mask"].detach().cpu().float().numpy(),
        g_edge=item["g_edge"].detach().cpu().float().numpy(),
        g_line=item["g_line"].detach().cpu().float().numpy(),
        l_edge=item["l_edge"].detach().cpu().float().numpy(),
        l_line=item["l_line"].detach().cpu().float().numpy(),
        l_mask=item["l_mask"].detach().cpu().float().numpy(),
    )


def run_check_ref_frame(opts, train_wrapper, logger) -> Dict[str, float]:
    """Validate global/local ref frame geometry against current GT by Chamfer Distance.
    Output summary CSV + sampled model-input visualizations to opts.check_ref_frame_dir.
    """
    out_dir = getattr(opts, "check_ref_frame_dir", "check_ref_frame")
    step = int(getattr(opts, "check_ref_frame_step", 1000))
    random_samples = int(getattr(opts, "check_ref_frame_random_samples", 20))
    seed = int(getattr(opts, "seed", 42))
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(seed)
    total_n = len(train_wrapper)
    periodic = list(range(0, total_n, max(step, 1)))
    random_ids = rng.sample(list(range(total_n)), k=min(random_samples, total_n)) if total_n > 0 else []
    vis_ids = sorted(set(periodic + random_ids))

    rows: List[Tuple[int, int, str, float, float, float, float]] = []
    global_cd_all, local_cd_all = [], []

    logger.info(f"[CheckRef] Start checking {total_n} frames. Output: {out_dir}")
    for idx in range(total_n):
        item = train_wrapper[idx]
        info = train_wrapper.idx_info.get(idx, {})

        c_line = _tensor_to_bin_np(item["c_line"])
        g_line = _tensor_to_bin_np(item["g_line"])
        l_line = _tensor_to_bin_np(item["l_line"])
        c_edge = _tensor_to_bin_np(item["c_edge"])
        g_edge = _tensor_to_bin_np(item["g_edge"])
        l_edge = _tensor_to_bin_np(item["l_edge"])

        cd_g_line = _chamfer_distance_binary(c_line, g_line)
        cd_l_line = _chamfer_distance_binary(c_line, l_line)
        cd_g_edge = _chamfer_distance_binary(c_edge, g_edge)
        cd_l_edge = _chamfer_distance_binary(c_edge, l_edge)
        global_cd_all.append(cd_g_line)
        local_cd_all.append(cd_l_line)

        rows.append((
            idx,
            int(info.get("global_idx", -1)),
            str(info.get("prev_idx", -1)),
            cd_g_line,
            cd_l_line,
            cd_g_edge,
            cd_l_edge,
        ))

        if idx in vis_ids:
            g_idx = int(info.get("global_idx", idx))
            text = [
                f"idx={idx} global={g_idx} prev={info.get('prev_idx', -1)}",
                f"line CD: curr-global={cd_g_line:.4f}, curr-local={cd_l_line:.4f}",
                f"edge CD: curr-global={cd_g_edge:.4f}, curr-local={cd_l_edge:.4f}",
            ]
            vis = _build_wireframe_panel(item, text)
            cv2.imwrite(os.path.join(out_dir, f"ref_check_{idx:06d}.jpg"), vis)
            _save_model_input_npz(item, os.path.join(out_dir, f"model_input_{idx:06d}.npz"))

    csv_path = os.path.join(out_dir, "chamfer_report.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("idx,global_idx,prev_idx,cd_global_line,cd_local_line,cd_global_edge,cd_local_edge\n")
        for r in rows:
            f.write(
                f"{r[0]},{r[1]},{r[2]},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f},{r[6]:.6f}\n"
            )

    finite_g = [x for x in global_cd_all if np.isfinite(x)]
    finite_l = [x for x in local_cd_all if np.isfinite(x)]
    summary = {
        "global_line_cd_mean": float(np.mean(finite_g)) if finite_g else float("inf"),
        "local_line_cd_mean": float(np.mean(finite_l)) if finite_l else float("inf"),
        "frames": float(total_n),
        "visualizations": float(len(vis_ids)),
    }
    logger.info(
        "[CheckRef] Done. global_line_cd_mean=%.6f | local_line_cd_mean=%.6f | vis=%d",
        summary["global_line_cd_mean"],
        summary["local_line_cd_mean"],
        len(vis_ids),
    )
    logger.info(f"[CheckRef] CSV saved to: {csv_path}")
    return summary
