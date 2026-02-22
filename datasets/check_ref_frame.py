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


def _read_rgb(image_path: str, image_size: int) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img


def _build_triptych(curr_img: np.ndarray, g_img: np.ndarray, l_img: np.ndarray, text_lines: List[str]) -> np.ndarray:
    vis = np.concatenate([curr_img, g_img, l_img], axis=1)
    y = 24
    for t in text_lines:
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        y += 24
    return vis


def run_check_ref_frame(opts, train_wrapper, logger) -> Dict[str, float]:
    """Validate global/local ref frame geometry against current GT by Chamfer Distance.
    Output summary CSV + sampled triptychs to opts.check_ref_frame_dir.
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
            c_path = train_wrapper.dataset.image_id_list[idx]
            g_idx = int(info.get("global_idx", idx))
            g_path = train_wrapper.dataset.image_id_list[g_idx]
            if bool(info.get("is_first", False)):
                l_path = c_path
            else:
                p_idx = int(info.get("prev_idx", idx))
                p_idx = p_idx if p_idx >= 0 else idx
                l_path = train_wrapper.dataset.image_id_list[p_idx]

            c_img = _read_rgb(c_path, train_wrapper.dataset.image_size)
            g_img = _read_rgb(g_path, train_wrapper.dataset.image_size)
            l_img = _read_rgb(l_path, train_wrapper.dataset.image_size)
            text = [
                f"idx={idx} global={g_idx} prev={info.get('prev_idx', -1)}",
                f"line CD: curr-global={cd_g_line:.4f}, curr-local={cd_l_line:.4f}",
                f"edge CD: curr-global={cd_g_edge:.4f}, curr-local={cd_l_edge:.4f}",
            ]
            vis = _build_triptych(c_img, g_img, l_img, text)
            cv2.imwrite(os.path.join(out_dir, f"ref_check_{idx:06d}.jpg"), vis)

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

