#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 COLMAP 深度 + COLMAP sparse 相机模型，将选中帧 sel_frame 的线图（pkl: data['lines']）
重投影到左右相邻两帧，并输出 5 张叠加线段的 RGB 图：

1) sel_frame 的线段 -> 左邻帧（绿色半透明线）
2) sel_frame 的线段 -> 右邻帧（绿色半透明线）
3) 左邻帧 GT 线段 -> 左邻帧（红色线）
4) sel_frame GT 线段 -> sel_frame（红色线）
5) 右邻帧 GT 线段 -> 右邻帧（红色线）

假设：
- COLMAP 工作目录结构：
    input_colmap_ws/
      images_256/  (frame_00XXX.png)
      depth/       (frame_00XXX.png.photometric.bin)
      dense/sparse      (cameras.bin, images.bin, points3D.bin)
- pkl 格式：data['lines']，shape = (N,4)，[x1,y1,x2,y2]
- 绘制时需要交换 (x,y) -> (y,x)，与 visualize_lines_pkl.py 一致
"""

import os
import argparse
import pickle
import numpy as np
from PIL import Image, ImageDraw


# =========================
# 工具：COLMAP TXT 模型读取
# =========================

def qvec2rotmat(qvec):
    """将 COLMAP 的四元数 [qw,qx,qy,qz] 转成 3x3 旋转矩阵（world->cam）。"""
    q0, q1, q2, q3 = qvec
    # 单位化
    norm = np.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    q0 /= norm; q1 /= norm; q2 /= norm; q3 /= norm

    R = np.array([
        [1 - 2*(q2*q2 + q3*q3),     2*(q1*q2 - q0*q3),         2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),         1 - 2*(q1*q1 + q3*q3),     2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),         2*(q2*q3 + q0*q1),         1 - 2*(q1*q1 + q2*q2)]
    ], dtype=np.float64)
    return R


def run_colmap_model_converter(sparse_dir, txt_dir):
    """如 txt_dir 下无 cameras.txt / images.txt，则调用 colmap model_converter 生成。"""
    cams_txt = os.path.join(txt_dir, "cameras.txt")
    imgs_txt = os.path.join(txt_dir, "images.txt")
    if os.path.isfile(cams_txt) and os.path.isfile(imgs_txt):
        print(f"[Info] Found existing COLMAP TXT model in {txt_dir}")
        return

    os.makedirs(txt_dir, exist_ok=True)
    cmd = f"colmap model_converter --input_path {sparse_dir} --output_path {txt_dir} --output_type TXT"
    print(f"[Info] Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"colmap model_converter failed with code {ret}")
    if not (os.path.isfile(cams_txt) and os.path.isfile(imgs_txt)):
        raise RuntimeError(f"cameras.txt or images.txt not found after model_converter in {txt_dir}")
    print(f"[Info] COLMAP TXT model generated in {txt_dir}")


def load_colmap_model(colmap_ws, target_size=256):
    """
    从 COLMAP sparse 模型中读取相机内外参，返回：
      name_to_info: {frame_name: {"index": idx, "Twc": Twc, "K": K}}
      order:        [frame_name0, frame_name1, ...]  按 IMAGE_ID 排序
    其中：
      - Twc: 4x4，相机坐标 -> 世界坐标
      - K:   3x3，缩放到 target_size×target_size 的内参
    """
    sparse_dir = os.path.join(colmap_ws, "dense/sparse")
    txt_dir = os.path.join(colmap_ws, "sparse_txt")

    if not os.path.isdir(sparse_dir):
        raise FileNotFoundError(f"sparse dir not found: {sparse_dir}")

    run_colmap_model_converter(sparse_dir, txt_dir)

    cams_txt = os.path.join(txt_dir, "cameras.txt")
    imgs_txt = os.path.join(txt_dir, "images.txt")

    # 1) 解析 cameras.txt
    camera_params = {}
    with open(cams_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            elems = line.split()
            cam_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = list(map(float, elems[4:]))

            # 默认 fx, fy, cx, cy 的解析
            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                f = params[0]
                cx, cy = params[1], params[2]
                fx, fy = f, f
            elif model == "OPENCV":
                fx, fy, cx, cy = params[:4]
            else:
                # 其他模型简单兜底：取前四个
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                print(f"[Warn] camera model {model} not explicitly handled, using first 4 params as fx,fy,cx,cy.")

            sx = float(target_size) / float(width)
            sy = float(target_size) / float(height)
            fx_s = fx * sx
            fy_s = fy * sy
            cx_s = cx * sx
            cy_s = cy * sy

            K = np.array([
                [fx_s, 0.0,  cx_s],
                [0.0,  fy_s, cy_s],
                [0.0,  0.0,  1.0]
            ], dtype=np.float64)

            camera_params[cam_id] = {
                "K": K,
                "w": width,
                "h": height,
                "model": model
            }

    # 2) 解析 images.txt
    name_to_info = {}
    order = []

    with open(imgs_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            elems = line.split()

            # 尝试把第一个 token 当 IMAGE_ID（int）解析；
            # 若失败，则说明这是第二行的 2D 点（以浮点数开头），直接跳过
            try:
                image_id = int(elems[0])
            except ValueError:
                # 这是 images.txt 中的第二行 (x y POINT3D_ID ...)，忽略
                continue

            if len(elems) < 10:
                # 保险起见：万一这一行格式不完整，也跳过
                continue

            qw, qx, qy, qz = map(float, elems[1:5])
            tx, ty, tz = map(float, elems[5:8])
            cam_id = int(elems[8])
            name = elems[9]

            # world->cam
            qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
            tvec = np.array([tx, ty, tz], dtype=np.float64)
            R = qvec2rotmat(qvec)  # world->cam

            Tcw = np.eye(4, dtype=np.float64)
            Tcw[:3, :3] = R
            Tcw[:3, 3] = tvec

            Twc = np.linalg.inv(Tcw)  # cam->world

            if cam_id not in camera_params:
                raise KeyError(f"Camera ID {cam_id} in images.txt not found in cameras.txt")

            K = camera_params[cam_id]["K"]

            # 取 basename 作为 frame_name，例如 "frame_00001.png"
            frame_name = os.path.splitext(os.path.basename(name))[0]

            name_to_info[frame_name] = {
                "index": image_id,
                "Twc": Twc,
                "K": K,
            }
            order.append((image_id, frame_name))

    # 按 IMAGE_ID 排序
    order = [fn for _, fn in sorted(order, key=lambda x: x[0])]
    print(f"[Info] Loaded COLMAP model for {len(order)} frames from {txt_dir}")
    return name_to_info, order



# =========================
# 深度读取
# =========================

def read_colmap_depth_bin(path):
    """
    读取 COLMAP 的 photometric / geometric 深度文件。

    文件格式（官方）：
      文本头:  b"width&height&channels&"
      后面紧跟: float32 深度数据，按 row-major 存储。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Depth file not found: {path}")

    with open(path, "rb") as f:
        # 读取文本头，直到第 3 个 '&'
        header_bytes = b""
        amp_count = 0
        while True:
            ch = f.read(1)
            if ch == b"":
                raise RuntimeError(f"Unexpected EOF while reading header of depth file: {path}")
            header_bytes += ch
            if ch == b"&":
                amp_count += 1
                if amp_count >= 3:
                    break

        parts = header_bytes.split(b"&")
        if len(parts) < 4:
            raise RuntimeError(f"Invalid depth header in {path}: {header_bytes}")

        width = int(parts[0])
        height = int(parts[1])
        channels = int(parts[2])

        if channels != 1:
            print(f"[Warn] depth map channels={channels}, not 1. Will still try to read it.")

        print(f"[Info] depth header: w={width}, h={height}, c={channels}")

        depth = np.fromfile(f, dtype=np.float32, count=width * height * channels)

    if depth.size != width * height * channels:
        raise RuntimeError(
            f"Depth data size mismatch in {path}: expected={width*height*channels}, got={depth.size}"
        )

    if channels == 1:
        depth = depth.reshape((height, width))
    else:
        depth = depth.reshape((height, width, channels))

    print(f"[Info] depth map loaded: {os.path.basename(path)} with size {width}x{height}, channels={channels}")
    return depth


def sample_depth_nearest(depth, pts_xy_256, target_size, eps=1e-8):
    """
    最近邻采样深度图:
      depth: (H_d, W_d)
      pts_xy_256: (N,2)，在 0~target_size-1 范围
    返回 (N,) 深度数组，非法点设为 0。
    """
    H_d, W_d = depth.shape[:2]
    scale_x = float(W_d) / float(target_size)
    scale_y = float(H_d) / float(target_size)

    pts = np.asarray(pts_xy_256, dtype=np.float64)
    x_256 = pts[:, 0]
    y_256 = pts[:, 1]

    u = np.rint(x_256 * scale_x).astype(np.int64)
    v = np.rint(y_256 * scale_y).astype(np.int64)

    u = np.clip(u, 0, W_d - 1)
    v = np.clip(v, 0, H_d - 1)

    d = depth[v, u]
    d[np.isnan(d)] = 0.0
    d[d < eps] = 0.0
    return d


def backproject_to_3d(pts_xy_256, depth_values, K):
    """2D + 深度 -> 相机坐标系 3D 点。"""
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    pts = np.asarray(pts_xy_256, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    z = depth_values

    valid = z > 0
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64), valid

    x = x[valid]
    y = y[valid]
    z = z[valid]

    x_n = (x - cx) / fx
    y_n = (y - cy) / fy

    Xc = x_n * z
    Yc = y_n * z
    Zc = z

    P_cam = np.stack([Xc, Yc, Zc], axis=1)  # (M,3)
    return P_cam, valid


# =========================
# PKL: lines 读取 & 重采样
# =========================

def load_lines_from_pkl(pkl_path, target_size=256):
    """
    从 pkl 中读取 data['lines']，格式 [x1,y1,x2,y2]，
    并按 visualize_lines_pkl.py 逻辑做坐标交换和缩放：

      - 原始 lines[i] = [x1, y1, x2, y2]
      - 可视化时使用 plt.plot([y1,y2], [x1,x2])，
        即在图像坐标中，我们希望使用：
          u1 = y1, v1 = x1
          u2 = y2, v2 = x2

    然后根据原始 W,H 线性缩放到 target_size×target_size。
    """
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Line pkl not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if "lines" not in data:
        raise KeyError(f"{pkl_path} 中未找到 'lines' 字段，请检查文件格式。")

    lines = np.asarray(data["lines"], dtype=np.float64)
    if lines.ndim != 2 or lines.shape[1] != 4:
        raise ValueError(f"lines 应为 (N,4) 数组，当前 shape={lines.shape}")

    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    # Matplotlib 中是 plot([y1,y2], [x1,x2])，这里按这个交换
    u1 = y1
    v1 = x1
    u2 = y2
    v2 = x2

    xs = np.hstack([u1, u2])
    ys = np.hstack([v1, v2])
    W_raw = int(xs.max()) + 1
    H_raw = int(ys.max()) + 1

    print(f"[Info] {os.path.basename(pkl_path)} lines raw resolution (after swap): W={W_raw}, H={H_raw}")

    sx = float(target_size) / float(W_raw)
    sy = float(target_size) / float(H_raw)

    u1_256 = u1 * sx
    v1_256 = v1 * sy
    u2_256 = u2 * sx
    v2_256 = v2 * sy

    lines_256 = np.stack([u1_256, v1_256, u2_256, v2_256], axis=1)
    return lines_256  # (N,4), [x1,y1,x2,y2] in 256×256 坐标


def sample_points_on_line(line_xyxy, n_samples=16):
    """
    在一条线段上均匀采样 n_samples 个点。
    line_xyxy: [x1,y1,x2,y2] in 256×256 coords
    返回 pts: (n_samples,2)
    """
    x1, y1, x2, y2 = line_xyxy
    t = np.linspace(0.0, 1.0, n_samples)
    x = x1 + (x2 - x1) * t
    y = y1 + (y2 - y1) * t
    pts = np.stack([x, y], axis=1)
    return pts


# =========================
# 画线到 RGB 图
# =========================

def draw_lines_on_rgb(rgb_path, lines_xyxy, out_path, target_size=256,
                      color=(255, 0, 0, 200), line_width=2):
    """
    在 RGB 图上叠加若干线段（[x1,y1,x2,y2]）并保存。
    color: RGBA，默认红色半透明。
    """
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")

    img = Image.open(rgb_path).convert("RGB")
    orig_w, orig_h = img.size
    if (orig_w, orig_h) != (target_size, target_size):
        print(f"[Info] resize RGB {os.path.basename(rgb_path)}: {orig_w}x{orig_h} -> {target_size}x{target_size}")
        img = img.resize((target_size, target_size), Image.BILINEAR)

    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    lines_xyxy = np.asarray(lines_xyxy, dtype=np.float32)
    for x1, y1, x2, y2 in lines_xyxy:
        draw.line([(float(x1), float(y1)), (float(x2), float(y2))],
                  fill=color, width=line_width)

    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")
    img.save(out_path)
    print(f"[Save] {out_path}")


def draw_polyline_segments_on_rgb(rgb_path, segments, out_path, target_size=256,
                                  color=(0, 255, 0, 160), line_width=2):
    """
    在 RGB 图上叠加若干折线段：
      segments: list of (M_i,2) ndarray，每个是一条折线的点序列。
    color: RGBA，默认绿色半透明。
    """
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")

    img = Image.open(rgb_path).convert("RGB")
    orig_w, orig_h = img.size
    if (orig_w, orig_h) != (target_size, target_size):
        print(f"[Info] resize RGB {os.path.basename(rgb_path)}: {orig_w}x{orig_h} -> {target_size}x{target_size}")
        img = img.resize((target_size, target_size), Image.BILINEAR)

    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for pts in segments:
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape[0] < 2:
            continue
        xy_list = [(float(x), float(y)) for x, y in pts]
        draw.line(xy_list, fill=color, width=line_width)

    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")
    img.save(out_path)
    print(f"[Save] {out_path}")


# =========================
# 利用深度对单条线段做重投影
# =========================

def project_points_sel_to_tgt(pts_sel_256, depth_sel, info_sel, info_tgt, target_size=256):
    """
    将选中帧上的 2D 点集合，利用深度和相机外参与内参，投影到目标帧：
      - info_sel["K"], info_sel["Twc"]  (cam->world)
      - info_tgt["K"], info_tgt["Twc"]

    返回在目标帧上的 2D 点坐标 (M,2)
    """
    K_sel = info_sel["K"]
    Twc_sel = info_sel["Twc"]
    K_tgt = info_tgt["K"]
    Twc_tgt = info_tgt["Twc"]

    Tcw_tgt = np.linalg.inv(Twc_tgt)

    # 1) 深度采样 + 反投影
    depth_vals = sample_depth_nearest(depth_sel, pts_sel_256, target_size=target_size)
    P_cam_sel, valid_mask = backproject_to_3d(pts_sel_256, depth_vals, K_sel)

    if P_cam_sel.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    N_valid = P_cam_sel.shape[0]
    P_cam_sel_h = np.concatenate(
        [P_cam_sel, np.ones((N_valid, 1), dtype=np.float64)], axis=1
    ).T  # (4, N)

    # 2) sel cam -> world
    P_world = Twc_sel @ P_cam_sel_h

    # 3) world -> tgt cam
    P_cam_tgt_h = Tcw_tgt @ P_world
    Xc = P_cam_tgt_h[0, :]
    Yc = P_cam_tgt_h[1, :]
    Zc = P_cam_tgt_h[2, :]

    front = Zc > 1e-6
    if not np.any(front):
        return np.zeros((0, 2), dtype=np.float64)

    Xc = Xc[front]
    Yc = Yc[front]
    Zc = Zc[front]

    xyz = np.vstack([Xc, Yc, Zc])
    uv = K_tgt @ xyz
    u = uv[0, :] / uv[2, :]
    v = uv[1, :] / uv[2, :]

    pts_tgt = np.stack([u, v], axis=1)

    # 只保留在图像范围内的点
    in_x = (pts_tgt[:, 0] >= 0) & (pts_tgt[:, 0] < target_size)
    in_y = (pts_tgt[:, 1] >= 0) & (pts_tgt[:, 1] < target_size)
    in_img = in_x & in_y

    pts_tgt = pts_tgt[in_img]
    return pts_tgt


def project_line_to_frame_with_depth(line_xyxy_sel_256, depth_sel,
                                     info_sel, info_tgt,
                                     target_size=256,
                                     n_samples=16):
    """
    对一条选中帧上的线段（[x1,y1,x2,y2]），沿线均匀采样若干点，
    用深度和相机外参与内参投影到目标帧，返回目标帧中的折线点序列。
    """
    pts_on_line = sample_points_on_line(line_xyxy_sel_256, n_samples=n_samples)
    pts_tgt = project_points_sel_to_tgt(pts_on_line, depth_sel, info_sel, info_tgt, target_size=target_size)
    return pts_tgt  # (M,2), M 可能 < n_samples（有裁剪）


def project_line_to_self_with_depth(line_xyxy_sel_256, depth_sel, info_sel,
                                    target_size=256, n_samples=32):
    """
    自投影检查：用 sel_frame 的深度 + 外参 + 内参，把线段
    反投影到 3D 再投回 sel_frame 本身。
    若参数 / 坐标系完全自洽，投回来的折线应该和原线几乎重合。
    """
    pts_on_line = sample_points_on_line(line_xyxy_sel_256, n_samples=n_samples)
    # 这里 target 也用 info_sel，自投影
    pts_self = project_points_sel_to_tgt(pts_on_line, depth_sel,
                                         info_sel, info_sel,
                                         target_size=target_size)
    return pts_self


# =========================
# 主流程
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Reproject line drawings using COLMAP depth and sparse camera model (lines version, COLMAP workspace)."
    )
    parser.add_argument("--input_pkl_dir", required=True,
                        help="线图 .pkl 目录（frame_00XXX.pkl）")
    parser.add_argument("--input_colmap_ws", required=True,
                        help="COLMAP 工作目录（包含 images_256, depth, sparse 子目录）")
    parser.add_argument("--sel_frame", required=True,
                        help="选中帧名称（不带后缀，例如 frame_00002）")
    parser.add_argument("--target_size", type=int, default=256,
                        help="统一分辨率（默认 256）")

    args = parser.parse_args()

    # 输出目录：固定在工作目录下
    output_dir = os.path.join(args.input_colmap_ws, "reproj_lines")
    os.makedirs(output_dir, exist_ok=True)

    img_dir = os.path.join(args.input_colmap_ws, "images_256")
    depth_dir = os.path.join(args.input_colmap_ws, "depth")

    # 1. 加载相机内外参（来自 COLMAP sparse）
    name_to_info, order = load_colmap_model(args.input_colmap_ws, target_size=args.target_size)

    sel_frame = os.path.splitext(args.sel_frame)[0]
    if sel_frame not in name_to_info:
        raise KeyError(f"Selected frame '{sel_frame}' not found in COLMAP model.")

    # 按 order 找左右邻帧
    idx_sel = order.index(sel_frame)
    if idx_sel <= 0 or idx_sel >= len(order) - 1:
        raise ValueError(f"Selected frame '{sel_frame}' has no left/right neighbour in COLMAP order (idx={idx_sel}).")

    left_frame = order[idx_sel - 1]
    right_frame = order[idx_sel + 1]
    print(f"[Info] sel_frame={sel_frame}, left_frame={left_frame}, right_frame={right_frame}")

    # 2. 路径拼接
    pkl_sel = os.path.join(args.input_pkl_dir, sel_frame + ".pkl")
    pkl_left = os.path.join(args.input_pkl_dir, left_frame + ".pkl")
    pkl_right = os.path.join(args.input_pkl_dir, right_frame + ".pkl")

    rgb_sel = os.path.join(img_dir, sel_frame + ".png")
    rgb_left = os.path.join(img_dir, left_frame + ".png")
    rgb_right = os.path.join(img_dir, right_frame + ".png")

    depth_sel_path = os.path.join(depth_dir, sel_frame + ".png.photometric.bin")

    # 3. 读取三帧的线段（已经交换 x,y 并缩放到 target_size×target_size）
    lines_sel_256 = load_lines_from_pkl(pkl_sel, target_size=args.target_size)
    lines_left_256 = load_lines_from_pkl(pkl_left, target_size=args.target_size)
    lines_right_256 = load_lines_from_pkl(pkl_right, target_size=args.target_size)

    print(f"[Info] sel lines: {lines_sel_256.shape[0]}")
    print(f"[Info] left lines: {lines_left_256.shape[0]}")
    print(f"[Info] right lines: {lines_right_256.shape[0]}")

    # 4. 读取选中帧的深度图
    depth_sel = read_colmap_depth_bin(depth_sel_path)

    info_sel = name_to_info[sel_frame]
    info_left = name_to_info[left_frame]
    info_right = name_to_info[right_frame]

    # 5. 将 sel_frame 的每条线段投影到左右帧
    segments_left = []
    segments_right = []

    for i, line in enumerate(lines_sel_256):
        pts_left = project_line_to_frame_with_depth(
            line, depth_sel, info_sel, info_left,
            target_size=args.target_size, n_samples=16
        )
        if pts_left.shape[0] >= 2:
            segments_left.append(pts_left)

        pts_right = project_line_to_frame_with_depth(
            line, depth_sel, info_sel, info_right,
            target_size=args.target_size, n_samples=16
        )
        if pts_right.shape[0] >= 2:
            segments_right.append(pts_right)

    print(f"[Info] sel->left projected segments: {len(segments_left)}")
    print(f"[Info] sel->right projected segments: {len(segments_right)}")

    # 6. 输出 5 张图
    # (1) sel_frame -> left_frame
    out1 = os.path.join(output_dir, f"{sel_frame}_to_{left_frame}_proj.png")
    draw_polyline_segments_on_rgb(
        rgb_left, segments_left, out1,
        target_size=args.target_size,
        color=(0, 255, 0, 200), line_width=2
    )

    # (2) sel_frame -> right_frame
    out2 = os.path.join(output_dir, f"{sel_frame}_to_{right_frame}_proj.png")
    draw_polyline_segments_on_rgb(
        rgb_right, segments_right, out2,
        target_size=args.target_size,
        color=(0, 255, 0, 200), line_width=2
    )

    # (3) 左邻帧 GT 线段
    out3 = os.path.join(output_dir, f"{left_frame}_gt.png")
    draw_lines_on_rgb(
        rgb_left, lines_left_256, out3,
        target_size=args.target_size,
        color=(255, 0, 0, 220), line_width=2
    )

    # (4) sel_frame GT 线段
    out4 = os.path.join(output_dir, f"{sel_frame}_gt.png")
    draw_lines_on_rgb(
        rgb_sel, lines_sel_256, out4,
        target_size=args.target_size,
        color=(255, 0, 0, 220), line_width=2
    )

    # (5) 右邻帧 GT 线段
    out5 = os.path.join(output_dir, f"{right_frame}_gt.png")
    draw_lines_on_rgb(
        rgb_right, lines_right_256, out5,
        target_size=args.target_size,
        color=(255, 0, 0, 220), line_width=2
    )

    print("[Done] Generated 5 projected RGB images in:", output_dir)

    # =========================
    # 额外输出：自投影检查图
    # =========================
    self_segments = []
    for line in lines_sel_256:
        pts_self = project_line_to_self_with_depth(
            line, depth_sel, info_sel,
            target_size=args.target_size, n_samples=32
        )
        if pts_self.shape[0] >= 2:
            self_segments.append(pts_self)

    # 在同一张图上画：
    #   红色：原始 GT 线段
    #   绿色：深度 + 外参自投影回来的线
    out_self = os.path.join(output_dir, f"{sel_frame}_self_check.png")

    # 先画红线 GT 到临时图
    base_tmp = os.path.join(output_dir, f"{sel_frame}_gt_tmp.png")
    draw_lines_on_rgb(
        rgb_sel, lines_sel_256, base_tmp,
        target_size=args.target_size,
        color=(255, 0, 0, 220), line_width=2
    )

    # 再在这张图上叠加绿线
    draw_polyline_segments_on_rgb(
        base_tmp, self_segments, out_self,
        target_size=args.target_size,
        color=(0, 255, 0, 200), line_width=2
    )

    # 删掉中间临时图
    try:
        os.remove(base_tmp)
    except Exception:
        pass

    print("[Info] Self-check image saved:", out_self)


if __name__ == "__main__":
    import numpy as np  # 确保 numpy 已导入（上面用到了）
    main()
