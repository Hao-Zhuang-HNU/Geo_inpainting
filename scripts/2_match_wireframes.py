#!/usr/bin/env python3
# three_frame_match.py

import os
import argparse
import pickle
import numpy as np
import pydegensac
from tqdm import tqdm

def load_lines(pkl_file):
    """从 .pkl 中加载 lines 数组 (N,4)。"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return np.array(data['lines'])

def save_lines(lines, src_pkl, dst_pkl):
    """将 lines 保存到 dst_pkl，并保留其它字段。"""
    with open(src_pkl, 'rb') as f:
        data = pickle.load(f)
    data['lines'] = lines
    with open(dst_pkl, 'wb') as f:
        pickle.dump(data, f)

def compute_features(lines):
    centers = (lines[:, :2] + lines[:, 2:]) / 2.0
    diffs   = lines[:, 2:] - lines[:, :2]
    lengths = np.linalg.norm(diffs, axis=1, keepdims=True)
    dirs    = diffs / (lengths + 1e-8)
    return centers, dirs, lengths

def coarse_match(src, dst, dist_thresh, angle_thresh):
    sc, sd, _ = compute_features(src)
    dc, dd, _ = compute_features(dst)
    matches = []
    for i, (c1, d1) in enumerate(zip(sc, sd)):
        dists = np.linalg.norm(dc - c1, axis=1)
        cos_a = np.clip(dd.dot(d1), -1.0, 1.0)
        angs  = np.arccos(cos_a)
        idxs  = np.where((dists < dist_thresh)&(angs < angle_thresh))[0]
        for j in idxs:
            matches.append((i, int(j)))
    return matches

def fine_match(src, dst, coarse, ransac_thresh):
    if not coarse:
        return [], None
    src_pts = np.array([(src[i,:2]+src[i,2:])/2 for i,_ in coarse])
    dst_pts = np.array([(dst[j,:2]+dst[j,2:])/2 for _,j in coarse])
    # ---- add guard: 如果候选少于 4 对，就直接返回空内点，跳过 RANSAC ----
    if len(coarse) < 4:
        return [], None

    H, inliers = pydegensac.findHomography(
        src_pts, dst_pts,
        ransac_thresh,  # reprojection threshold
        0.99,           # confidence
        2000            # max iterations
    )
    inlier_pairs = [m for m,flag in zip(coarse, inliers) if flag]
    return inlier_pairs, H

def main():
    parser = argparse.ArgumentParser(
        description="三帧匹配：i-1↔i+1→i_temp，再 i_temp↔i→i_output"
    )
    parser.add_argument('--input_path',  type=str,
                        default='/root/autodl-tmp/Inpainting/test_object/wireframes/reshi_01_4')
    parser.add_argument('--output_path', type=str,
                        default='/root/autodl-tmp/Inpainting/test_object/match_wireframes/reshi_01_4')
    parser.add_argument('--dist_thresh',  type=float, default=30.0)
    parser.add_argument('--angle_thresh', type=float, default=np.pi/18)
    parser.add_argument('--ransac_thresh',type=float, default=5.0)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # 按文件名排序所有 pkl
    pkls = sorted([
        os.path.join(args.input_path, f)
        for f in os.listdir(args.input_path)
        if f.lower().endswith('.pkl')
    ])

    N = len(pkls)
    if N < 3:
        raise RuntimeError("至少需要 3 帧才能做三帧匹配")

    # 遍历每一帧
    for idx in tqdm(range(N), desc="三帧匹配"):
        src = pkls[idx]
        dst = os.path.join(args.output_path, os.path.basename(src))

        # 第一帧和最后一帧直接拷贝
        if idx == 0 or idx == N-1:
            with open(src, 'rb') as f_in, open(dst, 'wb') as f_out:
                f_out.write(f_in.read())
            continue

        # 加载三帧线段
        lines_prev = load_lines(pkls[idx-1])
        lines_next = load_lines(pkls[idx+1])
        lines_cur  = load_lines(pkls[idx])

        # 第一步：i-1 <-> i+1  匹配 → i_temp (选取 inliers 来自 next 方向的线)
        coarse1 = coarse_match(
            lines_prev, lines_next,
            args.dist_thresh, args.angle_thresh
        )
        inliers1, H1 = fine_match(lines_prev, lines_next,
                                  coarse1, args.ransac_thresh)
        # 如果第一步内点不足4对，直接输出原始，不再做第二步
        if len(inliers1) < 4:
            # 拷贝原始 .pkl 到输出
            shutil.copy(src, dst)
            continue
        idxs_next = [j for (_, j) in inliers1]
        lines_temp = lines_next[idxs_next]  # i_temp

        # 第二步：i_temp <-> i 匹配 → i_output
        coarse2 = coarse_match(
            lines_temp, lines_cur,
            args.dist_thresh, args.angle_thresh
        )
        inliers2, H2 = fine_match(lines_temp, lines_cur,
                                  coarse2, args.ransac_thresh)
        # 如果第二步内点不足4对，也退回到原始线段
        if len(inliers2) < 4:
            lines_out = lines_cur
        else:
            idxs_cur = [j for (_, j) in inliers2]
            lines_out = lines_cur[idxs_cur]

        # 保存 i_output 到 output_path
        save_lines(lines_out, src, dst)

    print("全部三帧匹配完成，输出保存在:", args.output_path)

if __name__ == '__main__':
    main()
