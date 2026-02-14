#!/usr/bin/env python3
# gen_spec_mask.py
import os
import cv2
import numpy as np
import argparse
from src.config import Config

def generate_specular_mask(img, v_thresh=0.95, s_thresh=0.2, closing_kernel=11, min_area=500):
    """
    生成高光区域掩码：
    - 转 HSV，V > v_thresh 且 S < s_thresh
    - 形态学闭运算去噪
    - 连通域过滤 (只保留面积 >= min_area 的区域)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # 归一化到 [0,1]
    v = v.astype(np.float32) / 255.0
    s = s.astype(np.float32) / 255.0
    mask = ((v > v_thresh) & (s < s_thresh)).astype(np.uint8) * 255
    # 闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 连通域过滤
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def main():
    parser = argparse.ArgumentParser(description="生成高光掩码")
    parser.add_argument(
        "--config", "-c", type=str,
        default="../config_list/config_mine_512.yml",
        help="配置文件路径，默认 ../config_list/config_mine_512.yml"
    )
    parser.add_argument("--v_thresh", type=float, default=0.95, help="V 通道阈值，范围 [0,1]")
    parser.add_argument("--s_thresh", type=float, default=0.2, help="S 通道阈值，范围 [0,1]")
    parser.add_argument("--kernel", type=int, default=11, help="闭运算核大小 (奇数)")
    parser.add_argument("--min_area", type=int, default=500, help="最小连通域面积")
    args = parser.parse_args()

    # 读取配置
    cfg = Config(args.config)
    # 输入 GT 图像文件夹
    gt_folder = cfg.GT_Val_FOLDER.strip("'\"")  # 配置中可能带引号 :contentReference[oaicite:5]{index=5}
    # 输出掩码列表文件
    mask_list_file = cfg.TRAIN_MASK_FLIST       # 掩码列表路径 :contentReference[oaicite:6]{index=6}
    mask_dir = os.path.dirname(mask_list_file)
    os.makedirs(mask_dir, exist_ok=True)

    mask_paths = []
    # 遍历所有图片
    for root, _, files in os.walk(gt_folder):
        for fname in sorted(files):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                img_path = os.path.join(root, fname)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[Warning] 无法读取 {img_path}，跳过。")
                    continue
                mask = generate_specular_mask(
                    img,
                    v_thresh=args.v_thresh,
                    s_thresh=args.s_thresh,
                    closing_kernel=args.kernel,
                    min_area=args.min_area
                )
                out_name = os.path.splitext(fname)[0] + "_mask.png"
                out_path = os.path.join(mask_dir, out_name)
                cv2.imwrite(out_path, mask)
                mask_paths.append(out_path)
                print(f"已生成掩码：{out_path}")

    # 写入掩码列表文件
    with open(mask_list_file, "w") as f:
        for p in mask_paths:
            f.write(p + "\n")
    print(f"\n完成！共生成 {len(mask_paths)} 个掩码，列表保存到 {mask_list_file}")

if __name__ == "__main__":
    main()
