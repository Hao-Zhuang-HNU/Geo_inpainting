import os
import cv2
import numpy as np
from glob import glob
import argparse

def generate_mask_from_image(img, threshold=240, expansion_ratio=1/20):
    """
    从图像中生成基于亮度阈值的掩码，并进行等宽轮廓膨胀。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)

    # 计算结构元素的大小（根据图片尺寸和比例）
    h, w = mask.shape
    kernel_size = max(1, int(min(h, w) * expansion_ratio))
    if kernel_size % 2 == 0:
        kernel_size += 1  # 确保为奇数，保证膨胀对称性

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)

    return expanded_mask

def process_folder(input_path, output_path, threshold=240, expansion_ratio=1/20):
    os.makedirs(output_path, exist_ok=True)

    img_paths = sorted(glob(os.path.join(input_path, "*.*")))
    supported_exts = [".jpg", ".jpeg", ".png", ".bmp"]

    for path in img_paths:
        ext = os.path.splitext(path)[-1].lower()
        if ext not in supported_exts:
            continue

        img = cv2.imread(path)
        if img is None:
            print(f"[Warning] Could not read {path}")
            continue

        mask = generate_mask_from_image(img, threshold, expansion_ratio)
        filename = os.path.basename(path)
        out_path = os.path.join(output_path, filename)
        cv2.imwrite(out_path, mask)
        print(f"[OK] Saved mask: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate overexposure masks with expansion")
    parser.add_argument("--input_path", required=True, help="Input folder containing images")
    parser.add_argument("--output_path", required=True, help="Folder to save generated masks")
    parser.add_argument("--threshold", type=int, default=240, help="Brightness threshold (default: 240)")
    parser.add_argument("--expand_ratio", type=float, default=1/20, help="Expansion ratio relative to image size (default: 1/20)")
    args = parser.parse_args()

    process_folder(args.input_path, args.output_path, args.threshold, args.expand_ratio)
