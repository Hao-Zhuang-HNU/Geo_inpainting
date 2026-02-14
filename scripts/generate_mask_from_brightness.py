import os
import cv2
import numpy as np
from glob import glob
import argparse

def generate_mask_from_image(img, threshold=240):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    return mask

def process_folder(input_path, output_path, threshold=240):
    os.makedirs(output_path, exist_ok=True)

    # 支持的图片格式
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

        mask = generate_mask_from_image(img, threshold)
        filename = os.path.basename(path)
        out_path = os.path.join(output_path, filename)
        cv2.imwrite(out_path, mask)
        print(f"[OK] Saved mask: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate brightness-based masks")
    parser.add_argument("--input_path", required=True, help="Input folder containing images")
    parser.add_argument("--output_path", required=True, help="Folder to save generated masks")
    parser.add_argument("--threshold", type=int, default=240, help="Brightness threshold (default: 240)")
    args = parser.parse_args()

    process_folder(args.input_path, args.output_path, args.threshold)
