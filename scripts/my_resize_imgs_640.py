import os
import cv2
import argparse
from glob import glob
import numpy as np

def resize_and_pad_to_640(img):
    h, w = img.shape[:2]
    # 缩放比例：最长边不超过640
    scale = 640.0 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建黑色背景 640x640
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)

    # 将缩放后的图片居中放置
    start_x = (640 - new_w) // 2
    start_y = (640 - new_h) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    return canvas

def main(args):
    input_dir = args.input_path
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)

    img_paths = sorted(glob(os.path.join(input_dir, "*.*")))
    supported_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    for img_path in img_paths:
        ext = os.path.splitext(img_path)[-1].lower()
        if ext not in supported_exts:
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to load {img_path}")
            continue
        img_padded = resize_and_pad_to_640(img)
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, img_padded)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and pad images to 640x640 with black background")
    parser.add_argument('--input_path', required=True, help='Input folder with original images')
    parser.add_argument('--output_path', required=True, help='Output folder for processed images')
    args = parser.parse_args()
    main(args)
