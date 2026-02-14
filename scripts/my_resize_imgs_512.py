import os
import cv2
import argparse
from glob import glob

def resize_and_crop_to_512(img):
    h, w = img.shape[:2]
    # 缩放比例：最短边变为512
    scale = 512.0 / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 中心裁剪为512×512
    start_x = (new_w - 512) // 2
    start_y = (new_h - 512) // 2
    cropped = resized[start_y:start_y + 512, start_x:start_x + 512]

    return cropped

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
        img_cropped = resize_and_crop_to_512(img)
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, img_cropped)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and crop images to 512x512")
    parser.add_argument('--input_path', required=True, help='Input folder with original images')
    parser.add_argument('--output_path', required=True, help='Output folder for processed images')
    args = parser.parse_args()
    main(args)
