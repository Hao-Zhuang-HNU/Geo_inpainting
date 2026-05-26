import os
import argparse
import random
import hashlib
from PIL import Image, ImageOps, ImageFile
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing

# 防止处理破损图片时报错中断
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 支持的格式
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}


def choose_resample(file_path: str, mode: str):
    """
    根据参数选择插值方式。
    - rgb / lanczos: 适合普通 RGB 图像
    - bilinear: 适合普通 RGB 图像，速度略快
    - nearest: 适合 mask / line / label，避免灰度边缘
    - auto: 根据路径名自动判断 mask/line/label，否则用 LANCZOS
    """
    mode = mode.lower()
    if mode in ("rgb", "lanczos"):
        return Image.Resampling.LANCZOS
    if mode == "bilinear":
        return Image.Resampling.BILINEAR
    if mode == "bicubic":
        return Image.Resampling.BICUBIC
    if mode == "nearest":
        return Image.Resampling.NEAREST

    # auto
    path_lower = file_path.lower()
    keywords = ("mask", "masks", "line", "lines", "wireframe", "edge", "label", "seg")
    if any(k in path_lower for k in keywords):
        return Image.Resampling.NEAREST
    return Image.Resampling.LANCZOS


def resize_short_side(img: Image.Image, short_side: int, resample):
    """
    保持宽高比，将最短边缩放到 short_side。
    """
    w, h = img.size
    if min(w, h) == short_side:
        return img

    scale = short_side / float(min(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # 避免四舍五入导致短边小于 short_side
    if min(new_w, new_h) < short_side:
        if new_w <= new_h:
            new_w = short_side
            new_h = int(round(h * (short_side / float(w))))
        else:
            new_h = short_side
            new_w = int(round(w * (short_side / float(h))))

    return img.resize((new_w, new_h), resample=resample)


def get_crop_box(img: Image.Image, target_size: int, crop_mode: str, file_path: str, seed: int):
    """
    获取裁剪框。
    center: 中心裁剪，适合验证/测试
    random: 随机裁剪，适合训练；使用 file_path + seed 生成确定性随机数，保证可复现
    """
    w, h = img.size
    if w < target_size or h < target_size:
        raise ValueError(f"resize 后图像尺寸 {w}x{h} 小于裁剪尺寸 {target_size}x{target_size}")

    if crop_mode == "center":
        left = (w - target_size) // 2
        top = (h - target_size) // 2
    elif crop_mode == "random":
        # 多进程下不要直接用全局 random；用路径哈希保证每张图的随机裁剪可复现
        key = f"{file_path}|{seed}".encode("utf-8")
        digest = hashlib.md5(key).hexdigest()
        rng = random.Random(int(digest[:8], 16))
        left = rng.randint(0, w - target_size)
        top = rng.randint(0, h - target_size)
    else:
        raise ValueError(f"不支持的 crop_mode: {crop_mode}")

    return (left, top, left + target_size, top + target_size)


def process_single_image(args):
    """
    单张图片处理：
    1) 保持比例，将最短边缩放到 short_side
    2) 再裁剪为 target_size x target_size
    """
    file_path, target_size, short_side, crop_mode, resample_mode, seed = args

    try:
        if short_side < target_size:
            raise ValueError(f"--short_side({short_side}) 不能小于 --pix({target_size})")

        with Image.open(file_path) as img:
            # 处理 EXIF 旋转
            img = ImageOps.exif_transpose(img)

            # 已经是目标大小时跳过。注意：如果你想重新随机裁剪，请先保留原图备份。
            if img.size == (target_size, target_size):
                return "SKIPPED", file_path

            # 处理特殊模式
            # 对普通 RGB 图像转 RGB；对 mask/line 使用 nearest 时，尽量保留单通道/调色板模式
            resample = choose_resample(file_path, resample_mode)
            if resample != Image.Resampling.NEAREST and img.mode in ("RGBA", "P", "CMYK"):
                img = img.convert("RGB")
            elif img.mode == "CMYK":
                img = img.convert("RGB")

            # 短边缩放
            img = resize_short_side(img, short_side, resample=resample)

            # 裁剪
            crop_box = get_crop_box(img, target_size, crop_mode, file_path, seed)
            new_img = img.crop(crop_box)

        # 覆盖保存
        save_kwargs = {}
        if file_path.lower().endswith((".jpg", ".jpeg")):
            save_kwargs = {"quality": 95, "subsampling": 0}

        new_img.save(file_path, **save_kwargs)
        return "SUCCESS", file_path

    except Exception as e:
        return "ERROR", (file_path, str(e))


def get_image_files(input_path):
    """快速扫描所有图片路径"""
    image_files = []
    print(f"正在扫描目录: {input_path} ...")
    for root, _, files in os.walk(input_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, file))
    return image_files


def main():
    parser = argparse.ArgumentParser(
        description="保持比例的短边缩放 + 裁剪脚本。默认覆盖原图，请提前备份。"
    )
    parser.add_argument("-i", "--input_path", type=str, required=True, help="待处理图片根目录")
    parser.add_argument("--pix", type=int, default=256, help="最终裁剪尺寸，输出为 pix x pix")
    parser.add_argument(
        "--short_side",
        type=int,
        default=None,
        help="短边缩放到该尺寸后再裁剪；默认等于 --pix。训练可设为 288/320，测试可设为 256。"
    )
    parser.add_argument(
        "--crop_mode",
        type=str,
        default="center",
        choices=["center", "random"],
        help="裁剪方式：center 适合验证/测试；random 适合训练。"
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="auto",
        choices=["auto", "rgb", "lanczos", "bilinear", "bicubic", "nearest"],
        help="插值方式。auto 会对路径含 mask/line/label 等关键词的图片使用 nearest，否则用 lanczos。"
    )
    parser.add_argument("--seed", type=int, default=2026, help="random crop 的确定性随机种子")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="进程数")
    args = parser.parse_args()

    short_side = args.short_side if args.short_side is not None else args.pix
    if short_side < args.pix:
        raise ValueError(f"--short_side({short_side}) 不能小于 --pix({args.pix})")

    files = get_image_files(args.input_path)
    total_files = len(files)

    if total_files == 0:
        print("未找到图片文件。")
        return

    print(f"找到 {total_files} 张图片。")
    print(f"处理方式: short_side={short_side}, crop={args.pix}x{args.pix}, crop_mode={args.crop_mode}, resample={args.resample}")
    print(f"准备使用 {args.workers} 个核心并行处理...")

    tasks = [
        (f, args.pix, short_side, args.crop_mode, args.resample, args.seed)
        for f in files
    ]

    success_count = 0
    skipped_count = 0
    error_list = []

    chunk_size = max(1, total_files // max(1, args.workers * 10))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(process_single_image, tasks, chunksize=chunk_size),
            total=total_files,
            unit="img"
        ))

    for status, data in results:
        if status == "SUCCESS":
            success_count += 1
        elif status == "SKIPPED":
            skipped_count += 1
        elif status == "ERROR":
            error_list.append(data)

    print("\n" + "=" * 30)
    print("短边缩放 + 裁剪处理完成！")
    print(f"成功: {success_count}")
    print(f"跳过: {skipped_count}")
    print(f"失败: {len(error_list)}")
    print(f"最终分辨率: {args.pix}x{args.pix}")
    print(f"短边缩放尺寸: {short_side}")
    print(f"裁剪模式: {args.crop_mode}")
    print("=" * 30)

    if error_list:
        log_path = "resize_crop_failed.log"
        with open(log_path, "w", encoding="utf-8") as f:
            for path, reason in error_list:
                f.write(f"{path} : {reason}\n")
        print(f"详细错误日志已保存至 {log_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
