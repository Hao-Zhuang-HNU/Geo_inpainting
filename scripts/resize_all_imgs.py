import os
import argparse
from PIL import Image, ImageOps, ImageFile
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing

# 防止处理破损图片时报错中断
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 支持的格式
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}

def process_single_image(args):
    """
    单个图片处理函数：执行强制缩放（保持全图视野）
    """
    file_path, target_size = args
    
    try:
        # 1. 打开图片
        with Image.open(file_path) as img:
            # 2. 处理 EXIF 旋转
            img = ImageOps.exif_transpose(img)
            
            # 如果分辨率已经是目标大小，直接跳过
            if img.size == (target_size, target_size):
                return "SKIPPED", file_path
            
            # 3. 格式转换：处理特殊的图像模式
            if img.mode in ('RGBA', 'P', 'CMYK'):
                img = img.convert('RGB')
            
            # 【核心修改点】：使用 resize 而不是 fit
            # 这会将 1752x1168 压扁成 512x512，但保留了画面最边缘的线条。
            # 虽然视觉上看起来缩放了，但对于 LightGlue 匹配和 Warp 来说是完美的。
            new_img = img.resize(
                (target_size, target_size), 
                resample=Image.Resampling.LANCZOS
            )

        # 4. 覆盖保存
        save_kwargs = {}
        if file_path.lower().endswith(('.jpg', '.jpeg')):
            # 512 尺度下，质量 95 可以很好地保留线框特征
            save_kwargs = {'quality': 95, 'subsampling': 0}
            
        new_img.save(file_path, **save_kwargs)
        return "SUCCESS", file_path
            
    except Exception as e:
        return "ERROR", (file_path, str(e))

def get_image_files(input_path):
    """快速扫描所有文件路径"""
    image_files = []
    print(f"正在扫描目录: {input_path} ...")
    for root, _, files in os.walk(input_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, file))
    return image_files

def main():
    parser = argparse.ArgumentParser(description="ScanNet++ 全图缩放脚本 (不裁剪，保留视野)")
    parser.add_argument('--input_path', type=str, required=True, help="ScanNet++ 处理后的根目录")
    parser.add_argument('--pix', type=int, default=512, help="目标分辨率 (建议 512 用于提取 pkl 和生成 npz)")
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help="进程数")
    args = parser.parse_args()

    # 1. 获取所有图片路径
    files = get_image_files(args.input_path)
    total_files = len(files)
    
    if total_files == 0:
        print("未找到图片文件。")
        return

    print(f"找到 {total_files} 张图片。准备使用 {args.workers} 个核心进行并行缩放...")

    # 2. 准备参数列表
    tasks = [(f, args.pix) for f in files]

    # 3. 开启多进程池
    success_count = 0
    skipped_count = 0
    error_list = []

    # 针对 100w 级数据优化任务分发块大小
    chunk_size = max(1, total_files // (args.workers * 10))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(process_single_image, tasks, chunksize=chunk_size), 
            total=total_files, 
            unit="img"
        ))

    # 4. 统计结果
    for status, data in results:
        if status == "SUCCESS":
            success_count += 1
        elif status == "SKIPPED":
            skipped_count += 1
        elif status == "ERROR":
            error_list.append(data)

    # 5. 输出总结
    print(f"\n" + "="*30)
    print(f"缩放处理完成！")
    print(f"成功: {success_count}")
    print(f"跳过: {skipped_count}")
    print(f"失败: {len(error_list)}")
    print(f"最终分辨率: {args.pix}x{args.pix}")
    print(f"="*30)
    
    if error_list:
        with open("resize_failed.log", "w", encoding="utf-8") as f:
            for path, reason in error_list:
                f.write(f"{path} : {reason}\n")
        print(f"详细错误日志已保存至 resize_failed.log")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()