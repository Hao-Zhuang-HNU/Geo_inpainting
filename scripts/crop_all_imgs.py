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
    单个图片处理函数
    返回格式: (状态字符串, 附加信息)
    状态: "SUCCESS", "SKIPPED", "ERROR"
    """
    file_path, target_size = args
    
    try:
        # 1. 打开图片
        # 使用 with 上下文管理器，确保读取后及时释放文件句柄（对 Windows 覆盖写入很重要）
        with Image.open(file_path) as img:
            # 2. 处理 EXIF 旋转 (手机拍摄的图片通常需要这就一步)
            img = ImageOps.exif_transpose(img)
            
            # 【新增需求】: 如果分辨率已经是目标大小，直接跳过
            if img.size == (target_size, target_size):
                return "SKIPPED", file_path
            
            # 3. 格式转换：如果是 JPG 且模式不是 RGB (如 CMYK/P)，转换一下防止报错
            if file_path.lower().endswith(('.jpg', '.jpeg')) and img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # 4. 智能中心裁剪
            # 生成新的图像对象 new_img
            new_img = ImageOps.fit(
                img, 
                (target_size, target_size), 
                method=Image.Resampling.LANCZOS, 
                centering=(0.5, 0.5)
            )

        # with 块结束，源文件句柄已释放
        
        # 5. 覆盖保存
        # 【新增需求】: 只有代码运行到这里（说明处理没报错），才会执行保存。
        # 这样如果上面出错，原图不会被覆盖。
        save_kwargs = {}
        if file_path.lower().endswith(('.jpg', '.jpeg')):
            save_kwargs = {'quality': 90, 'subsampling': 0}
            
        new_img.save(file_path, **save_kwargs)
        return "SUCCESS", file_path
            
    except Exception as e:
        # 【新增需求】: 返回错误路径和具体原因
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
    parser = argparse.ArgumentParser(description="多进程极速裁剪图片脚本 (带错误日志及自动删除)")
    parser.add_argument('-i', '--input_path', type=str, required=True, help="图片目录")
    parser.add_argument('--pix', type=int, default=256, help="目标分辨率")
    parser.add_argument("--workers", type=int, default=16, help="parallel workers")
    # 【新增参数】 --rm
    parser.add_argument('--rm', action='store_true', help="开启后自动删除处理失败的图片文件")
    
    args = parser.parse_args()

    # 1. 获取所有图片路径
    files = get_image_files(args.input_path)
    total_files = len(files)
    
    if total_files == 0:
        print("未找到图片文件。")
        return

    print(f"找到 {total_files} 张图片。准备使用 {args.workers} 个核心进行并行处理...")

    # 2. 准备参数列表
    tasks = [(f, args.pix) for f in files]

    # 3. 开启多进程池
    success_count = 0
    skipped_count = 0
    error_list = [] # 存储 (path, reason)

    chunk_size = max(1, total_files // (args.workers * 4))

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
            error_list.append(data) # data 是 (path, reason)

    # 5. 输出总结
    print(f"\n" + "="*30)
    print(f"处理完成！")
    print(f"成功处理: {success_count}")
    print(f"直接跳过 (尺寸已合规): {skipped_count}")
    print(f"处理失败: {len(error_list)}")
    print(f"="*30)
    
    # 6. 保存失败日志
    if error_list:
        log_file = "crop_failed.log"
        print(f"\n发现 {len(error_list)} 个错误，正在写入日志到 {log_file} ...")
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"目录: {args.input_path}\n")
                f.write(f"目标分辨率: {args.pix}\n")
                # 标记是否执行了删除
                if args.rm:
                    f.write("注意: 脚本开启了 --rm 参数，以下文件已被删除。\n")
                f.write("-" * 50 + "\n")
                for path, reason in error_list:
                    f.write(f"文件: {path}\n原因: {reason}\n\n")
            print(f"日志保存成功。")
        except Exception as e:
            print(f"日志保存失败: {e}")

        # 【新增逻辑】: 执行删除操作
        if args.rm:
            print(f"\n[注意] 检测到 --rm 参数，正在删除 {len(error_list)} 个失败文件...")
            deleted_count = 0
            for path, reason in error_list:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        deleted_count += 1
                except Exception as del_e:
                    print(f"删除失败: {path} -> {del_e}")
            
            print(f"清理完毕。共删除了 {deleted_count} 个损坏/无法处理的文件。")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()