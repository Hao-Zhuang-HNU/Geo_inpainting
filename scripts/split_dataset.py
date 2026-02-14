import os
import argparse
import random

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def is_image(fname: str) -> bool:
    ext = os.path.splitext(fname)[1].lower()
    return ext in IMG_EXTS

def iter_images_recursive(root_dir: str):
    """递归遍历 root_dir 下的所有图片，返回绝对路径列表（排序保证稳定性）"""
    abs_paths = []
    for cur, _, files in os.walk(root_dir):
        for fn in files:
            if is_image(fn):
                abs_paths.append(os.path.abspath(os.path.join(cur, fn)))
    abs_paths.sort()
    return abs_paths

def main(input_path, output_path, train_ratio=0.95):
    # 列出 input_path 下的一级子目录
    subdirs = [d for d in os.listdir(input_path)
               if os.path.isdir(os.path.join(input_path, d))]
    if len(subdirs) == 0:
        raise ValueError(f"在 {input_path} 下没有找到子目录！")

    # 打乱并按比例划分子目录
    random.shuffle(subdirs)
    n_total = len(subdirs)
    n_train = int(n_total * train_ratio)
    train_subdirs = subdirs[:n_train]
    val_subdirs = subdirs[n_train:]

    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)

    # 输出文件
    train_file = os.path.join(output_path, "DL3DV_1K480P_train_list.txt")
    val_file   = os.path.join(output_path, "DL3DV_1K480P_val_list.txt")

    # 遍历“训练子目录”里的所有图片并写绝对路径
    train_count = 0
    with open(train_file, "w") as f:
        for sub in train_subdirs:
            sub_abs = os.path.abspath(os.path.join(input_path, sub))
            img_list = iter_images_recursive(sub_abs)
            for img_path in img_list:
                f.write(img_path + "\n")
            train_count += len(img_list)

    # 遍历“验证子目录”里的所有图片并写绝对路径
    val_count = 0
    with open(val_file, "w") as f:
        for sub in val_subdirs:
            sub_abs = os.path.abspath(os.path.join(input_path, sub))
            img_list = iter_images_recursive(sub_abs)
            for img_path in img_list:
                f.write(img_path + "\n")
            val_count += len(img_list)

    print(f"子目录总数: {n_total}  ->  训练: {len(train_subdirs)} / 验证: {len(val_subdirs)}")
    print(f"训练集图片数: {train_count}  -> 写入: {train_file}")
    print(f"验证集图片数: {val_count}  -> 写入: {val_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val image lists by subdirectories.")
    parser.add_argument("--input_path", type=str, required=True, help="输入数据集根目录（其下包含若干子目录）")
    parser.add_argument("--output_path", type=str, required=True, help="输出txt文件的目录")
    parser.add_argument("--train_ratio", type=float, default=0.95, help="训练集子目录比例，默认0.95")
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.train_ratio)
