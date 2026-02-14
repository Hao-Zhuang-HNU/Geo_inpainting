import os
import argparse

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

def main(input_path, output_path):
    # 列出 input_path 下的一级子目录
    subdirs = [d for d in os.listdir(input_path)
               if os.path.isdir(os.path.join(input_path, d))]
    if len(subdirs) == 0:
        raise ValueError(f"在 {input_path} 下没有找到子目录！")

    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)

    # 输出文件
    out_file = os.path.join(output_path, "irregular_list.txt")

    total_count = 0
    with open(out_file, "w") as f:
        for sub in subdirs:
            sub_abs = os.path.abspath(os.path.join(input_path, sub))
            img_list = iter_images_recursive(sub_abs)
            for img_path in img_list:
                f.write(img_path + "\n")
            total_count += len(img_list)

    print(f"子目录总数: {len(subdirs)}")
    print(f"总图片数: {total_count} -> 写入: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect all images into irregular_list.txt.")
    parser.add_argument("--input_path", type=str, required=True, help="输入数据集根目录（其下包含若干子目录）")
    parser.add_argument("--output_path", type=str, required=True, help="输出txt文件的目录")
    args = parser.parse_args()

    main(args.input_path, args.output_path)
