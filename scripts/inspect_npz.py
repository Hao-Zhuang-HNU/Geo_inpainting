import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="查看单个 .npz 文件的内容")
    parser.add_argument('--input_npz', type=str, required=True, help='npz 文件的路径')
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.input_npz):
        print(f"错误: 文件不存在 -> {args.input_npz}")
        return

    try:
        # 加载 npz 文件
        data = np.load(args.input_npz, allow_pickle=True)
        
        print("\n" + "="*50)
        print(f"正在查看文件: {os.path.basename(args.input_npz)}")
        print("="*50)

        # 列出所有包含的 Key
        keys = data.files
        print(f"包含的 Keys: {keys}\n")

        # 遍历并打印内容
        for key in keys:
            val = data[key]
            
            print(f"[{key}]")
            
            if key == 'homography':
                # 格式化打印 3x3 矩阵，保留 6 位小数
                if val.shape == (3, 3):
                    for row in val:
                        print(f"  [ {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f} ]")
                    # 计算行列式，用于判断变换是否极端
                    det = np.linalg.det(val[:2, :2])
                    print(f"  (矩阵左上 2x2 行列式: {det:.4f})")
                else:
                    print(f"  {val}")
            
            elif key == 'confidence_iou':
                print(f"  数值: {float(val):.4f}")
            
            elif key == 'valid':
                # 处理 bool 或 int
                is_valid = bool(val)
                color = "\033[92mTrue\033[0m" if is_valid else "\033[91mFalse\033[0m"
                print(f"  有效性: {color}")
            
            else:
                # 其他可能存在的键 (如 match_points 等)
                if isinstance(val, np.ndarray) and val.size > 10:
                    print(f"  形状: {val.shape}, 类型: {val.dtype} (数据量较大，仅显示概要)")
                else:
                    print(f"  内容: {val}")
            print("-" * 30)

    except Exception as e:
        print(f"解析失败: {e}")

if __name__ == "__main__":
    main()