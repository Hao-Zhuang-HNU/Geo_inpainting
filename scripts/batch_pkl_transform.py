#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pickle

def transform_line(line):
    """
    对单条线段坐标 [x1,y1,x2,y2] 做 90° 右旋 + 水平翻转。
    归一化坐标范围假设为 [0,1]。
    """
    x1, y1, x2, y2 = line
    # 先旋转： (x,y) -> (y, 1-x)
    # 再水平翻转： x -> 1-x
    x1_new = 1 - y1
    y1_new = 1 - x1
    x2_new = 1 - y2
    y2_new = 1 - x2
    return [x1_new, y1_new, x2_new, y2_new]

def process_file(in_path, out_path):
    # 读取 pkl
    with open(in_path, 'rb') as f:
        data = pickle.load(f)

    # 检查格式
    if not isinstance(data, dict) or 'lines' not in data or 'scores' not in data:
        raise ValueError(f"文件 {in_path} 格式不符合预期，应为包含 'lines' 和 'scores' 的 dict")

    lines = data['lines']
    scores = data['scores']
    if len(lines) != len(scores):
        raise ValueError(f"文件 {in_path} 中 'lines' 与 'scores' 长度不匹配")

    # 变换
    new_lines = [transform_line(line) for line in lines]

    # 保存结果
    new_data = {
        'lines': new_lines,
        'scores': scores
    }
    with open(out_path, 'wb') as f:
        pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    parser = argparse.ArgumentParser(description="批量处理 .pkl 文件：右旋 90° 并左右镜像")
    parser.add_argument('--input_path',  required=True,
                        help="输入目录，包含若干 .pkl 文件")
    parser.add_argument('--output_path', required=True,
                        help="输出目录，会在此生成同名 .pkl 文件")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    for fname in os.listdir(args.input_path):
        if not fname.lower().endswith('.pkl'):
            continue
        in_file  = os.path.join(args.input_path,  fname)
        out_file = os.path.join(args.output_path, fname)
        try:
            process_file(in_file, out_file)
            print(f"[OK] {fname}")
        except Exception as e:
            print(f"[ERROR] {fname}：{e}")

if __name__ == '__main__':
    main()
