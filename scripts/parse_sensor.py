#!/usr/bin/env python3
import yaml
import argparse

def parse_sensor(sensor_file):
    """
    读取 sensor.yml，提取分辨率、相机模型及内参 fx,fy,cx,cy
    返回一个 dict，供后续在 COLMAP 命令中使用
    """
    with open(sensor_file, 'r') as f:
        cfg = yaml.safe_load(f)

    # 从 resolution 字段取宽高
    width, height = cfg['resolution']
    # 相机模型（大写，以符合 COLMAP 要求，如 "PINHOLE" 或 "SIMPLE_RADIAL"）
    model = cfg['camera_model'].upper()
    # 内参顺序：[fx, fy, cx, cy]
    fx, fy, cx, cy = cfg['intrinsics']

    return {
        'width':  int(width),
        'height': int(height),
        'model':  model,
        'params': [float(fx), float(fy), float(cx), float(cy)]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse sensor.yml and print COLMAP camera parameters"
    )
    parser.add_argument(
        "--sensor", "-s",
        type=str,
        default="/root/auto-tmp/Inpainting/test_object/camera_yaml/sensor.yml",
        help="Path to sensor.yml"
    )
    args = parser.parse_args()

    cam = parse_sensor(args.sensor)
    # 输出到 stdout，方便在 shell 中 capture
    print(f"{cam['model']} {cam['params'][0]:.6f},{cam['params'][1]:.6f},"
          f"{cam['params'][2]:.6f},{cam['params'][3]:.6f}")
