import argparse
import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm

from diffusers import AnimateDiffPipeline, MotionAdapter, AutoencoderKL, ControlNetModel, EulerDiscreteScheduler
from transformers import CLIPVisionModelWithProjection

try:
    from diffusers import AnimateDiffControlNetPipeline
except ImportError:
    from diffusers import AnimateDiffPipeline as AnimateDiffControlNetPipeline

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="AnimateDiff Ablation Study - With/Without Wireframe")
    # 更新了 help 描述
    parser.add_argument('--img_list', type=str, required=True, help='Path to image list file OR directory containing images')
    parser.add_argument('--mask_list', type=str, required=True, help='Path to mask list file OR directory containing masks')
    parser.add_argument('--line_list', type=str, required=False, default=None, help='Path to line list file OR directory (Optional for ablation)')
    parser.add_argument('--output_dir', type=str, required=True)
    
    parser.add_argument('--prompt', type=str, default="high quality, realistic, interior design, photorealistic, cinematic lighting, ultra-detailed", help='Positive prompt')
    parser.add_argument('--negative_prompt', type=str, default="low quality, worst quality, sketches, artifacts, blue glow, white edge, blur, noisy, cartoon", help='Negative prompt')
    
    parser.add_argument('--context_length', type=int, default=16)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ip_scale', type=float, default=0.7) 
    return parser.parse_args()

# ================= 修改后的加载函数 =================
def get_file_list(path):
    """
    加载文件列表。
    - 如果 path 是目录：扫描目录下所有图片并排序返回。
    - 如果 path 是文件：按行读取（原逻辑）。
    - 如果 path 是 None：返回 None。
    """
    if path is None:
        return None
    
    if os.path.isdir(path):
        # 支持的图片扩展名
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        # 扫描并排序，确保帧顺序正确
        files = sorted([
            os.path.join(path, f) 
            for f in os.listdir(path) 
            if f.lower().endswith(valid_exts)
        ])
        if not files:
            print(f">>> WARNING: No images found in directory: {path}")
        return files
        
    elif os.path.isfile(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    
    else:
        raise ValueError(f"Path does not exist: {path}")
# ===================================================

def add_texture_grain(img_np, sigma=1.2):
    noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
    img_noisy = img_np.astype(np.float32) + noise
    return np.clip(img_noisy, 0, 255).astype(np.uint8)

def apply_pro_color_match(gen_np, ref_np, mask_np):
    target = gen_np.astype(np.float32)
    ref = ref_np.astype(np.float32)
    kernel = np.ones((11, 11), np.uint8)
    bg_mask = 1.0 - cv2.dilate((mask_np > 127).astype(np.uint8), kernel, iterations=1)
    bg_mask = bg_mask[..., None]
    bg_sum = np.sum(bg_mask)
    if bg_sum > 100:
        for i in range(3):
            ref_mean = np.sum(ref[..., i] * bg_mask[:,:,0]) / bg_sum
            tgt_mean = np.sum(target[..., i] * bg_mask[:,:,0]) / bg_sum
            target[..., i] += (ref_mean - tgt_mean)
    return np.clip(target, 0, 255).astype(np.uint8)

def main():
    args = parse_args()
    device = f"cuda:{args.gpu_id}"
    dtype = torch.float32 

    # ================= 1. 加载模型 =================
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=dtype)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_lineart", torch_dtype=dtype)
    base_model = "SG161222/Realistic_Vision_V5.1_noVAE" 

    pipe = AnimateDiffControlNetPipeline.from_pretrained(
        base_model, motion_adapter=adapter, controlnet=controlnet,
        vae=vae, image_encoder=image_encoder, torch_dtype=dtype, safety_checker=None
    ).to(device)
    
    pipe.enable_model_cpu_offload(gpu_id=args.gpu_id)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
    pipe.set_ip_adapter_scale(args.ip_scale)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

    # ================= 2. 数据准备 (使用新函数) =================
    img_paths = get_file_list(args.img_list)
    mask_paths = get_file_list(args.mask_list)
    line_paths = get_file_list(args.line_list)
    
    # 基本检查
    if len(img_paths) == 0:
        raise ValueError("Image list is empty!")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 模式判定
    if line_paths is None or len(line_paths) == 0:
        print(">>> WARNING: line_list NOT provided (or empty). Running ABLATION MODE (No structure guidance).")
        use_line_guidance = False
    else:
        print(">>> INFO: line_list provided. Running FULL MODE (Structure-guided).")
        use_line_guidance = True

    chunk_size = args.context_length
    total_images = len(img_paths)
    
    print(f"Total images to process: {total_images}")

    for start_idx in tqdm(range(0, total_images, chunk_size)):
        end_idx = min(start_idx + chunk_size, total_images)
        current_batch_len = end_idx - start_idx
        if current_batch_len < 2: break

        batch_ctrl, batch_masks, batch_orig, filenames = [], [], [], []
        ref_style_image = Image.open(img_paths[start_idx]).convert("RGB").resize((512, 512))

        for i in range(start_idx, end_idx):
            # 处理 Line / Control Image
            if use_line_guidance:
                # 确保 line_paths 索引不越界 (如果文件数量不一致)
                idx_line = i if i < len(line_paths) else i % len(line_paths)
                l_img = Image.open(line_paths[idx_line]).convert("RGB").resize((512, 512))
                if np.array(l_img).mean() < 127: l_img = ImageOps.invert(l_img)
                batch_ctrl.append(l_img)
            else:
                # 【新模式逻辑】生成全黑图作为占位符
                batch_ctrl.append(Image.new("RGB", (512, 512), (0, 0, 0)))
            
            # Mask & Orig
            # 确保 mask_paths 索引不越界
            idx_mask = i if i < len(mask_paths) else i % len(mask_paths)
            batch_masks.append(Image.open(mask_paths[idx_mask]).convert("L").resize((512, 512)))
            
            batch_orig.append(Image.open(img_paths[i]).convert("RGB").resize((512, 512)))
            filenames.append(os.path.basename(img_paths[i]))

        # --- 生成 ---
        generator = torch.Generator("cpu").manual_seed(args.seed)
        
        # 【新模式逻辑】如果无线框，将权重设为 0
        current_ctrl_scale = 1.0 if use_line_guidance else 0.0
        
        output = pipe(
            prompt=args.prompt, negative_prompt=args.negative_prompt, num_frames=end_idx - start_idx,
            guidance_scale=7.5, num_inference_steps=25, generator=generator,
            ip_adapter_image=ref_style_image, 
            conditioning_frames=batch_ctrl, 
            controlnet_conditioning_scale=current_ctrl_scale,
        )
        frames = output.frames[0]

        # --- 后处理 ---
        for j, gen_img in enumerate(frames):
            orig_img_pil = batch_orig[j]
            mask_pil = batch_masks[j]
            gen_np = np.array(gen_img)
            orig_np = np.array(orig_img_pil)
            mask_np = np.array(mask_pil)

            gen_matched_np = apply_pro_color_match(gen_np, orig_np, mask_np)
            gen_textured_np = add_texture_grain(gen_matched_np, sigma=1.5)
            gen_textured_pil = Image.fromarray(gen_textured_np)
            
            # V8 收缩掩码截取
            kernel_erode = np.ones((23, 23), np.uint8)
            eroded_mask_np = cv2.erode(mask_np, kernel_erode, iterations=1)
            soft_mask_pil = Image.fromarray(eroded_mask_np).filter(ImageFilter.GaussianBlur(radius=10))
            
            blended = Image.composite(gen_textured_pil, orig_img_pil, soft_mask_pil)
            final_img = Image.composite(blended, orig_img_pil, ImageOps.invert(mask_pil))

            final_img.resize((256, 256)).save(os.path.join(args.output_dir, filenames[j]))

    print(f"Inpainting Done. Mode: {'Full' if use_line_guidance else 'Ablation (No Line)'}")

if __name__ == "__main__":
    main()