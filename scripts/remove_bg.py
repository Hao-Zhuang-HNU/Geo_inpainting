#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch background removal / mask generation.

Features:
  - Recursively find images under --input_path by default.
  - Preserve original subdirectory structure in --output_path.
  - Default backend: rembg + IS-Net if available; otherwise OpenCV GrabCut fallback.
  - With --save_mask: ONLY output black-background / white-foreground masks.
  - Final mask dilation: --mask_dilate, default 3 pixels.
  - Generate Mask Overlays: --save_overlay generates images showing the mask over the original background.

Dependencies:
  Basic fallback:
      pip install opencv-python numpy pillow

  Recommended high-quality backend:
      pip install rembg pillow onnxruntime

Example:
  python remove_bg_batch_mask_dilate.py \
      --input_path ./images \
      --output_path ./removed \
      --save_overlay
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

# Avoid CPU oversubscription: multiprocessing already parallelizes images.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_REMBG_SESSION = None
_REMBG_REMOVE = None


def list_images(input_path: Path, recursive: bool = True) -> List[Path]:
    """Find all supported images. Suffix matching is case-insensitive."""
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTS:
            return [input_path]
        raise ValueError(f"Input file is not a supported image: {input_path}")

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*" if recursive else "*"
    paths = [
        p for p in input_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(paths)


def imread_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix.lower() or ".png", image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def make_output_path(img_path: Path, input_root: Path, output_root: Path, suffix: str = ".png") -> Path:
    if input_root.is_dir():
        rel = img_path.relative_to(input_root)
    else:
        rel = Path(img_path.name)
    return output_root / rel.with_suffix(suffix)


def make_mask_path(img_path: Path, input_root: Path, mask_root: Path, mask_suffix: str) -> Path:
    if input_root.is_dir():
        rel = img_path.relative_to(input_root)
    else:
        rel = Path(img_path.name)
    mask_name = rel.with_suffix("").name + mask_suffix + ".png"
    return mask_root / rel.with_name(mask_name)


def make_overlay_path(img_path: Path, input_root: Path, overlay_root: Path, overlay_suffix: str) -> Path:
    if input_root.is_dir():
        rel = img_path.relative_to(input_root)
    else:
        rel = Path(img_path.name)
    overlay_name = rel.with_suffix("").name + overlay_suffix + ".jpg"
    return overlay_root / rel.with_name(overlay_name)


def create_overlay(orig_bgr: np.ndarray, mask: np.ndarray, color_bgr: Tuple[int, int, int], opacity: float) -> np.ndarray:
    """Blend a color over the pristine original BGR image based on the mask."""
    bgr = orig_bgr.astype(np.float32)
    color_layer = np.full_like(bgr, color_bgr)
    
    # Normalize mask to 0.0 - 1.0
    mask_norm = (mask / 255.0)[..., None]
    
    # Effective opacity considering soft/feathered mask edges
    eff_opacity = mask_norm * opacity
    
    # Alpha blending on the original background
    blended = bgr * (1.0 - eff_opacity) + color_layer * eff_opacity
    return blended.astype(np.uint8)


def resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend

    try:
        import rembg  # noqa: F401
        return "rembg"
    except Exception:
        return "grabcut"


def warmup_rembg(model_name: str) -> None:
    try:
        from rembg import new_session
        _ = new_session(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize rembg model '{model_name}'. "
            f"Install/check rembg and onnxruntime. Original error: {exc}"
        )


def get_rembg_session(model_name: str):
    global _REMBG_SESSION, _REMBG_REMOVE
    if _REMBG_SESSION is None or _REMBG_REMOVE is None:
        from rembg import new_session, remove
        _REMBG_SESSION = new_session(model_name)
        _REMBG_REMOVE = remove
    return _REMBG_SESSION, _REMBG_REMOVE


def dilate_binary_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    dilate_px = int(dilate_px)
    if dilate_px <= 0:
        return mask

    ksize = dilate_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask, kernel, iterations=1)


def apply_mask_to_bgra_alpha(bgra: np.ndarray, mask: np.ndarray, feather: int = 0) -> np.ndarray:
    if bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError("Expected BGRA image with 4 channels.")

    out = bgra.copy()
    alpha = mask.copy()

    if feather > 0:
        k = max(3, int(feather))
        if k % 2 == 0:
            k += 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    out[:, :, 3] = alpha
    return out


def remove_background_rembg(img_path: Path, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return: bgra output, mask, original bgr"""
    from PIL import Image

    session, remove = get_rembg_session(args.rembg_model)

    img = Image.open(str(img_path)).convert("RGBA")
    
    # Store the pristine original image before Rembg blanks out the background
    orig_rgba = np.array(img)
    orig_bgr = cv2.cvtColor(orig_rgba, cv2.COLOR_RGBA2BGR)

    out = remove(
        img,
        session=session,
        alpha_matting=args.alpha_matting,
        alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
        alpha_matting_background_threshold=args.alpha_matting_background_threshold,
        alpha_matting_erode_size=args.alpha_matting_erode_size,
        post_process_mask=args.post_process_mask,
    )

    if out.mode != "RGBA":
        out = out.convert("RGBA")

    rgba = np.array(out)
    alpha = rgba[:, :, 3]
    mask = np.where(alpha >= args.mask_threshold, 255, 0).astype(np.uint8)

    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    return bgra, mask, orig_bgr


def resize_keep_ratio(image: np.ndarray, max_size: int) -> Tuple[np.ndarray, float]:
    if max_size <= 0:
        return image, 1.0

    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_size:
        return image, 1.0

    scale = max_size / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def largest_connected_component(mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return mask

    h, w = mask.shape[:2]
    min_area = int(h * w * min_area_ratio)
    best_label = 0
    best_area = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area and area >= min_area:
            best_area = area
            best_label = label

    out = np.zeros_like(mask)
    if best_label != 0:
        out[labels == best_label] = 255
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def postprocess_mask(mask: np.ndarray, keep_largest: bool = True) -> np.ndarray:
    h, w = mask.shape[:2]
    k = max(3, int(round(min(h, w) * 0.008)))
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if keep_largest:
        mask = largest_connected_component(mask)

    return fill_holes(mask)


def color_distance_initial_mask(bgr: np.ndarray) -> Optional[np.ndarray]:
    h, w = bgr.shape[:2]
    if h < 20 or w < 20:
        return None

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    border = max(4, int(round(min(h, w) * 0.04)))
    border_pixels = np.concatenate([
        lab[:border, :, :].reshape(-1, 3),
        lab[-border:, :, :].reshape(-1, 3),
        lab[:, :border, :].reshape(-1, 3),
        lab[:, -border:, :].reshape(-1, 3),
    ], axis=0)

    bg_med = np.median(border_pixels, axis=0)
    dist = np.linalg.norm(lab - bg_med[None, None, :], axis=2)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    mask = postprocess_mask(mask, keep_largest=True)
    area_ratio = float(np.count_nonzero(mask)) / float(h * w)

    if area_ratio < 0.005 or area_ratio > 0.90:
        return None

    return mask


def rect_initial_mask(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    margin_x = max(2, int(w * 0.04))
    margin_y = max(2, int(h * 0.04))

    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[margin_y:h - margin_y, margin_x:w - margin_x] = cv2.GC_PR_FGD

    border = max(2, int(round(min(h, w) * 0.025)))
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    return mask


def grabcut_from_initial_mask(
    bgr: np.ndarray,
    init_fg_mask: Optional[np.ndarray],
    iterations: int,
    opencv_mode: str,
) -> np.ndarray:
    h, w = bgr.shape[:2]

    if opencv_mode == "rect" or init_fg_mask is None:
        gc_mask = rect_initial_mask(bgr)
    else:
        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[init_fg_mask > 0] = cv2.GC_PR_FGD

        k = max(3, int(round(min(h, w) * 0.01)))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        sure_fg = cv2.erode(init_fg_mask, kernel, iterations=1) > 0
        gc_mask[sure_fg] = cv2.GC_FGD

        border = max(3, int(round(min(h, w) * 0.025)))
        gc_mask[:border, :] = cv2.GC_BGD
        gc_mask[-border:, :] = cv2.GC_BGD
        gc_mask[:, :border] = cv2.GC_BGD
        gc_mask[:, -border:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            bgr,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            max(1, iterations),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        if init_fg_mask is not None:
            return init_fg_mask
        rect_mask = rect_initial_mask(bgr)
        return np.where(
            (rect_mask == cv2.GC_FGD) | (rect_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

    binary = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    return postprocess_mask(binary, keep_largest=True)


def remove_background_grabcut(image: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return: bgra output, mask, original bgr"""
    if image is None:
        raise ValueError("Input image is None.")

    if image.ndim == 2:
        bgr_full = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr_full = image[:, :, :3].copy()
    else:
        bgr_full = image[:, :, :3].copy()

    original_h, original_w = bgr_full.shape[:2]
    bgr_small, scale = resize_keep_ratio(bgr_full, args.max_size)

    init_mask = None
    if args.opencv_mode in {"auto", "color"}:
        init_mask = color_distance_initial_mask(bgr_small)

    mask_small = grabcut_from_initial_mask(
        bgr=bgr_small,
        init_fg_mask=init_mask,
        iterations=args.grabcut_iter,
        opencv_mode=args.opencv_mode,
    )

    if scale != 1.0:
        mask = cv2.resize(
            mask_small,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = postprocess_mask(mask, keep_largest=True)
    else:
        mask = mask_small

    bgra = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    return bgra, mask, bgr_full


def process_one_image(
    img_path: Path,
    input_root: Path,
    output_root: Path,
    mask_root: Optional[Path],
    overlay_root: Optional[Path],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    try:
        if args.resolved_backend == "rembg":
            # get the pristine original image (orig_bgr) directly before rembg alters it
            bgra, mask, orig_bgr = remove_background_rembg(img_path, args)
        elif args.resolved_backend == "grabcut":
            image = imread_unicode(img_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                return False, f"Failed to read: {img_path}"
            bgra, mask, orig_bgr = remove_background_grabcut(image, args)
        else:
            return False, f"Unknown backend: {args.resolved_backend}"

        # New parameter: mask dilation.
        mask = dilate_binary_mask(mask, args.mask_dilate)

        # Keep transparent-background output consistent with final mask.
        bgra = apply_mask_to_bgra_alpha(
            bgra,
            mask,
            feather=0 if args.save_mask else args.feather,
        )

        messages = []

        # Generate and save overlay using the pristine original background
        if args.save_overlay and overlay_root is not None:
            color_strs = args.overlay_color.split(',')
            color_bgr = (int(color_strs[0]), int(color_strs[1]), int(color_strs[2]))
            
            overlay_img = create_overlay(orig_bgr, mask, color_bgr, args.overlay_opacity)
            overlay_path = make_overlay_path(img_path, input_root, overlay_root, args.overlay_suffix)
            
            if imwrite_unicode(overlay_path, overlay_img):
                messages.append(f"overlay: {overlay_path.name}")
            else:
                return False, f"Failed to write overlay: {overlay_path}"

        if args.save_mask:
            if mask_root is None:
                raise ValueError("mask_root should not be None when save_mask is enabled.")
            mask_path = make_mask_path(img_path, input_root, mask_root, args.mask_suffix)
            if imwrite_unicode(mask_path, mask):
                messages.append(f"mask: {mask_path.name}")
            else:
                return False, f"Failed to write mask: {mask_path}"
        else:
            out_path = make_output_path(img_path, input_root, output_root, ".png")
            if imwrite_unicode(out_path, bgra):
                messages.append(f"out: {out_path.name}")
            else:
                return False, f"Failed to write out: {out_path}"

        return True, f"OK: {img_path.name} -> " + ", ".join(messages)

    except Exception as exc:
        return False, f"Failed: {img_path} | {type(exc).__name__}: {exc}"


def process_one_image_worker(
    img_path: Path,
    input_root: Path,
    output_root: Path,
    mask_root: Optional[Path],
    overlay_root: Optional[Path],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    cv2.setNumThreads(1)
    return process_one_image(img_path, input_root, output_root, mask_root, overlay_root, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch remove background from images, optionally export binary masks, and generate overlay previews."
    )

    parser.add_argument("--input_path", type=str, required=True, help="Input image directory or a single image path.")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory. Original subdirectory structure is preserved.")

    parser.add_argument(
        "--save_mask",
        action="store_true",
        help="If enabled, ONLY save binary masks and do NOT save transparent-background images.",
    )
    parser.add_argument("--mask_output_path", type=str, default=None, help="Mask output directory. Default: <output_path>/masks.")
    parser.add_argument("--mask_suffix", type=str, default="_mask", help="Suffix for mask files. Default: _mask.")
    parser.add_argument(
        "--mask_dilate",
        type=int,
        default=3,
        help="Dilate final foreground mask by N pixels. Default: 3. Use 0 to disable.",
    )
    
    # Overlay feature arguments
    parser.add_argument("--save_overlay", action="store_true", help="Generate an image showing the mask overlaid on the ORIGINAL image background.")
    parser.add_argument("--overlay_output_path", type=str, default=None, help="Overlay output directory. Default: <output_path>/overlays.")
    parser.add_argument("--overlay_suffix", type=str, default="_overlay", help="Suffix for overlay files. Default: _overlay.")
    parser.add_argument("--overlay_color", type=str, default="0,255,0", help="Overlay color in B,G,R format. Default: 0,255,0 (Green).")
    parser.add_argument("--overlay_opacity", type=float, default=0.5, help="Opacity of the overlay color (0.0 to 1.0). Default: 0.5.")

    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Only process images directly under --input_path. By default, subdirectories are scanned recursively.",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "rembg", "grabcut"],
        help="auto: use rembg if available, otherwise GrabCut. Default: auto.",
    )

    # rembg options
    parser.add_argument(
        "--rembg_model",
        type=str,
        default="isnet-general-use",
        help="rembg model name. Recommended: isnet-general-use. Default: isnet-general-use.",
    )
    parser.add_argument("--mask_threshold", type=int, default=128, help="Alpha threshold for binary mask when using rembg. Default: 128.")
    parser.add_argument("--alpha_matting", action="store_true", help="Enable rembg alpha matting. Better edges but slower.")
    parser.add_argument("--alpha_matting_foreground_threshold", type=int, default=240, help="rembg alpha matting foreground threshold.")
    parser.add_argument("--alpha_matting_background_threshold", type=int, default=10, help="rembg alpha matting background threshold.")
    parser.add_argument("--alpha_matting_erode_size", type=int, default=10, help="rembg alpha matting erode size.")
    parser.add_argument("--post_process_mask", action="store_true", help="Enable rembg post_process_mask.")

    # OpenCV fallback options
    parser.add_argument(
        "--opencv_mode",
        type=str,
        default="auto",
        choices=["auto", "color", "rect"],
        help="OpenCV GrabCut initialization mode. Default: auto.",
    )
    parser.add_argument("--max_size", type=int, default=1200, help="OpenCV fallback: resize long side for segmentation. 0 disables resizing.")
    parser.add_argument("--grabcut_iter", type=int, default=5, help="OpenCV fallback: GrabCut iterations.")
    parser.add_argument("--feather", type=int, default=3, help="Transparent PNG alpha feather. Does not affect saved binary mask.")

    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel worker processes. Default: 16.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(1)

    input_path = Path(args.input_path).expanduser().resolve()
    output_root = Path(args.output_path).expanduser().resolve()
    
    mask_root = (
        Path(args.mask_output_path).expanduser().resolve()
        if args.mask_output_path
        else output_root / "masks"
    )
    
    overlay_root = (
        Path(args.overlay_output_path).expanduser().resolve()
        if args.overlay_output_path
        else output_root / "overlays"
    )

    output_root.mkdir(parents=True, exist_ok=True)
    if args.save_mask:
        mask_root.mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        overlay_root.mkdir(parents=True, exist_ok=True)

    args.resolved_backend = resolve_backend(args.backend)

    if args.backend == "rembg" and args.resolved_backend != "rembg":
        raise RuntimeError("Backend was set to rembg, but rembg is not available.")

    print(f"Backend: {args.resolved_backend}")
    print(f"Mask dilation: {args.mask_dilate} px")

    if args.resolved_backend == "grabcut":
        print("Warning: using OpenCV GrabCut fallback. For better results, install: pip install rembg pillow onnxruntime")

    if args.resolved_backend == "rembg":
        print(f"rembg model: {args.rembg_model}")
        warmup_rembg(args.rembg_model)

    images = list_images(input_path, recursive=not args.no_recursive)
    if not images:
        print(f"No images found in: {input_path}")
        return

    print(f"Found {len(images)} image(s).")
    print(f"Recursive scan: {not args.no_recursive}")
    if input_path.is_dir():
        print("Original subdirectory structure will be preserved.")

    if args.save_mask:
        print("Mask-only mode enabled.")
        print(f"Output masks to: {mask_root}")
    else:
        print(f"Output transparent PNGs to: {output_root}")
        
    if args.save_overlay:
        print(f"Overlay mode enabled (Color BGR: {args.overlay_color}, Opacity: {args.overlay_opacity})")
        print(f"Output overlays to: {overlay_root}")

    ok_count = 0
    fail_count = 0

    num_workers = max(1, int(args.num_workers))
    num_workers = min(num_workers, len(images))

    worker = partial(
        process_one_image_worker,
        input_root=input_path,
        output_root=output_root,
        mask_root=mask_root if args.save_mask else None,
        overlay_root=overlay_root if args.save_overlay else None,
        args=args,
    )

    if num_workers == 1:
        for idx, (ok, msg) in enumerate(map(worker, images), start=1):
            ok_count += int(ok)
            fail_count += int(not ok)
            print(f"[{idx}/{len(images)}] {msg}")
    else:
        print(f"Using {num_workers} worker processes.")
        with Pool(processes=num_workers) as pool:
            for idx, (ok, msg) in enumerate(pool.imap_unordered(worker, images), start=1):
                ok_count += int(ok)
                fail_count += int(not ok)
                print(f"[{idx}/{len(images)}] {msg}")

    print("=" * 60)
    print(f"Done. Success: {ok_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()