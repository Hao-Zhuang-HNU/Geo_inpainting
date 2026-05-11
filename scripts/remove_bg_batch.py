#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch background removal / mask generation.

Main improvement:
  - Default backend is "auto": use rembg + IS-Net if available, otherwise fallback to OpenCV GrabCut.
  - This is much more robust for hard cases such as metallic thin objects, textured floor,
    shadows, and foreground/background colors that are close.

Input:
  --input_path   image directory or a single image; directories are scanned recursively by default
  --output_path  output directory; original subdirectory structure is preserved

Behavior:
  - Without --save_mask:
      output transparent-background PNG images.
  - With --save_mask:
      ONLY output black-background / white-foreground masks.

Dependencies:
  Basic OpenCV fallback:
      pip install opencv-python numpy

  Recommended high-quality backend:
      pip install rembg pillow onnxruntime
  If you have CUDA onnxruntime:
      pip install rembg pillow onnxruntime-gpu

Examples:
  # High-quality transparent PNG output, recursively preserve directory structure
  python remove_bg_batch.py \
      --input_path ./images \
      --output_path ./removed

  # Only output masks
  python remove_bg_batch.py \
      --input_path ./images \
      --output_path ./removed \
      --save_mask

  # Force deep-learning backend
  python remove_bg_batch.py \
      --input_path ./images \
      --output_path ./removed \
      --backend rembg \
      --save_mask

  # Force OpenCV fallback
  python remove_bg_batch.py \
      --input_path ./images \
      --output_path ./removed \
      --backend grabcut \
      --save_mask
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

# Avoid CPU oversubscription before libraries start their own thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np


# Case-insensitive because paths are matched with suffix.lower(); supports .png, .jpg, .JPG, etc.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


_REMBG_SESSION = None
_REMBG_REMOVE = None


def list_images(input_path: Path, recursive: bool = True) -> List[Path]:
    """Return image paths from a directory or a single image path."""
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
    """Read image safely when path contains non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    """Write image safely when path contains non-ASCII characters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == "":
        suffix = ".png"

    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def rel_output_path(img_path: Path, input_root: Path, output_root: Path, suffix: str = ".png") -> Path:
    """Build output path while preserving original subdirectory structure."""
    if input_root.is_dir():
        rel = img_path.relative_to(input_root)
    else:
        rel = Path(img_path.name)
    return output_root / rel.with_suffix(suffix)


def rel_mask_path(
    img_path: Path,
    input_root: Path,
    mask_root: Path,
    mask_suffix: str = "_mask",
) -> Path:
    """Build mask path while preserving original subdirectory structure."""
    if input_root.is_dir():
        rel = img_path.relative_to(input_root)
    else:
        rel = Path(img_path.name)

    mask_name = rel.with_suffix("").name + mask_suffix + ".png"
    return mask_root / rel.with_name(mask_name)


def resolve_backend(backend: str) -> str:
    """
    Resolve backend.

    auto:
      use rembg if installed, otherwise fallback to grabcut.
    """
    if backend != "auto":
        return backend

    try:
        import rembg  # noqa: F401
        return "rembg"
    except Exception:
        return "grabcut"


def warmup_rembg_model(model_name: str) -> None:
    """
    Download/load rembg model once in the main process before spawning workers.
    This avoids multiple workers trying to download the same model at the same time.
    """
    try:
        from rembg import new_session
        _ = new_session(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize rembg model '{model_name}'. "
            f"Check whether rembg/onnxruntime are installed correctly. Original error: {exc}"
        )


def get_rembg_session(model_name: str):
    """Lazy per-process rembg session."""
    global _REMBG_SESSION, _REMBG_REMOVE

    if _REMBG_SESSION is None or _REMBG_REMOVE is None:
        from rembg import new_session, remove
        _REMBG_SESSION = new_session(model_name)
        _REMBG_REMOVE = remove

    return _REMBG_SESSION, _REMBG_REMOVE


def remove_background_rembg(
    img_path: Path,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deep-learning background removal using rembg.

    Return:
      rgba_bgra: BGRA image for OpenCV writing.
      mask:      uint8 binary mask, 0 background / 255 foreground.
    """
    from PIL import Image

    session, remove = get_rembg_session(args.rembg_model)

    img = Image.open(str(img_path)).convert("RGBA")

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

    rgba = np.array(out)  # RGBA
    alpha = rgba[:, :, 3]

    mask = np.where(alpha >= args.mask_threshold, 255, 0).astype(np.uint8)

    # Convert RGBA -> BGRA for cv2.imwrite/imencode.
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

    return bgra, mask


def resize_keep_ratio(image: np.ndarray, max_size: int) -> Tuple[np.ndarray, float]:
    """Resize image so that max(h, w) <= max_size. Return resized image and scale."""
    if max_size <= 0:
        return image, 1.0

    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_size:
        return image, 1.0

    scale = max_size / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def remove_border_connected_components(mask: np.ndarray) -> np.ndarray:
    """
    Remove foreground components touching image border.
    Useful for excluding table edges, wall edges, and frame borders.
    """
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    out = np.zeros_like(mask)
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])

        touches_border = (
            x <= 0 or y <= 0 or
            x + ww >= w or y + hh >= h
        )
        if not touches_border:
            out[labels == label] = 255

    return out


def largest_connected_component(mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    """
    Keep the largest foreground connected component.
    mask: uint8, values 0/255.
    """
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

    if best_label == 0:
        return mask

    out = np.zeros_like(mask)
    out[labels == best_label] = 255
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes inside foreground.
    mask: uint8, values 0/255.
    """
    h, w = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, holes)
    return filled


def postprocess_mask_opencv(mask: np.ndarray, keep_largest: bool = True) -> np.ndarray:
    """
    Smooth and clean a binary mask.
    mask: uint8, values 0/255.
    """
    h, w = mask.shape[:2]
    k = max(3, int(round(min(h, w) * 0.008)))
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if keep_largest:
        mask = largest_connected_component(mask)

    mask = fill_holes(mask)

    return mask


def color_distance_initial_mask(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Build an initial foreground mask by comparing every pixel with border background color.
    Works only when object differs clearly from the background.
    """
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

    # Force image border to background.
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    mask = remove_border_connected_components(mask)
    mask = postprocess_mask_opencv(mask, keep_largest=True)

    area_ratio = float(np.count_nonzero(mask)) / float(h * w)

    # Too small or too large means the color-distance estimate is unreliable.
    if area_ratio < 0.005 or area_ratio > 0.90:
        return None

    return mask


def rect_initial_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Rectangle initialization.
    Assumption: the main object is mostly inside the image center.
    """
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
    mode: str,
) -> np.ndarray:
    """
    Run GrabCut and return a binary foreground mask in uint8 0/255.
    mode:
      auto  - color-distance mask first; fallback to rectangle
      color - color-distance mask; fallback to rectangle if invalid
      rect  - rectangle only
    """
    h, w = bgr.shape[:2]

    if mode == "rect" or init_fg_mask is None:
        gc_mask = rect_initial_mask(bgr)
    else:
        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        fg = init_fg_mask > 0
        gc_mask[fg] = cv2.GC_PR_FGD

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

    binary = remove_border_connected_components(binary)
    return postprocess_mask_opencv(binary, keep_largest=True)


def build_alpha(mask: np.ndarray, feather: int) -> np.ndarray:
    """
    Convert binary mask to alpha channel.
    feather:
      0 means hard edge.
      >0 means Gaussian blur edge.
    """
    alpha = mask.copy()
    if feather > 0:
        k = max(3, int(feather))
        if k % 2 == 0:
            k += 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return alpha


def remove_background_grabcut(
    image: np.ndarray,
    max_size: int = 1200,
    grabcut_iter: int = 5,
    opencv_mode: str = "auto",
    feather: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OpenCV fallback.

    Return:
      rgba_removed: BGRA image with transparent background.
      mask_full:    uint8 binary mask, 0 background / 255 foreground.
    """
    if image is None:
        raise ValueError("Input image is None.")

    if image.ndim == 2:
        bgr_full = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr_full = image[:, :, :3].copy()
    else:
        bgr_full = image[:, :, :3].copy()

    original_h, original_w = bgr_full.shape[:2]
    bgr_small, scale = resize_keep_ratio(bgr_full, max_size=max_size)

    init_mask = None
    if opencv_mode in {"auto", "color"}:
        init_mask = color_distance_initial_mask(bgr_small)

    mask_small = grabcut_from_initial_mask(
        bgr=bgr_small,
        init_fg_mask=init_mask,
        iterations=grabcut_iter,
        mode=opencv_mode,
    )

    if scale != 1.0:
        mask_full = cv2.resize(
            mask_small,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_full = postprocess_mask_opencv(mask_full, keep_largest=True)
    else:
        mask_full = mask_small

    alpha = build_alpha(mask_full, feather=feather)

    bgra = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    return bgra, mask_full


def process_one_image(
    img_path: Path,
    input_root: Path,
    output_root: Path,
    mask_root: Optional[Path],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    try:
        if args.resolved_backend == "rembg":
            bgra, mask = remove_background_rembg(img_path, args)
        elif args.resolved_backend == "grabcut":
            image = imread_unicode(img_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                return False, f"Failed to read: {img_path}"

            bgra, mask = remove_background_grabcut(
                image=image,
                max_size=args.max_size,
                grabcut_iter=args.grabcut_iter,
                opencv_mode=args.opencv_mode,
                feather=args.feather,
            )
        else:
            return False, f"Unknown backend: {args.resolved_backend}"

        # If --save_mask is enabled, ONLY output mask.
        if args.save_mask:
            if mask_root is None:
                raise ValueError("mask_root should not be None when save_mask is enabled.")
            mask_path = rel_mask_path(
                img_path=img_path,
                input_root=input_root,
                mask_root=mask_root,
                mask_suffix=args.mask_suffix,
            )
            ok = imwrite_unicode(mask_path, mask)
            if not ok:
                return False, f"Failed to write mask: {mask_path}"
            return True, f"OK(mask only): {img_path} -> {mask_path}"

        # Otherwise output transparent-background PNG.
        out_path = rel_output_path(
            img_path=img_path,
            input_root=input_root,
            output_root=output_root,
            suffix=".png",
        )
        ok = imwrite_unicode(out_path, bgra)
        if not ok:
            return False, f"Failed to write: {out_path}"

        return True, f"OK: {img_path} -> {out_path}"

    except Exception as exc:
        return False, f"Failed: {img_path} | {type(exc).__name__}: {exc}"


def process_one_image_worker(
    img_path: Path,
    input_root: Path,
    output_root: Path,
    mask_root: Optional[Path],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    """
    Multiprocessing wrapper.
    Keep this function at top level so it can be pickled by multiprocessing.
    """
    cv2.setNumThreads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    return process_one_image(
        img_path=img_path,
        input_root=input_root,
        output_root=output_root,
        mask_root=mask_root,
        args=args,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch remove background from images and optionally export binary masks."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input image directory or a single image path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory. Original subdirectory structure is preserved.",
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        help="If enabled, ONLY save binary masks and do NOT save transparent-background images.",
    )
    parser.add_argument(
        "--mask_output_path",
        type=str,
        default=None,
        help="Mask output directory. Default: <output_path>/masks.",
    )
    parser.add_argument(
        "--mask_suffix",
        type=str,
        default="_mask",
        help="Suffix for mask files. Default: _mask.",
    )
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
        help=(
            "Segmentation backend. "
            "auto: use rembg if available, otherwise OpenCV GrabCut; "
            "rembg: deep-learning segmentation; "
            "grabcut: OpenCV fallback. Default: auto."
        ),
    )

    # rembg options
    parser.add_argument(
        "--rembg_model",
        type=str,
        default="isnet-general-use",
        help=(
            "rembg model name. Recommended for objects: isnet-general-use. "
            "Other common choices: u2net, u2netp. Default: isnet-general-use."
        ),
    )
    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=128,
        help="Alpha threshold for binary mask when using rembg. Default: 128.",
    )
    parser.add_argument(
        "--alpha_matting",
        action="store_true",
        help="Enable rembg alpha matting. Usually improves edge quality but is slower.",
    )
    parser.add_argument(
        "--alpha_matting_foreground_threshold",
        type=int,
        default=240,
        help="rembg alpha matting foreground threshold. Default: 240.",
    )
    parser.add_argument(
        "--alpha_matting_background_threshold",
        type=int,
        default=10,
        help="rembg alpha matting background threshold. Default: 10.",
    )
    parser.add_argument(
        "--alpha_matting_erode_size",
        type=int,
        default=10,
        help="rembg alpha matting erode size. Default: 10.",
    )
    parser.add_argument(
        "--post_process_mask",
        action="store_true",
        help="Enable rembg post_process_mask.",
    )

    # OpenCV fallback options
    parser.add_argument(
        "--opencv_mode",
        type=str,
        default="auto",
        choices=["auto", "color", "rect"],
        help=(
            "OpenCV GrabCut initialization mode. "
            "auto: color-distance first, fallback to rect; "
            "color: color-distance; rect: center rectangle. Default: auto."
        ),
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=1200,
        help="OpenCV fallback: resize long side for segmentation. 0 disables resizing. Default: 1200.",
    )
    parser.add_argument(
        "--grabcut_iter",
        type=int,
        default=5,
        help="OpenCV fallback: GrabCut iterations. Default: 5.",
    )
    parser.add_argument(
        "--feather",
        type=int,
        default=3,
        help="OpenCV fallback: alpha edge feather kernel size. 0 means hard edge. Default: 3.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help=(
            "Number of parallel worker processes. Default: 16. "
            "For rembg, reduce this if memory is insufficient."
        ),
    )

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

    output_root.mkdir(parents=True, exist_ok=True)
    if args.save_mask:
        mask_root.mkdir(parents=True, exist_ok=True)

    args.resolved_backend = resolve_backend(args.backend)

    if args.backend == "rembg" and args.resolved_backend != "rembg":
        raise RuntimeError("Backend was set to rembg, but rembg is not available.")

    print(f"Backend: {args.resolved_backend}")
    if args.resolved_backend == "grabcut":
        print(
            "Warning: using OpenCV GrabCut fallback. "
            "For hard images like metallic thin objects or textured backgrounds, "
            "install rembg for better results: pip install rembg pillow onnxruntime"
        )

    if args.resolved_backend == "rembg":
        print(f"rembg model: {args.rembg_model}")
        print("Preparing rembg model/session in main process...")
        warmup_rembg_model(args.rembg_model)

    images = list_images(input_path, recursive=not args.no_recursive)
    if not images:
        print(f"No images found in: {input_path}")
        return

    ok_count = 0
    fail_count = 0

    print(f"Found {len(images)} image(s).")
    print(f"Recursive scan: {not args.no_recursive}")
    if input_path.is_dir():
        print("Original subdirectory structure will be preserved in the output directory.")

    if args.save_mask:
        print("Mask-only mode enabled.")
        print(f"Output binary masks to: {mask_root}")
    else:
        print(f"Output transparent PNGs to: {output_root}")

    num_workers = max(1, int(args.num_workers))
    num_workers = min(num_workers, len(images))

    worker = partial(
        process_one_image_worker,
        input_root=input_path,
        output_root=output_root,
        mask_root=mask_root if args.save_mask else None,
        args=args,
    )

    if num_workers == 1:
        results_iter = map(worker, images)
        for idx, (ok, msg) in enumerate(results_iter, start=1):
            if ok:
                ok_count += 1
            else:
                fail_count += 1
            print(f"[{idx}/{len(images)}] {msg}")
    else:
        print(f"Using {num_workers} worker processes.")
        with Pool(processes=num_workers) as pool:
            for idx, (ok, msg) in enumerate(pool.imap_unordered(worker, images), start=1):
                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
                print(f"[{idx}/{len(images)}] {msg}")

    print("=" * 60)
    print(f"Done. Success: {ok_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()
