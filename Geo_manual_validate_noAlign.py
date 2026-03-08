# -*- coding: utf-8 -*-
"""Manual one-step validation for no-align setting.

This script mimics the validation data flow in ``Geo_train_noAlign.py`` / ``Geo_train.py``
for a single sample and saves a collage image similar to TB visualizations.
"""

import argparse
import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml

import Geo_train as base
from datasets.dataset_TSR import ContinuousEdgeLineDatasetMaskFinetune


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Manual validation (no-align) with TB-like collage output")
    p.add_argument("--config_path", type=str, default=None, help="YAML config path (optional)")
    p.add_argument("--validation_path", type=str, default=None, help="val image list txt")
    p.add_argument("--valid_mask_path", type=str, default=None, help="val mask list txt")
    p.add_argument("--val_wireframes_list", type=str, default=None, help="val wireframe pkl list txt")
    p.add_argument("--pretrain_ckpt", type=str, default=None, help="model ckpt path")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=16)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--AMP", action="store_true")

    p.add_argument("--seq_id", type=str, default=None, help="validate this sequence id")
    p.add_argument("--index", type=int, default=None, help="validate this dataset index directly")
    p.add_argument("--position", type=str, default="last", choices=["first", "middle", "last"],
                   help="target position inside selected sequence")
    p.add_argument("--output", type=str, default="manual_val_tb_collage.jpg", help="output collage path")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--mask_rates", type=float, nargs="+", default=[0.4, 0.8, 1.0])
    return p


def _get_nested(cfg, path):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _pick_first(cfg, paths):
    for p in paths:
        v = _get_nested(cfg, p)
        if v is not None:
            return v
    return None




def _flatten_dict(cfg, prefix=""):
    out = {}
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            nk = f"{prefix}.{k}" if prefix else str(k)
            out[nk] = v
            if isinstance(v, dict):
                out.update(_flatten_dict(v, nk))
    return out


def _normalize_path_from_cfg(path_value, config_path):
    p = str(path_value).strip().strip('"').strip("'")
    if not p:
        return p
    if os.path.isabs(p):
        return p
    base_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()
    return os.path.normpath(os.path.join(base_dir, p))


def _fallback_pick_from_keys(cfg, keys_any, keys_endswith):
    flat = _flatten_dict(cfg)
    for k, v in flat.items():
        if v in (None, ""):
            continue
        lk = k.lower()
        if any(token in lk for token in keys_any) or any(lk.endswith(suf) for suf in keys_endswith):
            return v
    return None
def _apply_yaml_overrides(opts):
    if not opts.config_path or not os.path.exists(opts.config_path):
        return opts
    with open(opts.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Support multiple config styles: train config + manual config section.
    # CLI still has highest priority; config only fills missing/default values.
    if opts.n_layer == 16:
        v = _pick_first(cfg, ["model.n_layer", "model_settings.n_layer", "manual_validate.n_layer"])
        if v is not None:
            opts.n_layer = int(v)
    if opts.n_embd == 256:
        v = _pick_first(cfg, ["model.n_embd", "model_settings.n_embd", "manual_validate.n_embd"])
        if v is not None:
            opts.n_embd = int(v)
    if opts.n_head == 8:
        v = _pick_first(cfg, ["model.n_head", "model_settings.n_head", "manual_validate.n_head"])
        if v is not None:
            opts.n_head = int(v)

    if not opts.pretrain_ckpt:
        v = _pick_first(cfg, ["model.pretrain_ckpt", "model_settings.pretrain_ckpt", "manual_validate.pretrain_ckpt"])
        if v is not None:
            opts.pretrain_ckpt = _normalize_path_from_cfg(v, opts.config_path)

    if not opts.validation_path:
        v = _pick_first(cfg, [
            "datasets.validation_path",
            "manual_validate.validation_path",
            "manual_validate.val_imgs_list",
            "manual_config.validation_path",
            "manual_config.val_imgs_list",
            "dataset1.val_imgs_list",
            "val_imgs_list",
            "validation_path",
        ])
        if v is None:
            v = _fallback_pick_from_keys(
                cfg,
                keys_any=["val_img", "validation_img", "validation_path", "val_images"],
                keys_endswith=["val_imgs_list", "validation_path"],
            )
        if v is not None:
            opts.validation_path = _normalize_path_from_cfg(v, opts.config_path)

    if not opts.val_wireframes_list:
        v = _pick_first(cfg, [
            "datasets.val_wireframes_list",
            "manual_validate.val_wireframes_list",
            "manual_validate.val_pkls_list",
            "manual_config.val_wireframes_list",
            "manual_config.val_pkls_list",
            "dataset1.val_pkls_list",
            "val_pkls_list",
            "val_wireframes_list",
        ])
        if v is None:
            v = _fallback_pick_from_keys(
                cfg,
                keys_any=["val_pkl", "wireframe", "line_list", "val_line"],
                keys_endswith=["val_pkls_list", "val_wireframes_list", "wireframes_list"],
            )
        if v is not None:
            opts.val_wireframes_list = _normalize_path_from_cfg(v, opts.config_path)

    if not opts.valid_mask_path:
        v = _pick_first(cfg, [
            "datasets.valid_mask_path",
            "masks.val_mask_list",
            "manual_validate.valid_mask_path",
            "manual_validate.val_mask_list",
            "manual_config.valid_mask_path",
            "manual_config.val_mask_list",
            "valid_mask_path",
            "val_mask_list",
        ])
        if v is None:
            v = _fallback_pick_from_keys(
                cfg,
                keys_any=["val_mask", "valid_mask"],
                keys_endswith=["val_mask_list", "valid_mask_path"],
            )
        if v is not None:
            opts.valid_mask_path = _normalize_path_from_cfg(v, opts.config_path)

    if opts.mask_rates == [0.4, 0.8, 1.0]:
        v = _pick_first(cfg, ["datasets.mask_rates", "manual_validate.mask_rates"])
        if v is not None:
            opts.mask_rates = [float(x) for x in v]

    if opts.image_size == 256:
        v = _pick_first(cfg, ["datasets.image_size", "manual_validate.image_size"])
        if v is not None:
            opts.image_size = int(v)

    if not opts.AMP:
        v = _pick_first(cfg, ["training_params.AMP", "manual_validate.AMP"])
        if v is not None:
            opts.AMP = bool(v)

    if opts.device == "cuda":
        v = _pick_first(cfg, ["manual_validate.device"])
        if v is not None:
            opts.device = str(v)

    if opts.seq_id is None:
        v = _pick_first(cfg, ["manual_validate.seq_id"])
        if v is not None:
            opts.seq_id = str(v)

    if opts.index is None:
        v = _pick_first(cfg, ["manual_validate.index"])
        if v is not None:
            opts.index = int(v)

    if opts.position == "last":
        v = _pick_first(cfg, ["manual_validate.position"])
        if v is not None:
            opts.position = str(v)

    if opts.output == "manual_val_tb_collage.jpg":
        v = _pick_first(cfg, ["manual_validate.output"])
        if v is not None:
            opts.output = _normalize_path_from_cfg(v, opts.config_path)

    return opts


def _ensure_ready(opts):
    required = ["validation_path", "valid_mask_path", "val_wireframes_list"]
    missing = [k for k in required if not getattr(opts, k)]
    if missing:
        raise ValueError(f"Missing required args: {missing}")


def _seq_sorted_indices(dataset, seq_id):
    idxs = dataset.seq_to_indices.get(seq_id, [])
    pairs = sorted([(dataset.image_id_list[i], i) for i in idxs], key=lambda x: x[0])
    return [i for _, i in pairs]


def _pick_target_index(dataset, seq_id=None, index=None, position="last"):
    if index is not None:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"index out of range: {index}")
        sid = dataset.index_to_seq[index]
        ordered = _seq_sorted_indices(dataset, sid)
        pos = ordered.index(index)
        return sid, ordered, pos

    if seq_id is None:
        seq_id = sorted(dataset.seq_to_indices.keys())[0]

    ordered = _seq_sorted_indices(dataset, seq_id)
    if not ordered:
        raise ValueError(f"empty sequence: {seq_id}")

    if position == "first":
        pos = 0
    elif position == "middle":
        pos = len(ordered) // 2
    else:
        pos = len(ordered) - 1
    return seq_id, ordered, pos


def _to_chw_01(t):
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.size(0) == 1:
        t = t.repeat(3, 1, 1)
    if t.size(0) > 3:
        t = t[:3]
    return t.detach().float().clamp(0, 1)


def _to_bgr_u8(chw):
    if chw.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(chw.shape)}")
    if chw.size(0) == 1:
        chw = chw.repeat(3, 1, 1)
    elif chw.size(0) not in (3, 4):
        raise ValueError(f"Invalid channel count for visualization: C={chw.size(0)}")
    arr = (chw.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _panel(img_bgr, title):
    pad = 30
    h, w = img_bgr.shape[:2]
    canvas = np.zeros((h + pad, w, 3), dtype=np.uint8)
    canvas[pad:, :, :] = img_bgr
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    opts = _build_parser().parse_args()
    opts = _apply_yaml_overrides(opts)
    _ensure_ready(opts)

    device = torch.device(opts.device if (opts.device == "cpu" or torch.cuda.is_available()) else "cpu")

    val_dataset = ContinuousEdgeLineDatasetMaskFinetune(
        pt_dataset=opts.validation_path,
        mask_path=opts.valid_mask_path,
        test_mask_path=opts.valid_mask_path,
        is_train=False,
        mask_rates=opts.mask_rates,
        image_size=opts.image_size,
        line_path=opts.val_wireframes_list,
    )

    seq_id, ordered, pos = _pick_target_index(val_dataset, opts.seq_id, opts.index, opts.position)
    t_idx = ordered[pos]
    p_idx = ordered[pos - 1] if pos > 0 else -1
    g_idx = ordered[0]

    # Build model via existing train utility (keeps ckpt loading behavior consistent)
    logger = base.build_logger(os.path.join("./ckpt", "manual_validate_tmp"))
    mopts = SimpleNamespace(
        n_layer=opts.n_layer,
        n_head=opts.n_head,
        n_embd=opts.n_embd,
        pretrain_ckpt=opts.pretrain_ckpt,
    )
    model = base.build_model(mopts, device, logger)
    model.eval()

    tgt = val_dataset[t_idx]
    g = val_dataset[g_idx]
    prev = val_dataset[p_idx] if p_idx >= 0 else None

    t_img = tgt["img"].unsqueeze(0).to(device=device, dtype=torch.float32)
    t_edge_gt = tgt["edge"].unsqueeze(0).to(device=device, dtype=torch.float32)
    t_line_gt = tgt["line"].unsqueeze(0).to(device=device, dtype=torch.float32)
    t_mask = tgt["mask"].unsqueeze(0).to(device=device, dtype=torch.float32)
    if t_mask.dim() == 3:
        t_mask = t_mask.unsqueeze(1)

    g_edge = g["edge"].unsqueeze(0).to(device=device, dtype=torch.float32)
    g_line = g["line"].unsqueeze(0).to(device=device, dtype=torch.float32)

    if prev is not None:
        l_edge = prev["edge"].unsqueeze(0).to(device=device, dtype=torch.float32)
        l_line = prev["line"].unsqueeze(0).to(device=device, dtype=torch.float32)
        l_mask = torch.zeros_like(t_mask)
    else:
        l_edge = torch.zeros_like(t_edge_gt)
        l_line = torch.zeros_like(t_line_gt)
        l_mask = torch.ones_like(t_mask)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=bool(opts.AMP and device.type == "cuda"), dtype=torch.bfloat16):
            ref_feat = model.extract_reference_features(
                global_img=torch.zeros_like(t_img),
                global_edge=g_edge,
                global_line=g_line,
                local_img=torch.zeros_like(t_img),
                local_edge=l_edge,
                local_line=l_line,
                local_mask=l_mask,
            )
            _, line_logits = model.forward_with_logits(
                img_idx=t_img,
                edge_idx=t_edge_gt,
                line_idx=t_line_gt,
                masks=t_mask,
                ref_feat=ref_feat,
            )

    line_pred = torch.sigmoid(line_logits).clamp(0, 1)

    # Keep TB-style 3-channel overlay:
    #   R: target GT line, G: predicted line, B: local reference line * alpha.
    overlay = torch.cat(
        [
            t_line_gt[0].clamp(0, 1),
            line_pred[0].clamp(0, 1),
            (l_line[0].clamp(0, 1) * 0.5),
        ],
        dim=0,
    )

    panels = [
        _panel(_to_bgr_u8(_to_chw_01(t_img)), "input_image"),
        _panel(_to_bgr_u8(_to_chw_01(t_mask)), "mask"),
        _panel(_to_bgr_u8(_to_chw_01(g_line)), "global_ref_line"),
        _panel(_to_bgr_u8(_to_chw_01(l_line)), "local_ref_line(prev)"),
        _panel(_to_bgr_u8(_to_chw_01(t_line_gt)), "target_line_gt"),
        _panel(_to_bgr_u8(_to_chw_01(line_pred)), "pred_line"),
        _panel(_to_bgr_u8(overlay), "overlay_line_tb_style"),
    ]

    h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < h:
            ext = np.zeros((h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.concatenate([p, ext], axis=0)
        padded.append(p)
    collage = np.concatenate(padded, axis=1)

    os.makedirs(os.path.dirname(opts.output) or ".", exist_ok=True)
    cv2.imwrite(opts.output, collage)

    print(f"[OK] seq={seq_id} t_idx={t_idx} prev_idx={p_idx} global_idx={g_idx}")
    print(f"[OK] saved collage: {opts.output}")


if __name__ == "__main__":
    main()
