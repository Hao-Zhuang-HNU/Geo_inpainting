# -*- coding: utf-8 -*-
"""No-align training entry based on Geo_train.py.

Changes:
- Remove homography/npz alignment logic.
- Local reference frame is strictly previous frame (when available).
"""

import argparse
import os
import collections

import torch
from torch.utils.data import DataLoader, Dataset
import Geo_train as base

_ORIG_BUILD_DATASETS_AND_LOADER = base.build_datasets_and_loader
_ORIG_EVALUATE_SEQUENCE = base.evaluate_sequence
_ORIG_LOAD_CONFIG_TO_OPTS = base.load_config_to_opts
_ORIG_PREPARE_FIXED_VALIDATION_SET = base.prepare_fixed_validation_set
from datasets.dataset_TSR import (
    ContinuousEdgeLineDatasetMask,
    ContinuousEdgeLineDatasetMaskFinetune,
)


class MaPDatasetWrapperNoAlign(Dataset):
    def __init__(
        self,
        dataset,
        seq_to_indices,
        npz_path_list=None,
        logger=None,
        local_used_gt=False,
        no_global_ref=False,
        no_local_ref=False,
        local_used_last_frame=False,
        **kwargs,
    ):
        self.dataset = dataset
        self.seq_to_indices = seq_to_indices
        self.logger = logger
        self.no_global_ref = no_global_ref
        self.no_local_ref = no_local_ref
        # keep signature-compatible with Geo_train.MaPDatasetWrapper
        self.npz_path_list = npz_path_list
        self.local_used_gt = local_used_gt
        self.local_used_last_frame = local_used_last_frame

        self.idx_info = {}
        count = 0
        for seq_id, idxs in seq_to_indices.items():
            try:
                paths = [self.dataset.image_id_list[i] for i in idxs]
                sorted_pairs = sorted(zip(paths, idxs), key=lambda x: x[0])
                sorted_idxs = [p[1] for p in sorted_pairs]
            except Exception:
                sorted_idxs = sorted(idxs)

            global_ref_idx = sorted_idxs[0]
            for i, curr_idx in enumerate(sorted_idxs):
                self.idx_info[curr_idx] = {
                    "seq_id": seq_id,
                    "is_first": (i == 0),
                    "prev_idx": sorted_idxs[i - 1] if i > 0 else -1,
                    "global_idx": global_ref_idx,
                }
            count += 1

        if self.logger:
            self.logger.info(f"[Dataset-NoAlign] Configured {count} sequences. local_ref=previous_frame_only")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        curr_item = self.dataset[idx]
        info = self.idx_info[idx]

        is_first = 1.0 if info["is_first"] else 0.0
        current_conf = 0.0
        npz_ok = 1.0
        warp_valid = 0.0
        local_used = 0.0

        if self.no_global_ref:
            g_edge = torch.zeros_like(curr_item["edge"])
            g_line = torch.zeros_like(curr_item["line"])
        elif idx == info["global_idx"]:
            g_edge = curr_item["edge"]
            g_line = curr_item["line"]
        else:
            g_item = self.dataset[info["global_idx"]]
            g_edge = g_item["edge"]
            g_line = g_item["line"]

        if self.no_local_ref or info["is_first"]:
            l_edge_t = torch.zeros_like(curr_item["edge"])
            l_line_t = torch.zeros_like(curr_item["line"])
            l_mask_t = torch.ones_like(curr_item["mask"])
        else:
            prev_item = self.dataset[info["prev_idx"]]
            l_edge_t = prev_item["edge"].clone()
            l_line_t = prev_item["line"].clone()
            l_mask_t = torch.zeros_like(curr_item["mask"])
            current_conf = 1.0
            warp_valid = 1.0
            local_used = 1.0

        if l_mask_t.dim() == 2:
            l_mask_t = l_mask_t.unsqueeze(0)

        return {
            "c_img": curr_item["img"].contiguous(),
            "c_edge": curr_item["edge"].contiguous(),
            "c_line": curr_item["line"].contiguous(),
            "c_mask": curr_item["mask"].contiguous(),
            "g_edge": g_edge.contiguous(),
            "g_line": g_line.contiguous(),
            "l_edge": l_edge_t.contiguous(),
            "l_line": l_line_t.contiguous(),
            "l_mask": l_mask_t.contiguous(),
            "conf": torch.tensor(current_conf, dtype=torch.float32),
            "npz_ok": torch.tensor(npz_ok, dtype=torch.float32),
            "warp_valid": torch.tensor(warp_valid, dtype=torch.float32),
            "local_used": torch.tensor(local_used, dtype=torch.float32),
            "is_first": torch.tensor(is_first, dtype=torch.float32),
            "seq_hash": hash(str(info["seq_id"])),
            "orig_idx": idx,
        }


def build_datasets_and_loader_noalign(opts, logger, train_npz_list=None):
    if not opts.MaP:
        raise ValueError("Only MaP mode is supported.")

    base_dataset = ContinuousEdgeLineDatasetMask(
        pt_dataset=opts.data_path,
        mask_path=opts.mask_path,
        test_mask_path=opts.mask_path,
        is_train=True,
        mask_rates=opts.mask_rates,
        image_size=opts.image_size,
        line_path=opts.train_wireframes_list,
    )

    seq_to_indices = collections.defaultdict(list)
    for i, path in enumerate(base_dataset.image_id_list):
        seq_to_indices[os.path.dirname(path)].append(i)

    train_wrapper = MaPDatasetWrapperNoAlign(
        base_dataset,
        seq_to_indices,
        logger=logger,
        no_global_ref=getattr(opts, "no_global_ref", False),
        no_local_ref=getattr(opts, "no_local_ref", False),
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_wrapper, shuffle=True) if opts.dist else None

    train_loader = DataLoader(
        train_wrapper,
        batch_size=opts.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=opts.num_workers,
        pin_memory=True,
        persistent_workers=opts.persistent_workers,
        prefetch_factor=opts.prefetch_factor,
        worker_init_fn=base._dataloader_worker_init,
        drop_last=False,
    )

    val_dataset = ContinuousEdgeLineDatasetMaskFinetune(
        pt_dataset=opts.validation_path,
        mask_path=opts.valid_mask_path,
        test_mask_path=opts.valid_mask_path,
        is_train=False,
        mask_rates=opts.mask_rates,
        image_size=opts.image_size,
        line_path=opts.val_wireframes_list,
    )
    return train_loader, val_dataset, train_sampler, base_dataset


@torch.no_grad()
def evaluate_sequence_noalign(model, val_dataset, seq_to_ref, device, logger, amp, opts, writer=None, epoch=0, val_npz_list=None):
    # Delegate to base evaluator with no-align options forced.
    return _ORIG_EVALUATE_SEQUENCE(model, val_dataset, seq_to_ref, device, logger, amp, opts, writer=writer, epoch=epoch, val_npz_list=None)


def prepare_fixed_validation_set_noalign(opts, val_dataset, logger):
    """Reuse base fixed-frame selection, then remap seq keys for TB to真实序列名."""
    _ORIG_PREPARE_FIXED_VALIDATION_SET(opts, val_dataset, logger)

    fixed_meta = getattr(opts, "_val_viz_fixed15_meta", {})
    if not fixed_meta:
        return

    remapped = {}
    for old_sid, info in fixed_meta.items():
        t_idx = int(info.get("t_idx", -1))
        seq_name = str(old_sid)

        if 0 <= t_idx < len(getattr(val_dataset, "image_id_list", [])):
            img_path = val_dataset.image_id_list[t_idx]
            seq_name = os.path.basename(os.path.dirname(img_path)) or os.path.dirname(img_path) or str(old_sid)

        final_name = seq_name
        dedup_id = 1
        while final_name in remapped:
            dedup_id += 1
            final_name = f"{seq_name}_{dedup_id}"

        remapped[final_name] = info

    opts._val_viz_fixed15_meta = remapped


def load_config_to_opts_noalign(opts):
    opts = _ORIG_LOAD_CONFIG_TO_OPTS(opts)
    # Force no-align behavior
    opts.local_used_gt = False
    opts.local_used_last_frame = True
    opts.train_npz_list = None
    opts.val_npz_list = None
    return opts


def main_worker_noalign(opts):
    # Monkey patches used by base.main_worker
    base.MaPDatasetWrapper = MaPDatasetWrapperNoAlign
    base.build_datasets_and_loader = build_datasets_and_loader_noalign
    base.evaluate_sequence = evaluate_sequence_noalign
    base.load_config_to_opts = load_config_to_opts_noalign
    base.prepare_fixed_validation_set = prepare_fixed_validation_set_noalign

    # Force no-align behavior regardless of CLI/YAML
    opts.local_used_gt = False
    opts.local_used_last_frame = True
    opts.train_npz_list = None
    opts.val_npz_list = None

    return base.main_worker(opts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--name", type=str, default="RefKV_train_seq_noAlign")
    parser.add_argument("--GPU_ids", type=str, default="0")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt")
    parser.add_argument("--data_path", type=str, nargs='+', default=None)
    parser.add_argument("--validation_path", type=str, nargs='+', default=None)
    parser.add_argument("--mask_path", type=str, default="")
    parser.add_argument("--valid_mask_path", type=str, default="")
    parser.add_argument("--train_wireframes_list", type=str, nargs='+', default=None)
    parser.add_argument("--val_wireframes_list", type=str, nargs='+', default=None)
    parser.add_argument("--train_npz_list", type=str, nargs='+', default=None)
    parser.add_argument("--val_npz_list", type=str, nargs='+', default=None)
    parser.add_argument("--mask_rates", type=float, nargs="+", default=[0.4, 0.8, 1.0])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=16)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_epoch", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--AMP", action="store_true")
    parser.add_argument("--MaP", action="store_true")
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--tb_logdir", type=str, default=None)
    parser.add_argument("--tb_images", action="store_true")
    parser.add_argument("--tb_max_images", type=int, default=3)
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_freq", type=int, default=2)
    parser.add_argument("--focal_weight", type=float, default=1.0)
    parser.add_argument("--tversky_weight", type=float, default=1.0)
    parser.add_argument("--disable_edge", action="store_true")
    parser.add_argument("--line_tv_alpha", type=float, default=0.3)
    parser.add_argument("--line_tv_beta", type=float, default=0.7)
    parser.add_argument("--dilate_switch_ep", type=int, default=3)
    parser.add_argument("--dilate1_ep", type=int, default=1)
    parser.add_argument("--no_global_ref", action="store_true")
    parser.add_argument("--no_local_ref", action="store_true")
    parser.add_argument("--debug_line", action="store_true")
    parser.add_argument("--debug_line_dir", type=str, default="debug_line")
    parser.add_argument("--check_ref_frame", action="store_true")
    parser.add_argument("--check_ref_frame_dir", type=str, default="check_ref_frame")
    parser.add_argument("--check_ref_frame_step", type=int, default=1000)
    parser.add_argument("--check_ref_frame_random_samples", type=int, default=20)

    opts = parser.parse_args()

    if opts.GPU_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = opts.GPU_ids

    main_worker_noalign(opts)
