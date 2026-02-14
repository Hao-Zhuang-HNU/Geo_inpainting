import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as FF
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR

##--##
import cv2
##--##




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            # mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB',
                    (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)


class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


# progressiveley sampling edge line
##-251031-## 增加每轮迭代查看详细图片功能
def _save_gray_prob(x, path):
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu()
        if x.dim() == 4:
            x = x[0, 0]
        elif x.dim() == 3:
            x = x[0]
    x = x.numpy() if hasattr(x, "numpy") else np.array(x)
    xmin, xmax = float(x.min()), float(x.max())
    if xmax > xmin:
        x = (x - xmin) / (xmax - xmin)
    else:
        x = np.zeros_like(x)
    cv2.imwrite(path, (x * 255.0).astype(np.uint8))


def SampleEdgeLineLogits(
    model, context, mask=None, iterations=1, device='cuda',
    add_v=0, mul_v=4,
    debug_save_dir=None,             # 可选：保存中间可视化
    edge_bin_thr=0.25,               # 可视化阈值
    line_bin_thr=0.25,
    prob_boost=0.5,                  # 正类概率加权
    ref_feat=None,                  #加入参考特征
):
    """
    Progressive sampling for edge/line completion with optional Ref-KV guidance
    Args:
        model: TSR model
        context: [img, edge, line] tensors
        mask: [B, 1, H, W] mask tensor (1=hole, 0=visible)
        iterations: number of progressive sampling iterations
        device: computation device
        add_v, mul_v: line prediction adjustment parameters
        debug_save_dir: optional directory to save intermediate visualizations
        edge_bin_thr, line_bin_thr: binarization thresholds for visualization
        prob_boost: boost positive class probability for selection
        ref_feat: [B, C, H', W'] reference features (optional, for Ref-KV)
    
    Returns:
        edge, line: [B, 1, H, W] completed continuous predictions (range [0,1])
    """

    [img, edge, line] = context
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = img.device if hasattr(img, "device") else torch.device(device)
    if isinstance(device, str):
        dev = torch.device(device) if device else dev

    img  = img.to(dev)
    edge = edge.to(dev)
    line = line.to(dev)

    if mask is None:
        mask = torch.zeros_like(edge, device=dev)
    else:
        mask = mask.to(dev)

    # Move reference features to device if provided 参考特征移入到设备中
    if ref_feat is not None:
        ref_feat = ref_feat.to(dev)

    # Initialize: clear holes, keep visible context 洞内清零
    img  = img  * (1 - mask)
    edge = edge * (1 - mask)
    line = line * (1 - mask)

    model.eval()
    with torch.no_grad():
        for i in range(iterations):
            # ------- 前向：得到两个分支的 logits -------
            # e_logits, l_logits = model.forward_with_logits(
            #     img.to(torch.float16) if img.dtype == torch.float16 else img,
            #     edge.to(torch.float16) if edge.dtype == torch.float16 else edge,
            #     line.to(torch.float16) if line.dtype == torch.float16 else line,
            #     masks=mask.to(torch.float16) if mask.dtype == torch.float16 else mask
            # )
            if ref_feat is not None and hasattr(model, 'use_ref_kv') and model.use_ref_kv:
                e_logits, l_logits = model.forward_with_logits(
                    img, edge, line, masks=mask, ref_feat=ref_feat
                )
            else:
                e_logits, l_logits = model.forward_with_logits(
                    img, edge, line, masks=mask #原版，但是没有强转16
                )


            # Probability maps概率图
            edge_pred = torch.sigmoid(e_logits)
            line_pred = torch.sigmoid((l_logits + add_v) * mul_v)

            # Update continuous values (only in holes)更新连续值（只在洞内）
            edge = (edge + edge_pred * mask).clamp_(0, 1)
            line = (line + line_pred * mask).clamp_(0, 1)

            # 洞内渐进揭露
            b, _, h, w = edge_pred.shape
            edge_pred_f = edge_pred.reshape(b, -1, 1)
            line_pred_f = line_pred.reshape(b, -1, 1)
            mask_f      = mask.reshape(b, -1)

            edge_probs = torch.cat([1 - edge_pred_f, edge_pred_f], dim=-1)
            line_probs = torch.cat([1 - line_pred_f, line_pred_f], dim=-1)

            # Boost positive class priority 提升积极类的优先级
            if prob_boost and prob_boost != 0.0:
                edge_probs[:, :, 1] = edge_probs[:, :, 1] + prob_boost
                line_probs[:, :, 1] = line_probs[:, :, 1] + prob_boost

            # Penalize non-hole regions 惩罚非洞区
            edge_max = edge_probs.max(dim=-1)[0] + (1 - mask_f) * (-100.0)
            line_max = line_probs.max(dim=-1)[0] + (1 - mask_f) * (-100.0)

            indices = torch.sort(edge_max + line_max, dim=-1, descending=True)[1]

            for ii in range(b):
                total_holes = int(torch.sum(mask_f[ii]).item())
                keep = int((i + 1) / iterations * total_holes)
                if keep > 0:
                    sel = indices[ii, :keep]
                    mask_f[ii][sel] = 0  # 0 = revealed 0 表示“已揭露”

            # Reshape mask and clear unrevealed regions 还原mask；清零未揭露区
            mask = mask_f.reshape(b, 1, h, w)
            edge = edge * (1 - mask)
            line = line * (1 - mask)

            # Visualization only (binarized copies, don't write back to edge/line) 二进制可视化
            if debug_save_dir is not None:
                try:
                    os.makedirs(debug_save_dir, exist_ok=True)
                    edge_vis = (edge >= edge_bin_thr).float()
                    line_vis = (line >= line_bin_thr).float()

                    if "_save_gray_prob" in globals():
                        _save_gray_prob(edge_vis, os.path.join(debug_save_dir, f"iter_{i:02d}_edge_bin.png"))
                        _save_gray_prob(line_vis, os.path.join(debug_save_dir, f"iter_{i:02d}_line_bin.png"))
                    else:
                        import torchvision.utils as vutils
                        vutils.save_image(edge_vis, os.path.join(debug_save_dir, f"iter_{i:02d}_edge_bin.png"))
                        vutils.save_image(line_vis, os.path.join(debug_save_dir, f"iter_{i:02d}_line_bin.png"))
                except Exception as _:
                    # 可视化失败不影响主流程
                    pass

        return edge, line

def SampleEdgeLineLogitsWithRefExtraction(model, context, mask, iterations, debug_save_dir=None, ref_feat=None, extract_ref=False):
    """
    采样/推理的主循环函数。
    修改适配：适配新的 extract_reference_features 接口 (global_img/local_img)
    """
    # context: [img, edge, line]
    img, edge, line = context
    
    # 确保在 GPU
    if img.device != next(model.parameters()).device:
        img = img.to(next(model.parameters()).device)
        edge = edge.to(next(model.parameters()).device)
        line = line.to(next(model.parameters()).device)
        mask = mask.to(next(model.parameters()).device)

    # 迭代推理
    for i in range(iterations):
        # 1. 输入当前状态，获取预测
        # forward_with_logits 的接口没变，依然是 img_idx, ...
        edge_logits, line_logits = model.forward_with_logits(
            img_idx=img, 
            edge_idx=edge, 
            line_idx=line, 
            masks=mask, 
            ref_feat=ref_feat
        )

        # 2. Sigmoid 归一化
        edge_prob = torch.sigmoid(edge_logits)
        line_prob = torch.sigmoid(line_logits)

        # 3. 更新输入 (把预测结果填回 mask 区域，作为下一次迭代的输入)
        # 非 Mask 区域保持 GT，Mask 区域更新为预测值
        edge = edge * (1 - mask) + edge_prob * mask
        line = line * (1 - mask) + line_prob * mask
        
        # (可选) 保存中间结果用于调试
        if debug_save_dir is not None:
            import os
            import cv2
            import numpy as np
            
            # 转为可视化的 numpy
            def to_vis(t):
                return (t[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            
            # 合成图片
            vis_edge = to_vis(edge)
            vis_line = to_vis(line)
            
            # 保存
            cv2.imwrite(os.path.join(debug_save_dir, f"iter_{i}_edge.png"), vis_edge)
            cv2.imwrite(os.path.join(debug_save_dir, f"iter_{i}_line.png"), vis_line)

    # 推理结束
    
    # 4. (关键修改) 如果需要提取参考特征
    new_ref_feat = None
    if extract_ref:
        with torch.no_grad():
            # 【修改点】: 使用新的参数名 global_img, global_edge, ...
            # 推理阶段，我们把补全后的结果作为 Global Reference
            new_ref_feat = model.extract_reference_features(
                global_img=img, 
                global_edge=edge, 
                global_line=line
                # local_img=None (推理阶段提取参考帧特征时，通常只提取它作为 Global Ref)
            )

    return edge, line, new_ref_feat

def SampleEdgeLineLogits_Standard(model, context, mask=None, iterations=1, device='cuda', add_v=0, mul_v=4):
    [img, edge, line] = context
    img = img.to(device)
    edge = edge.to(device)
    line = line.to(device)
    mask = mask.to(device)
    
    # 初始：遮罩区域置零
    img = img * (1 - mask)
    edge = edge * (1 - mask)
    line = line * (1 - mask)
    
    model.eval()
    with torch.no_grad():
        for i in range(iterations):
            # 注意：原版强制转了 float16，如果模型报错可以去掉 .to(torch.float16)
            edge_logits, line_logits = model.forward_with_logits(
                img, edge, line, masks=mask
            )
            
            edge_pred = torch.sigmoid(edge_logits)
            # 这里的 mul_v=4 非常重要
            line_pred = torch.sigmoid((line_logits + add_v) * mul_v)
            
            edge = edge + edge_pred * mask
            
            # 【关键修复】还原 Places2 模型依赖的二值化逻辑
            edge[edge >= 0.25] = 1
            edge[edge < 0.25] = 0
            
            line = line + line_pred * mask

            b, _, h, w = edge_pred.shape
            edge_pred_f = edge_pred.reshape(b, -1, 1)
            line_pred_f = line_pred.reshape(b, -1, 1)
            mask_f = mask.reshape(b, -1)

            # 模拟原版的置信度计算
            edge_probs = torch.cat([1 - edge_pred_f, edge_pred_f], dim=-1)
            line_probs = torch.cat([1 - line_pred_f, line_pred_f], dim=-1)
            edge_probs[:, :, 1] += 0.5
            line_probs[:, :, 1] += 0.5
            
            edge_max_probs = edge_probs.max(dim=-1)[0] + (1 - mask_f) * (-100)
            line_max_probs = line_probs.max(dim=-1)[0] + (1 - mask_f) * (-100)

            indices = torch.sort(edge_max_probs + line_max_probs, dim=-1, descending=True)[1]

            for ii in range(b):
                # 渐进式揭露遮罩
                keep = int((i + 1) / iterations * torch.sum(mask_f[ii, ...]))
                mask_f[ii][indices[ii, :keep]] = 0

            mask = mask_f.reshape(b, 1, h, w)
            edge = edge * (1 - mask)
            line = line * (1 - mask)

        return edge, line

def get_lr_schedule_with_warmup(optimizer, num_warmup_steps, milestone_step, gamma, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            lr_weight = 1.0
            decay_times = current_step // milestone_step
            for _ in range(decay_times):
                lr_weight *= gamma
        return lr_weight

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def torch_init_model(model, total_dict, key, rank=0):
    state_dict = total_dict[key]
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    if rank == 0:
        print("missing keys:{}".format(missing_keys))
        print('unexpected keys:{}'.format(unexpected_keys))
        print('error msgs:{}'.format(error_msgs))


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = FF.to_tensor(img).float()
    return img_t


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    if isinstance(data, list):
        return [to_device(d, device) for d in data]
