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
    # x: BCHW / CHW / HW，数值任意范围
    import numpy as np, cv2, torch
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu()
        if x.dim() == 4:      # [B,1,H,W] 或 [B,C,H,W]
            x = x[0, 0]
        elif x.dim() == 3:    # [1,H,W] / [C,H,W]
            x = x[0]
    x = x.numpy() if hasattr(x, "numpy") else np.array(x)
    # 按每张图做 min-max 归一化，确保可见
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
):
    """
    Progressive sampling for edge/line completion.
    内部用连续值更新，仅在保存图时二值化副本 edge_vis/line_vis。

    Returns:
        edge, line  (连续值，范围[0,1]，未揭露区已清零)
    """

    [img, edge, line] = context

    # ------- 设备选择（兼容无参数模型） -------
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

    # 初始：洞内清零，仅保留可见上下文
    img  = img  * (1 - mask)
    edge = edge * (1 - mask)
    line = line * (1 - mask)

    model.eval()
    with torch.no_grad():
        for i in range(iterations):
            # ------- 前向：得到两个分支的 logits -------
            # 若你的模型/CPU 不支持 fp16，可改成 float32
            e_logits, l_logits = model.forward_with_logits(
                img.to(torch.float16) if img.dtype == torch.float16 else img,
                edge.to(torch.float16) if edge.dtype == torch.float16 else edge,
                line.to(torch.float16) if line.dtype == torch.float16 else line,
                masks=mask.to(torch.float16) if mask.dtype == torch.float16 else mask
            )

            # ------- 概率图（注意 line 的锐化） -------
            edge_pred = torch.sigmoid(e_logits)
            line_pred = torch.sigmoid((l_logits + add_v) * mul_v)

            # ------- 连续值回写（只在洞内），保留上下文连续性 -------
            edge = (edge + edge_pred * mask).clamp_(0, 1)
            line = (line + line_pred * mask).clamp_(0, 1)

            # ------- 概率融合排序，渐进揭露 -------
            b, _, h, w = edge_pred.shape
            edge_pred_f = edge_pred.reshape(b, -1, 1)
            line_pred_f = line_pred.reshape(b, -1, 1)
            mask_f      = mask.reshape(b, -1)

            edge_probs = torch.cat([1 - edge_pred_f, edge_pred_f], dim=-1)
            line_probs = torch.cat([1 - line_pred_f, line_pred_f], dim=-1)

            # 提升正类优先级
            if prob_boost and prob_boost != 0.0:
                edge_probs[:, :, 1] = edge_probs[:, :, 1] + prob_boost
                line_probs[:, :, 1] = line_probs[:, :, 1] + prob_boost

            # 非洞区打大负号，避免被选中
            edge_max = edge_probs.max(dim=-1)[0] + (1 - mask_f) * (-100.0)
            line_max = line_probs.max(dim=-1)[0] + (1 - mask_f) * (-100.0)

            indices = torch.sort(edge_max + line_max, dim=-1, descending=True)[1]

            for ii in range(b):
                total_holes = int(torch.sum(mask_f[ii]).item())
                keep = int((i + 1) / iterations * total_holes)
                if keep > 0:
                    # 仅揭露洞内 top-k
                    sel = indices[ii, :keep]
                    # 保险：确保被揭露的确在洞内
                    # （如你的实现不需要可去掉断言）
                    # assert torch.sum(mask_f[ii][sel]).item() == keep, "Reveal set contains non-hole pixels!"
                    mask_f[ii][sel] = 0  # 0 表示“已揭露”

            # 还原形状；清零未揭露区（保持上下文仅来自可见+已揭露）
            mask = mask_f.reshape(b, 1, h, w)
            edge = edge * (1 - mask)
            line = line * (1 - mask)

            # ------- 仅用于可视化的二值化：不回写到 edge/line -------
            if debug_save_dir is not None:
                try:
                    os.makedirs(debug_save_dir, exist_ok=True)
                    # 全图二值化副本
                    edge_vis = (edge >= edge_bin_thr).float()
                    line_vis = (line >= line_bin_thr).float()

                    # 你项目里若已有 _save_gray_prob 用它；否则用 torchvision 保存
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

        # 返回连续值结果（未揭露区已清零）
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
