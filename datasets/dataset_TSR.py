import os
import random
import sys
from glob import glob

import io


import cv2
import numpy as np
import torchvision.transforms.functional as F
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.utils.data import Dataset
import pickle5 as pickle
import skimage.draw
from collections import Counter


sys.path.append('..')


def to_int(x):
    return tuple(map(int, x))


class ContinuousEdgeLineDatasetMask(Dataset):
    """
    1) line_path为pkls_list线框清单，不提供则报错
    2) 修复匹配规则：
       - 原始图片名为 frame_00086.png
       - 线框文件名为 <子目录名>_images_8_frame_00086.pkl
       - 即 .pkl 文件名需要以 "_<原图片名>" 作为后缀匹配。
    """

    def __init__(self, pt_dataset, mask_path=None, test_mask_path=None, is_train=False,
                 mask_rates=None, image_size=256, line_path=None):

        self.is_train = is_train
        self.pt_dataset = pt_dataset

        ##-251104-## 读取图像清单
        self.image_id_list = []
        with open(self.pt_dataset) as f:
            for line in f:
                self.image_id_list.append(line.strip())


        # 训练 / 测试掩码列表
        if is_train:
            # 训练集：mask_path 是清单 .txt（每行一个掩码绝对路径）
            with open(mask_path, 'r', encoding='utf-8') as f:
                self.mask_list = sorted(
                    [ln.strip().strip('"').strip("'") for ln in f if ln.strip() and not ln.lstrip().startswith('#')],
                    key=lambda x: x.split('/')[-1]
                )
        else:
            # 验证集：valid_mask_path 也按清单 .txt 读取（与训练一致）
            with open(test_mask_path, 'r', encoding='utf-8') as f:
                self.mask_list = sorted(
                    [ln.strip().strip('"').strip("'") for ln in f if ln.strip() and not ln.lstrip().startswith('#')],
                    key=lambda x: x.split('/')[-1]
                )
            # 可选：把掩码清单“复制填满”到与验证图片数相同的长度（避免每次取模）
            if len(self.mask_list) > 0 and len(self.mask_list) < len(self.image_id_list):
                reps = (len(self.image_id_list) + len(self.mask_list) - 1) // len(self.mask_list)
                self.mask_list = (self.mask_list * reps)[:len(self.image_id_list)]


        self.image_size = image_size
        self.training = is_train
        self.mask_rates = mask_rates
        self.line_path = line_path

        # —— 线框 .pkl 索引 ——
        self.line_dict = []  # list of (seq_id, line_path, basename)
        if self.line_path is not None:
            if not os.path.isfile(self.line_path):
                raise FileNotFoundError(f"pkl_list 不存在: {self.line_path}")
            with open(self.line_path, 'r') as f:
                pkl_paths = [ln.strip() for ln in f if ln.strip()]
            pkl_paths = [p for p in pkl_paths if not p.lstrip().startswith('#')]

            # 统计“子目录名”是否重名（跨父目录）
            child_names_pkl = [os.path.basename(os.path.dirname(p)) for p in pkl_paths]
            child_cnt_pkl = Counter(child_names_pkl)
            # 重名集合：这些子目录名在pkl里出现>1
            dup_children_pkl = {name for name, c in child_cnt_pkl.items() if c > 1}

            def _seq_id_for_pkl(path: str) -> str:
                """若子目录名冲突，则退化为父级目录名；否则用子目录名。"""
                child = os.path.basename(os.path.dirname(path))                  # 子目录名
                if child in dup_children_pkl:
                    parent = os.path.basename(os.path.dirname(os.path.dirname(path)))  # 上一级目录
                    # 如果父级目录也可能为空（极少见），兜底回退到子目录名
                    return parent if parent else child
                return child

            # 遍历清单并把信息提取为三元组
            for p in pkl_paths:
                if not p.lower().endswith('.pkl'):
                    continue
                if not os.path.isabs(p):
                    raise ValueError(f"pkl_list 中包含非绝对路径: {p}")
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"pkl_list 中的文件不存在: {p}")
                basename = os.path.splitext(os.path.basename(p))[0]
                seq_id = _seq_id_for_pkl(p)
                self.line_dict.append((seq_id, p, basename))
        else:
            raise ValueError("必须提供 pkl_list")


        # ##-- imgs_dict定义 --##
        # imgs_dict: list of (seq_id, img_path, basename)
        # seq_id 判定规则：
        #   - 默认：使用图片路径的“子目录名”
        #   - 若子目录名在全体图片中存在重名冲突（跨不同父目录），则改用“上一级（父级）目录名”
        self.imgs_dict = []
        self.seq_to_indices = {}   # seq_id -> [idxs]（用于Finetune阶段按序列采样）
        self.index_to_seq = []     # 与 image_id_list 对齐，记录每个idx所属seq_id

        # 统计子目录名重名
        child_names_img = [os.path.basename(os.path.dirname(p)) for p in self.image_id_list]
        child_cnt_img = Counter(child_names_img)
        dup_children_img = {name for name, c in child_cnt_img.items() if c > 1}

        def _seq_id_for_img(path: str) -> str:
            child = os.path.basename(os.path.dirname(path))                      # 子目录名
            if child in dup_children_img:
                parent = os.path.basename(os.path.dirname(os.path.dirname(path)))     # 上一级目录
                return parent if parent else child
            return child

        for j, img_p in enumerate(self.image_id_list):
            img_basename = os.path.splitext(os.path.basename(img_p))[0]
            seq_id = _seq_id_for_img(img_p)
            self.imgs_dict.append((seq_id, img_p, img_basename))
            self.index_to_seq.append(seq_id)
            self.seq_to_indices.setdefault(seq_id, []).append(j)

        # 方便使用的序列列表
        self.sequences = sorted(list(self.seq_to_indices.keys()))

        # —— 反向索引（basename -> list of indices into line_dict / imgs_dict） —— 便于快速第三维匹配
        self.line_index = {}  # basename -> [indices into self.line_dict]
        for idx, (_, path, bname) in enumerate(self.line_dict):
            key = bname
            if key not in self.line_index:
                self.line_index[key] = []
            self.line_index[key].append(idx)

        self.img_index = {}  # basename -> [indices into self.imgs_dict]
        for idx, (_, path, bname) in enumerate(self.imgs_dict):
            key = bname
            if key not in self.img_index:
                self.img_index[key] = []
            self.img_index[key].append(idx)

        # 兼容/提示：若需要旧版 dict(key->path) 的接口，可同时生成一个映射（可选）
        # 下面这行保留了一个老接口 self.line_map（basename -> first path），以免其他旧代码马上崩掉
        self.line_map = {}
        for seq_id, pth, bname in self.line_dict:
            if bname not in self.line_map:
                self.line_map[bname] = pth  # 保存第一个出现的映射

        self.wireframe_th = 0.85

    def __len__(self):
        return len(self.image_id_list)

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)
        return img

    def load_mask(self, img, index):
        """
        简化版 load_mask：
        - 不再依赖或检测 self.mask_dict（已移除冗余判断）
        - 训练/验证统一从 self.mask_list 中取 mask_path（按 index % len(self.mask_list) 循环）
        - 保持原来的图像大小适配和二值化逻辑
        """
        imgh, imgw = img.shape[0:2]

        if len(self.mask_list) == 0:
            raise RuntimeError("mask_list 为空：未找到任何掩码文件")

        # 统一用循环取模，保证 index 即使超出 mask_list 长度也能取到
        mask_index = index % len(self.mask_list)
        mask_path = self.mask_list[mask_index]

        # 读取并做尺寸适配
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if mask.shape[0] != imgh or mask.shape[1] != imgw:
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 127).astype(np.uint8) * 255
        return mask


    def to_tensor(self, img, norm=False):
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def load_edge(self, img):
        return canny(img, sigma=2, mask=None).astype(np.float32)

    ##-251104-## 舍弃原来的idx按顺序匹配，改为selected_img_name按名称匹配
    def load_wireframe(self, selected_basename, size):
        """
        参数:
            selected_basename: 图片名的 basename（不带路径、不带后缀），例如 "frame_001"
            size: 输出 lmap 的边长（int）
        返回:
            lmap: numpy.float32, shape (size, size)
        逻辑:
            - 在 self.line_index 中按 basename 精确匹配，获得 candidate 索引列表
            - 若多个 candidate，优先匹配 seq_id 相同的项（如果 __init__ 中生成了 seq_id），否则取第一个
            - 载入对应 .pkl（兼容 numpy 版本差异的反序列化）
            - 把 wf['lines'] -> rasterize 成 lmap（保持原来的坐标/归一化/交换 x<->y 逻辑）
        """
        import os, io, pickle as pkl
        import numpy as np
        import skimage.draw

        # 1) 基本校验
        base = selected_basename
        if base is None:
            raise FileNotFoundError(f"没有找到 {selected_basename}")  # ADDED

        # 简单缓存，避免重复解析同一 pkl（需在 __init__ 中初始化 self._lmap_cache = {}） 
        if hasattr(self, '_lmap_cache') and base in self._lmap_cache:  # ADDED
            return self._lmap_cache[base]  # ADDED

        # 2) 找到 candidate pkl 路径（只按第三维 basename 精确匹配）
        line_pkl_path = None
        chosen_idx = None

        # 优先使用预构建的索引 self.line_index（basename -> [indices]）
        if hasattr(self, 'line_index') and base in self.line_index:
            cand_idxs = self.line_index[base]
            # 如果只有一个候选直接用
            if len(cand_idxs) == 1:
                chosen_idx = cand_idxs[0]
            else:
                # 尝试按 seq_id 优先匹配（如果传入了 seq_hint 到此处可做更细策略）
                # 这里尽量利用 self.imgs_dict / __getitem__ 传入的 seq 信息（若你想）
                # 简单策略：先尝试匹配同 seq_id（如果 self.__getitem__ 提供了 seq_hint，则可改造接口）
                # 默认退回到第一个候选
                chosen_idx = cand_idxs[0]
            # 从 line_dict 取出路径
            if chosen_idx is not None:
                seq_id_line, pth, bname = self.line_dict[chosen_idx]  # self.line_dict: list of (seq_id, path, basename)
                line_pkl_path = pth

        # 兼容旧接口：若 self.line_map 存在（basename -> path）
        if line_pkl_path is None and hasattr(self, 'line_map') and base in self.line_map:
            line_pkl_path = self.line_map[base]

        # 后备：如果有 self.line_path（目录），做一次 glob 模糊查找（尽量避免频繁使用）
        if line_pkl_path is None and hasattr(self, 'line_path') and self.line_path:
            import glob
            cand = glob.glob(os.path.join(self.line_path, f"*{base}*.pkl"))
            if cand:
                line_pkl_path = cand[0]

        # 未找到 -> 报错（并给出少量示例便于调试）
        if line_pkl_path is None:
            sample = []
            if hasattr(self, 'line_index'):
                sample = list(self.line_index.keys())[:10]
            elif hasattr(self, 'line_dict'):
                sample = [os.path.splitext(os.path.basename(p))[0] for (_, p, _) in self.line_dict[:10]]
            raise FileNotFoundError(f"没有找到 {selected_basename}。示例可用 names（前10）：{sample}")

        # 3) 载入 pkl（兼容 numpy 反序列化差异）
        def _load_pickle_compat(path):
            data = open(path, 'rb').read()
            class NPCompatUnpickler(pkl.Unpickler):
                def find_class(self, module, name):
                    if module.startswith('numpy._core'):
                        module = module.replace('numpy._core', 'numpy.core')
                    return super().find_class(module, name)
            return NPCompatUnpickler(io.BytesIO(data)).load()

        wf = _load_pickle_compat(line_pkl_path)

        # 4) 生成 lmap（保留你先前实现的像素化逻辑）
        lmap = np.zeros((size, size), dtype=np.float32)
        lines_arr = wf.get('lines', None)
        if lines_arr is None:
            # 缓存空结果也有意义
            if hasattr(self, '_lmap_cache'):
                self._lmap_cache[base] = lmap  # ADDED
            return lmap

        scores = wf.get('scores', None)

        def _clamp_int(v, low, high):
            iv = int(round(float(v)))
            if iv < low: iv = low
            if iv > high: iv = high
            return iv

        for i, line_item in enumerate(lines_arr):
            score = scores[i] if (scores is not None and i < len(scores)) else 1.0
            if score <= getattr(self, 'wireframe_th', 0.0):
                continue

            xy = np.asarray(line_item, dtype=np.float32).copy()
            if xy.size < 4:
                continue

            # 若为归一化坐标（0~1），则放大
            if np.max(np.abs(xy[:4])) <= 1.5:
                xy[:4] = xy[:4] * float(size)

            x1_raw, y1_raw, x2_raw, y2_raw = float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])
            # 保持你原来的 x<->y 交换逻辑（若需要改可在此调整）
            x1, y1 = y1_raw, x1_raw
            x2, y2 = y2_raw, x2_raw

            r0 = _clamp_int(y1, 0, size - 1)
            c0 = _clamp_int(x1, 0, size - 1)
            r1 = _clamp_int(y2, 0, size - 1)
            c1 = _clamp_int(x2, 0, size - 1)

            if r0 == r1 and c0 == c1:
                lmap[r0, c0] = max(lmap[r0, c0], 1.0)
                continue

            rr, cc, val = skimage.draw.line_aa(r0, c0, r1, c1)
            valid = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
            if np.any(valid):
                lmap[rr[valid], cc[valid]] = np.maximum(lmap[rr[valid], cc[valid]], val[valid])

        # 5) 缓存并返回
        if hasattr(self, '_lmap_cache'):
            self._lmap_cache[base] = lmap  # ADDED
        return lmap


    def __getitem__(self, idx):
        import os, glob, random

        # --------- 从 imgs_dict 读取（第三维为 basename） ---------
        # imgs_dict 每项格式：(seq_id, img_path, basename)
        seq_id_img, selected_img_path, selected_basename = self.imgs_dict[idx]

        # ------------- 读取图片 -------------
        img = cv2.imread(selected_img_path)
        # 如果读图失败，随机重采（保持原来行为，但保证 selected_* 一致更新）
        while img is None:
            print('Bad image {}...'.format(selected_img_path))  # 打印路径更利于定位
            idx = random.randint(0, len(self.imgs_dict) - 1)
            seq_id_img, selected_img_path, selected_basename = self.imgs_dict[idx]
            img = cv2.imread(selected_img_path)
        img = img[:, :, ::-1]  # BGR -> RGB

        # ------------- 预处理（resize / edge / line / mask） -------------
        img = self.resize(img, self.image_size, self.image_size, center_crop=False)
        img_gray = rgb2gray(img)
        edge = self.load_edge(img_gray)

        # load_wireframe 现在以 basename（第三维）作为输入（函数需与此签名一致）
        # selected_basename 是不带路径、不带后缀的名称（例如 "frame_001"）
        line = self.load_wireframe(selected_basename, self.image_size)

        # mask 使用原有逻辑（按 index 取模），这里继续使用 idx 以保持原行为
        mask = self.load_mask(img, idx)

        # ------------- 转为 tensor（保持原逻辑） -------------
        img_t = self.to_tensor(img, norm=True)
        edge_t = self.to_tensor(edge)
        line_t = self.to_tensor(line)
        mask_t = self.to_tensor(mask)

        # --------- 最小调试信息：路径映射（安全且不打印） ---------
        img_path = selected_img_path  # 使用图片绝对路径（更直观）
        # mask_path：从 self.mask_list 中取 （循环取模），若不存在则 None
        mask_path = None
        if hasattr(self, 'mask_list') and len(self.mask_list) > 0:
            try:
                mask_path = self.mask_list[idx % len(self.mask_list)]
            except Exception:
                mask_path = None

        # --------- 解析对应的 line_pkl_path（从 line_index / line_dict 中取） ---------
        line_pkl_path = None
        try:
            # 优先使用已构建好的反向索引 self.line_index (basename -> [indices])
            if hasattr(self, 'line_index') and selected_basename in self.line_index:
                cand_idxs = self.line_index[selected_basename]
                # 选择第一个候选（若需更复杂的优先级，可在此处实现）
                chosen_idx = cand_idxs[0]
                # self.line_dict 是 list of tuples (seq_id, path, basename) as in __init__
                seq_id_line, pth, bname = self.line_dict[chosen_idx]
                line_pkl_path = pth
            # 否则尝试兼容旧接口 self.line_map（basename -> first path）
            elif hasattr(self, 'line_map') and selected_basename in self.line_map:
                line_pkl_path = self.line_map[selected_basename]
            else:
                # 退回到在 self.line_path 目录里模糊匹配（保持原有后备机制）
                if hasattr(self, 'line_path') and self.line_path is not None and os.path.isdir(self.line_path):
                    cand = glob.glob(os.path.join(self.line_path, f"*{selected_basename}*.pkl"))
                    if len(cand) > 0:
                        line_pkl_path = cand[0]
        except Exception:
            line_pkl_path = None

        # edge_path：如果你在 __init__ 建了 edge_dict，则返回对应路径，否则 None（保留原逻辑）
        edge_path = None
        try:
            if hasattr(self, 'edge_dict') and selected_basename in self.edge_dict:
                edge_path = self.edge_dict[selected_basename]
        except Exception:
            edge_path = None

        # --------- erode mask（和原代码一致） ---------
        erode = mask
        while True:
            if random.random() > 0.5:
                erode = self.to_tensor(erode)
                break
            k_size = random.randint(5, 25)
            erode2 = cv2.erode(erode // 255, np.ones((k_size, k_size), np.uint8), iterations=1)
            if np.sum(erode2) > 0:
                erode = self.to_tensor(erode2 * 255)
                break

        # --------- 组合 meta 并返回（保持字段） ---------
        mask_img = img_t * (1 - mask_t)

        meta = {
            'img': img_t,
            'mask_img': mask_img,
            'mask': mask_t,
            'erode_mask': erode,
            'edge': edge_t,
            'line': line_t,
            'name': selected_basename,           # 仅文件名（无后缀），与第三维保持一致
            'img_path': img_path,                # 图片实际路径
            'mask_path': mask_path,
            'line_pkl_path': line_pkl_path,      # 便于调试/日志
            # 'edge_path': edge_path
        }

        return meta




class ContinuousEdgeLineDatasetMaskFinetune(ContinuousEdgeLineDatasetMask):
    """
    Finetune 数据集（MaP 阶段）：
    - 完全复用 ContinuousEdgeLineDatasetMask 的文件组织、wireframe 匹配、序列划分与工具函数
    - 仅在 __getitem__ 中收敛 mask 策略：使用给定的 mask 列表（index 对齐，取模循环），不做随机 mask_rate
    - 返回字段与父类保持一致，并额外提供 seq_id（若父类在 __init__ 中已构建 index_to_seq）
    """

    def __init__(self,
                 pt_dataset,
                 mask_path=None,
                 test_mask_path=None,
                 is_train=False,
                 mask_rates=None,
                 image_size=256,
                 line_path=None,
                 wireframes_list=None):
        """
        兼容你原来的调用签名；父类不需要 wireframes_list，这里接收后不向上游传递即可。
        其它参数保持与父类一致，全部交给 super() 构造，确保：
          - image_id_list / mask_list / test_mask_list
          - imgs_dict / line_dict / line_index / img_index
          - seq_to_indices / index_to_seq / sequences
          - load_* / resize / to_tensor 等方法
        """
        super().__init__(pt_dataset=pt_dataset,
                         mask_path=mask_path,
                         test_mask_path=test_mask_path,
                         is_train=is_train,
                         mask_rates=mask_rates,
                         image_size=image_size,
                         line_path=line_path)

    def __getitem__(self, idx):
        """
        与父类新版实现保持一致的流程：
          1) 从 imgs_dict 读取 (seq_id, img_path, basename)
          2) 读取与 resize 图像、转灰度，计算 edge
          3) 以 basename 精确匹配 wireframe（父类的新版 load_wireframe 已按 basename 匹配）
          4) mask：优先从 self.mask_list 取（按 idx 对齐，取模循环），没有则使用全 0 mask
          5) 可选翻转增强（与父类保持一致）
          6) 张量化，输出字段与父类一致，并补充 seq_id
        """
        # --------- 读取 (seq_id, img_path, basename) ----------
        seq_id_img, selected_img_path, selected_basename = self.imgs_dict[idx]

        # --------- 读图 + 容错 ----------
        img = cv2.imread(selected_img_path)
        while img is None:
            print('Bad image {}...'.format(selected_img_path))
            idx = random.randint(0, len(self.imgs_dict) - 1)
            seq_id_img, selected_img_path, selected_basename = self.imgs_dict[idx]
            img = cv2.imread(selected_img_path)
        img = img[:, :, ::-1]  # BGR -> RGB

        # --------- 预处理 ----------
        img = self.resize(img, self.image_size, self.image_size, center_crop=False)
        img_gray = rgb2gray(img)
        edge = self.load_edge(img_gray)

        # 以 basename 精确匹配 wireframe（父类新版签名：load_wireframe(selected_basename, size)）
        line = self.load_wireframe(selected_basename, self.image_size)

        # --------- Finetune 的 mask 策略 ----------
        if hasattr(self, 'mask_list') and self.mask_list is not None and len(self.mask_list) > 0:
            mask_path = self.mask_list[idx % len(self.mask_list)]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            if mask.shape[:2] != (self.image_size, self.image_size):
                mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
        else:
            # 未提供 mask 列表则退化为全 0 mask（完整监督）
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # --------- erode mask（与父类一致） ----------
        erode = mask
        while True:
            if random.random() > 0.5:
                erode_t = self.to_tensor(erode)
                break
            k_size = random.randint(5, 25)
            erode2 = cv2.erode(erode // 255, np.ones((k_size, k_size), np.uint8), iterations=1)
            if np.sum(erode2) > 0:
                erode_t = self.to_tensor(erode2 * 255)
                break

        # --------- 张量化 ----------
        img_t  = self.to_tensor(img, norm=True)
        edge_t = self.to_tensor(edge)
        line_t = self.to_tensor(line)
        mask_t = self.to_tensor(mask)
        mask_img = img_t * (1 - mask_t)

        # 便于调试：记录映射路径
        mask_path_dbg = None
        if hasattr(self, 'mask_list') and len(self.mask_list) > 0:
            mask_path_dbg = self.mask_list[idx % len(self.mask_list)]

        # --------- 组装返回 ----------
        meta = {
            'img': img_t,
            'mask_img': mask_img,
            'mask': mask_t,
            'erode_mask': erode_t,
            'edge': edge_t,
            'line': line_t,
            'name': selected_basename,          # 与 wireframe basename 一致
            'img_path': selected_img_path,      # 真实图像路径
            'mask_path': mask_path_dbg,         # 真实 mask 路径（便于日志）
        }

        # 若父类在 __init__ 中已构建 index_to_seq，则暴露 seq_id，支持“按序列训练”
        if hasattr(self, "index_to_seq"):
            meta["seq_id"] = self.index_to_seq[idx]
        else:
            meta["seq_id"] = seq_id_img

        # 也可在此补充 line_pkl_path（同父类 __getitem__ 的做法），按需解开注释：
        # try:
        #     line_pkl_path = None
        #     if hasattr(self, 'line_index') and selected_basename in self.line_index:
        #         cand_idxs = self.line_index[selected_basename]
        #         chosen_idx = cand_idxs[0]
        #         _, pth, _ = self.line_dict[chosen_idx]
        #         line_pkl_path = pth
        #     elif hasattr(self, 'line_map') and selected_basename in self.line_map:
        #         line_pkl_path = self.line_map[selected_basename]
        #     meta['line_pkl_path'] = line_pkl_path
        # except Exception:
        #     pass

        return meta
