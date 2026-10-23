# -*- coding: utf-8 -*-
"""CrackMix 数据增强（面向细长裂缝的 Copy-Paste 式增强）。

动机：
    裂缝像素在图中占比极小、形态细长且方向多样，小样本训练下模型容易
    “记住”训练集里有限的裂缝形态，泛化受限。CrackMix 的做法是从训练集
    随机取另一张样本，提取其裂缝区域（连通分量）作为“贴片”，经随机
    旋转 / 缩放 / 翻转后粘贴到当前样本，同时把贴片裂缝合并进当前标签，
    从而在线扩充裂缝的形态、方向与背景组合，提升分割泛化能力。

实现要点：
    1. 完全自包含：通过 ``data_root / img_subdir / seg_subdir`` 在初始化时
       构建源样本清单（只存路径字符串，可随 Dataset 安全序列化进 worker），
       每次触发时随机读一张源图及其掩码；
    2. 贴片按连通分量抽取（8 邻接），并过滤过小的分量，避免粘贴无意义噪声；
    3. 贴片做随机旋转 + 缩放 + 翻转，使粘贴的裂缝朝向、尺度更自然；
    4. 掩码合并时保留目标样本的“忽略”像素（255），不把忽略区改写为裂缝。

用法（配置示例，插在 RandomRotate 之后、RandomCrop 之前）：
    dict(type='CrackMix', prob=0.5,
         data_root='data/Crack',
         img_subdir='images/train',
         seg_subdir='annotations/train',
         min_area=60, max_rotate=30.0,
         scale_range=(0.5, 1.5), flip_prob=0.5)
"""
import glob
import os
import random

import cv2
import numpy as np

from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CrackMix(BaseTransform):
    """CrackMix 数据增强。

    Args:
        prob (float): 触发概率，默认 0.5。
        data_root (str): 源样本根目录。
        img_subdir (str): 源图像子目录（相对 data_root，jpg）。
        seg_subdir (str): 源掩码子目录（相对 data_root，png，取值 0/1）。
        min_area (int): 参与粘贴的裂缝连通分量最小像素数，默认 60。
        max_rotate (float): 贴片最大随机旋转角度（度），默认 30.0。
        scale_range (tuple[float]): 贴片随机缩放范围，默认 (0.5, 1.5)。
        flip_prob (float): 贴片随机水平/垂直翻转概率，默认 0.5。
        bbox_pad (int): 裁剪裂缝分量时的外扩像素，默认 3。
    """

    def __init__(self,
                 prob=0.5,
                 data_root=None,
                 img_subdir=None,
                 seg_subdir=None,
                 min_area=60,
                 max_rotate=30.0,
                 scale_range=(0.5, 1.5),
                 flip_prob=0.5,
                 bbox_pad=3):
        super().__init__()
        assert data_root and img_subdir and seg_subdir, \
            'CrackMix 需要 data_root / img_subdir / seg_subdir'
        self.prob = float(prob)
        self.min_area = int(min_area)
        self.max_rotate = float(max_rotate)
        self.scale_range = tuple(scale_range)
        self.flip_prob = float(flip_prob)
        self.bbox_pad = int(bbox_pad)
        self._src_list = self._build_src_list(
            data_root, img_subdir, seg_subdir)

    def _build_src_list(self, data_root, img_subdir, seg_subdir):
        """构建源样本 (img_path, seg_path) 清单。"""
        img_dir = os.path.join(data_root, img_subdir)
        seg_dir = os.path.join(data_root, seg_subdir)
        pairs = []
        for img_path in sorted(glob.glob(os.path.join(img_dir, '*.jpg'))):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            seg_path = os.path.join(seg_dir, stem + '.png')
            if os.path.exists(seg_path):
                pairs.append((img_path, seg_path))
        if not pairs:
            raise FileNotFoundError(
                'CrackMix: 源样本目录无有效样本: %s' % img_dir)
        return pairs

    def _load_src(self):
        """随机加载一张源样本，返回 (img BGR uint8, seg 0/1 uint8)。"""
        img_path, seg_path = random.choice(self._src_list)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        if img is None or seg is None:
            return None, None
        if seg.max() > 1:
            seg = (seg > 0).astype(np.uint8)
        return img, seg

    def _pick_component(self, seg):
        """从掩码中随机选一个不小于 min_area 的裂缝连通分量。

        Returns:
            (x1, y1, x2, y2) 裁剪包围盒（含外扩），或 None。
        """
        fg = (seg == 1).astype(np.uint8)
        n, labels = cv2.connectedComponents(fg, connectivity=8)
        if n <= 1:
            return None
        comps = []
        for lab in range(1, n):
            ys, xs = np.where(labels == lab)
            if len(ys) < self.min_area:
                continue
            comps.append((len(ys), xs.min(), ys.min(), xs.max(), ys.max()))
        if not comps:
            return None
        _, x1, y1, x2, y2 = random.choice(comps)
        h, w = seg.shape[:2]
        x1 = max(0, x1 - self.bbox_pad)
        y1 = max(0, y1 - self.bbox_pad)
        x2 = min(w - 1, x2 + self.bbox_pad)
        y2 = min(h - 1, y2 + self.bbox_pad)
        return x1, y1, x2, y2

    def _transform_patch(self, patch_img, patch_seg):
        """对贴片做随机旋转 + 缩放 + 翻转，返回 (img, seg 0/1)。"""
        h, w = patch_img.shape[:2]
        scale = random.uniform(*self.scale_range)
        angle = random.uniform(-self.max_rotate, self.max_rotate)
        center = (w / 2.0, h / 2.0)
        m = cv2.getRotationMatrix2D(center, angle, scale)
        # 旋转后外接矩形尺寸，并平移旋转矩阵使内容居中
        cos_a = abs(m[0, 0])
        sin_a = abs(m[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        m[0, 2] += new_w / 2.0 - center[0]
        m[1, 2] += new_h / 2.0 - center[1]
        out_img = cv2.warpAffine(
            patch_img, m, (new_w, new_h), flags=cv2.INTER_LINEAR)
        out_seg = cv2.warpAffine(
            patch_seg, m, (new_w, new_h), flags=cv2.INTER_NEAREST)
        if random.random() < self.flip_prob:
            out_img = out_img[:, ::-1]
            out_seg = out_seg[:, ::-1]
        out_seg = (out_seg > 0).astype(np.uint8)
        return out_img, out_seg

    def transform(self, results):
        """把另一张训练图的裂缝贴片粘贴到当前样本。"""
        if random.random() > self.prob:
            return results
        img = results.get('img', None)
        seg = results.get('seg_map', None)
        if img is None or seg is None:
            return results

        src_img, src_seg = self._load_src()
        if src_img is None or src_seg is None:
            return results
        box = self._pick_component(src_seg)
        if box is None:
            return results

        x1, y1, x2, y2 = box
        p_img = src_img[y1:y2 + 1, x1:x2 + 1]
        p_seg = src_seg[y1:y2 + 1, x1:x2 + 1]
        p_img, p_seg = self._transform_patch(p_img, p_seg)

        ph, pw = p_seg.shape[:2]
        H, W = seg.shape[:2]
        if ph >= H or pw >= W:
            return results

        # 随机粘贴位置
        x0 = random.randint(0, W - pw)
        y0 = random.randint(0, H - ph)

        # 图像：贴片直接覆盖
        img[y0:y0 + ph, x0:x0 + pw] = p_img

        # 掩码：贴片裂缝并入；目标“忽略”(255) 像素保持忽略
        dst_seg = seg[y0:y0 + ph, x0:x0 + pw].copy()
        paste = (p_seg == 1)
        valid = (dst_seg != 255)
        seg[y0:y0 + ph, x0:x0 + pw] = np.where(
            paste & valid, 1, dst_seg)

        results['img'] = img
        results['seg_map'] = seg
        return results
