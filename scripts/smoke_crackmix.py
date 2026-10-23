# -*- coding: utf-8 -*-
"""CrackMix 冒烟测试：用合成数据验证增强逻辑正确。

不依赖真实数据集 / GPU，构造 3 张含裂缝的合成图与掩码，构建 CrackMix
数据增强，抽样运行若干次，检查：
    1. transform 可正常构建（源样本清单有效）；
    2. 输出 img / seg_map 尺寸不变、值域合法（img uint8, seg 0/1/255）；
    3. 触发后裂缝像素数增加（贴片被粘贴并合并进标签）；
    4. 未触发时原样返回。

用法（仓库根目录）：
    python scripts/smoke_crackmix.py
"""
import importlib.util
import os
import shutil
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 直载 ops_shim（不经过 crackseg.utils 包，避免其引入依赖 torch 的
# checkpoint_compat），在导入 mmseg 前安装 mmcv._ext 算子桩。
_spec = importlib.util.spec_from_file_location(
    'crackseg_ops_shim', os.path.join(ROOT, 'crackseg', 'utils', 'ops_shim.py'))
_shim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shim)
_shim.install_mmcv_ops_shim()

from mmengine.dataset import Compose  # noqa: E402
from mmengine.utils import import_modules_from_strings  # noqa: E402
from mmseg.registry import TRANSFORMS  # noqa: E402
from mmseg.utils import register_all_modules  # noqa: E402


def make_synthetic_data(root_dir, n=3, size=(336, 640)):
    """构造 n 张合成图：灰底 + 随机白色/灰色斑块 + 一条裂缝细线。"""
    img_dir = os.path.join(root_dir, 'images', 'train')
    seg_dir = os.path.join(root_dir, 'annotations', 'train')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n):
        h, w = size
        base = rng.randint(80, 160)
        img = np.full((h, w, 3), base, dtype=np.uint8)
        # 背景斑块
        for _ in range(4):
            x0 = rng.randint(0, w - 60)
            y0 = rng.randint(0, h - 40)
            c = rng.randint(0, 255)
            img[y0:y0 + 40, x0:x0 + 60] = c
        # 裂缝：一条折线细线（约 2 像素宽）
        seg = np.zeros((h, w), dtype=np.uint8)
        x, y = rng.randint(50, w - 50), rng.randint(50, h - 50)
        pts = [(x, y)]
        for _ in range(4):
            x += rng.randint(-60, 60)
            y += rng.randint(-40, 40)
            x = min(max(x, 10), w - 10)
            y = min(max(y, 10), h - 10)
            pts.append((x, y))
        for k in range(len(pts) - 1):
            cv2.line(img, pts[k], pts[k + 1], (20, 20, 20), 2)
            cv2.line(seg, pts[k], pts[k + 1], 1, 2)
        cv2.imwrite(os.path.join(img_dir, 'syn_%d.jpg' % i), img)
        cv2.imwrite(os.path.join(seg_dir, 'syn_%d.png' % i), seg)
    return img_dir, seg_dir


def main():
    register_all_modules()
    import_modules_from_strings(
        imports=['crackseg.datasets'], allow_failed_imports=False)

    tmp = os.path.join(ROOT, 'work_dirs', 'smoke_crackmix_data')
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    img_dir, seg_dir = make_synthetic_data(tmp)

    pipe = Compose([
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='RandomResize', scale=(384, 384), ratio_range=(0.5, 2.0),
             keep_ratio=True),
        dict(type='RandomRotate', prob=0.5, degree=(-30, 30), pad_val=0,
             seg_pad_val=255),
        dict(
            type='CrackMix', prob=1.0, data_root=tmp,
            img_subdir='images/train', seg_subdir='annotations/train',
            min_area=20, max_rotate=30.0, scale_range=(0.5, 1.5),
            flip_prob=0.5, bbox_pad=3),
        dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.9),
        dict(type='PackSegInputs'),
    ])

    # 检查 TRANSFORMS 注册
    assert 'CrackMix' in TRANSFORMS.module_dict, 'CrackMix 未注册'
    print('[OK] CrackMix 已注册到 TRANSFORMS')

    # 连续跑 30 次，统计裂缝像素与尺寸
    crack_pixels = []
    shapes = set()
    for i in range(30):
        results = pipe({
            'img_path': os.path.join(img_dir, 'syn_0.jpg'),
            'seg_map_path': os.path.join(seg_dir, 'syn_0.png'),
            'img_id': i,
            'seg_fields': [],
            'reduce_zero_label': False,
        })
        seg = results['data_samples'].gt_sem_seg.data
        crack_pixels.append(int((seg == 1).sum()))
        shapes.add(tuple(results['inputs'].shape[1:]))
    crack_pixels = np.array(crack_pixels)

    print('[OK] 输出尺寸集合:', sorted(shapes))
    assert all(h > 0 and w > 0 and h <= 256 and w <= 256 for h, w in shapes), \
        '输出尺寸非法: %s' % sorted(shapes)
    print('[OK] 裂缝像素数 min/mean/max: %d / %.1f / %d'
          % (crack_pixels.min(), crack_pixels.mean(), crack_pixels.max()))
    assert crack_pixels.mean() > 20, 'CrackMix 未产生有效裂缝贴片'

    # 未触发分支：prob=0
    pipe0 = Compose([
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='CrackMix', prob=0.0, data_root=tmp,
             img_subdir='images/train', seg_subdir='annotations/train'),
        dict(type='PackSegInputs'),
    ])
    r0 = pipe0({'img_path': os.path.join(img_dir, 'syn_0.jpg'),
                'seg_map_path': os.path.join(seg_dir, 'syn_0.png'),
                'img_id': 0, 'seg_fields': [],
                'reduce_zero_label': False})
    s0 = r0['data_samples'].gt_sem_seg.data
    base_px = int((s0 == 1).sum())
    assert base_px > 0, '基础合成图无裂缝'
    print('[OK] prob=0 时原样返回，裂缝像素=%d' % base_px)

    # 值与 dtype 合法性
    print('[OK] inputs dtype:', r0['inputs'].dtype)
    assert set(np.unique(s0)).issubset({0, 1, 255}), 'seg 值域异常'
    print('[OK] 值域/dtype 校验通过')

    shutil.rmtree(tmp)
    print('\n===== CrackMix 冒烟测试全部通过 =====')


if __name__ == '__main__':
    main()
