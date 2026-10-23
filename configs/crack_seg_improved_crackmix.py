# -*- coding: utf-8 -*-
"""改进配置 + CrackMix 增强：UAV 裂缝分割（384×384，ECA-UNet）。

在 ``crack_seg_improved.py``（Focal-Tversky + 3 辅助头 + 384 + ±30° 旋转）
基础上，训练管线新增 **CrackMix** 数据增强：
    从训练集随机取另一张样本，提取其裂缝连通分量作为贴片，经随机旋转 /
    缩放 / 翻转后粘贴到当前样本，同时合并标签，在线扩充裂缝形态与背景
    组合，提升小样本裂缝分割的泛化能力。

管线顺序说明：
    CrackMix 放在 RandomRotate 之后（整体旋转会带动贴片，方向更自然）、
    RandomCrop 之前（粘贴越界部分由裁剪统一处理）、PhotoMetricDistortion
    之前（亮度/颜色扰动会同时作用于贴片，色调过渡更平滑）。

用法（仓库根目录）：
    python tools/train.py configs/crack_seg_improved_crackmix.py \
        --work-dir work_dirs/improved_crackmix --amp
"""
# 导入并注册自定义模型与自定义数据增强
custom_imports = dict(
    imports=['crackseg.models', 'crackseg.datasets'],
    allow_failed_imports=False)

_base_ = ['./crack_seg_improved.py']

crop_size = (384, 384)

# ---- 训练管线：在 improved 管线中插入 CrackMix ----
crackmix_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(384, 384),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomRotate', prob=0.5, degree=(-30, 30), pad_val=0,
         seg_pad_val=255),
    dict(
        type='CrackMix',
        prob=0.5,
        data_root='data/Crack',
        img_subdir='images/train',
        seg_subdir='annotations/train',
        min_area=60,
        max_rotate=30.0,
        scale_range=(0.5, 1.5),
        flip_prob=0.5,
        bbox_pad=3),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    dataset=dict(pipeline=crackmix_train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader
