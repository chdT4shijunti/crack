# -*- coding: utf-8 -*-
"""预训练 ResNet34 + UNet + CrackMix：UAV 裂缝分割（384×384）。

在 ``crack_seg_preunet.py``（ImageNet 预训练 ResNet34 编码器 + Focal-Tversky
+ 3 辅助头 + AdamW + 滑窗推理）基础上，训练管线新增 **CrackMix** 数据增强，
与 ``crack_seg_improved_crackmix.py`` 完全同构。

预期：预训练编码器提供通用特征，CrackMix 在线扩充裂缝形态多样性，
两者叠加通常可在验证集 mIoU 上再获稳定增益（待训练验证）。

用法（仓库根目录）：
    python tools/train.py configs/crack_seg_preunet_crackmix.py \
        --work-dir work_dirs/preunet_crackmix --amp
"""
# 导入并注册自定义模型与自定义数据增强
custom_imports = dict(
    imports=['crackseg.models', 'crackseg.datasets'],
    allow_failed_imports=False)

_base_ = ['./crack_seg_preunet.py']

crop_size = (384, 384)

# ---- 训练管线：在 preunet 管线中插入 CrackMix ----
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
