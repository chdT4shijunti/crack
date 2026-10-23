# -*- coding: utf-8 -*-
"""临时探针：确认 efficientnet 编码器特征级/通道/预训练URL，适配 TimmUNet。"""
import os
os.environ.setdefault('HF_HUB_DISABLE', '1')
import timm
for name in ['efficientnet_b3', 'efficientnet_b4', 'tf_efficientnet_b3']:
    try:
        m = timm.create_model(name, pretrained=False, features_only=True,
                              out_indices=(0, 1, 2, 3, 4))
        fi = m.feature_info
        print('==', name)
        print('  reductions:', fi.reduction())
        print('  channels  :', fi.channels())
        cfg = timm.get_pretrained_cfg(name)
        print('  url       :', getattr(cfg, 'url', None))
    except Exception as e:
        print('==', name, 'ERROR:', repr(e))
