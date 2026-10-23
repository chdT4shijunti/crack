# -*- coding: utf-8 -*-
"""crackseg.datasets：自定义数据增强。

导入即注册到 MMSegmentation（``TRANSFORMS``）。
"""
from .crack_mix import CrackMix

__all__ = ['CrackMix']
