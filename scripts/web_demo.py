# -*- coding: utf-8 -*-
"""Gradio 网页 Demo：上传无人机航拍路面图，自动识别并叠加裂缝。

运行方式（仓库根目录，激活 uav_crack 环境）：
    python scripts/web_demo.py
然后浏览器打开终端显示的地址（默认 http://127.0.0.1:7860）。

功能：
    - 输入：任意尺寸的路面照片（无人机航拍 / 手机拍摄均可）
    - 输出：
        1. 原图 + 红色半透明裂缝叠加（便于人眼检查）
        2. 纯裂缝二值掩码（白=裂缝，黑=背景）

默认加载验证集最强单模 EfficientNetV2-S + TimmUNet
（work_dirs/preunet_v2s/best_mIoU_iter_14000.pth，验证 mIoU 79.93）。
想换成其它模型，改下方 CONFIG / WEIGHTS 两个常量即可。
"""
import importlib.util
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# --- 在 import mmseg 之前安装 mmcv 算子桩（本机 mmcv-lite 环境需要） ---
_spec = importlib.util.spec_from_file_location(
    'crackseg_ops_shim', os.path.join(ROOT, 'crackseg', 'utils', 'ops_shim.py'))
_shim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shim)
_shim.install_mmcv_ops_shim()

import torch  # noqa: E402
# PyTorch>=2.6 默认 weights_only=True 会拦截 mmengine checkpoint 里的
# HistoryBuffer / numpy 内部对象；权重为本仓库自训，此处关闭该限制。
_orig_torch_load = torch.load
def _torch_load_wo_false(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_torch_load(*a, **kw)
torch.load = _torch_load_wo_false  # noqa: E402
from mmseg.apis import init_model, inference_model  # noqa: E402

# ======================= 模型配置（可切换其它成员） =======================
CONFIG = os.path.join(ROOT, 'configs', 'crack_seg_preunet_v2s.py')
WEIGHTS = os.path.join(
    ROOT, 'work_dirs', 'preunet_v2s', 'best_mIoU_iter_14000.pth')
# =========================================================================

# 输入图片最长边限制：避免大图滑窗推理爆显存 / 过慢
MAX_LONG_EDGE = 1280
# 裂缝叠加色（RGB）
CRACK_COLOR = (255, 40, 40)
# 叠加透明度
OVERLAY_ALPHA = 0.55

_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print('[WebDemo] 加载模型: %s' % os.path.basename(WEIGHTS))
print('[WebDemo] 设备: %s' % _DEVICE)
_MODEL = init_model(CONFIG, WEIGHTS, device=_DEVICE)
_MODEL.eval()
print('[WebDemo] 模型加载完成')


def _resize_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    """等比例缩放，使最长边 = long_edge（不放大原图）。"""
    h, w = img.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_LINEAR)


def predict(image_rgb: np.ndarray):
    """Gradio 回调：输入 RGB ndarray，返回 (叠加图 RGB, 二值掩码 RGB)。"""
    if image_rgb is None:
        raise gr.Error('请先上传一张路面图片')

    h0, w0 = image_rgb.shape[:2]

    # 1) 缩放长边，控制显存与耗时
    small_rgb = _resize_long_edge(image_rgb, MAX_LONG_EDGE)
    # mmseg inference_model 期望 BGR ndarray
    bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)

    # 2) 推理（模型内部按 test_cfg 滑窗分割）
    with torch.no_grad():
        result = inference_model(_MODEL, bgr)
    pred = result.pred_sem_seg.data[0].cpu().numpy()  # HxW, 0=bg 1=crack

    # 3) 掩码放回原始分辨率（最近邻保持边缘硬）
    if pred.shape != (h0, w0):
        pred = cv2.resize(pred.astype(np.uint8), (w0, h0),
                          interpolation=cv2.INTER_NEAREST)

    # 4) 叠加可视化：原图 + 红色半透明裂缝
    overlay = image_rgb.copy()
    mask = pred.astype(bool)
    overlay[mask] = (
        overlay[mask] * (1 - OVERLAY_ALPHA)
        + np.array(CRACK_COLOR) * OVERLAY_ALPHA
    ).astype(np.uint8)

    # 5) 纯掩码：白裂缝 / 黑背景
    mask_rgb = np.zeros((h0, w0, 3), dtype=np.uint8)
    mask_rgb[mask] = (255, 255, 255)

    return overlay, mask_rgb


# --- Gradio 界面 ---
import gradio as gr  # noqa: E402

EXAMPLES_DIR = os.path.join(ROOT, 'data', 'Crack', 'images', 'val')
_examples = []
if os.path.isdir(EXAMPLES_DIR):
    for fn in sorted(os.listdir(EXAMPLES_DIR))[:4]:
        if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            _examples.append([os.path.join(EXAMPLES_DIR, fn)])

with gr.Blocks(title='无人机裂缝智能识别系统') as demo:
    gr.Markdown(
        '# 🛣️ 无人机裂缝智能识别系统\n'
        '上传一张航拍照片（路面、墙体、建筑缝隙等），模型自动分割裂缝并以红色标注输出。\n'
        '后端模型：EfficientNetV2-S + UNet 语义分割网络。'
    )
    with gr.Row():
        inp = gr.Image(type='numpy', label='输入：路面图片')
        out_overlay = gr.Image(type='numpy', label='输出：裂缝叠加识别图')
    out_mask = gr.Image(type='numpy', label='输出：裂缝二值掩码（白=裂缝）')
    btn = gr.Button('开始识别', variant='primary')
    btn.click(fn=predict, inputs=inp, outputs=[out_overlay, out_mask])
    if _examples:
        gr.Examples(examples=_examples, inputs=inp)

if __name__ == '__main__':
    import socket
    # 取本机局域网 IPv4，方便同一 WiFi / 局域网内其它电脑直接访问
    lan_ip = '127.0.0.1'
    try:
        _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _s.connect(('8.8.8.8', 80))
        lan_ip = _s.getsockname()[0]
        _s.close()
    except OSError:
        pass
    print('-' * 60)
    print('本机访问:        http://127.0.0.1:7860')
    print('局域网其它电脑:  http://%s:7860   （同一 WiFi / 局域网）' % lan_ip)
    print('-' * 60)
    demo.launch(server_name='0.0.0.0', server_port=7860, inbrowser=True)
