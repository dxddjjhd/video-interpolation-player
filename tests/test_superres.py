"""超分引擎单元测试：Lanczos 尺寸/守护降级/RealESRGAN 模型。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from player.superres.lanczos import LanczosSuperRes  # noqa: E402


def _test_frame(h=96, w=128):
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_lanczos_2x():
    sr = LanczosSuperRes(2)
    frame = _test_frame()
    out = sr.upscale(frame)
    assert out.shape == (192, 256, 3)
    assert out.dtype == np.uint8
    # 内容应接近放大（均值大致保持）
    assert abs(out.mean() - frame.mean()) < 15


def test_lanczos_4x():
    sr = LanczosSuperRes(4)
    out = sr.upscale(_test_frame())
    assert out.shape == (384, 512, 3)


def test_lanczos_odd_size():
    """奇数尺寸也应正常放大。"""
    sr = LanczosSuperRes(2)
    out = sr.upscale(_test_frame(95, 127))
    assert out.shape == (190, 254, 3)


# ------------------------------------------------------------------
def test_fsr_style_sizes():
    from player.superres.fsr_style import FsrStyleSuperRes
    for factor, (h, w) in [(2, (96, 128)), (4, (48, 64))]:
        sr = FsrStyleSuperRes(factor)
        out = sr.upscale(_test_frame(h, w))
        assert out.shape == (h * factor, w * factor, 3)
        assert out.dtype == np.uint8


def test_fsr_style_sharper_than_lanczos():
    """FSR 风格应比纯 Lanczos 更锐利（感知锐度 Tenengrad 度量）。

    注意：用中等频率的真实感测试图而非高频棋盘 —— 棋盘会让
    Lanczos 因振铃伪影虚高边缘能量（那恰是 Lanczos 的缺点）。
    """
    import cv2
    from player.superres.fsr_style import FsrStyleSuperRes

    # 深灰背景 + 白色方块 + 平滑渐变（中等频率边缘）
    rng = np.random.default_rng(7)
    base = np.full((128, 128, 3), 40, dtype=np.uint8)
    base[24:96, 24:96] = (200, 200, 220)
    base[40:80, 40:80] = (120, 120, 255)
    yy = np.linspace(0, 60, 128, dtype=np.uint8)
    base[:, :, 2] = np.clip(base[:, :, 2].astype(int) + yy[:, None], 0, 255)

    lz = LanczosSuperRes(2).upscale(base)
    fsr = FsrStyleSuperRes(2).upscale(base)
    assert lz.shape == fsr.shape

    def tenengrad(img):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return float((gx * gx + gy * gy).mean())

    assert tenengrad(fsr) > tenengrad(lz) * 1.05, \
        f"FSR 风格锐度应更高: {tenengrad(fsr):.1f} vs {tenengrad(lz):.1f}"


def test_fsr_style_performance():
    """720p 2x 应实时（< 40ms/帧，共享 vCPU 上留余量）。"""
    import time
    from player.superres.fsr_style import FsrStyleSuperRes

    sr = FsrStyleSuperRes(2)
    frame = _test_frame(720, 1280)
    sr.upscale(frame)   # 预热
    t0 = time.perf_counter()
    n = 5
    for _ in range(n):
        sr.upscale(frame)
    per = (time.perf_counter() - t0) / n * 1000
    print(f"  FSR 720p→1440p: {per:.1f}ms/帧")
    assert per < 40, f"FSR 风格性能不达标: {per:.1f}ms"


# ------------------------------------------------------------------
MODELS = ROOT / "player/superres/models"


@pytest.mark.skipif(not (MODELS / "RealESRGANv2-animevideo-xsx2.onnx").exists(),
                    reason="RealESRGAN 2x 模型未就绪")
def test_realesrgan_2x_model():
    from player.superres.real_esrgan import RealEsrganSuperRes
    sr = RealEsrganSuperRes(
        str(MODELS / "RealESRGANv2-animevideo-xsx2.onnx"), 2, use_cuda=False)
    try:
        frame = _test_frame(64, 80)
        out = sr.upscale(frame)
        assert out.shape == (128, 160, 3)
        assert out.dtype == np.uint8
        # 内容合理：与输入均值差异不大
        assert abs(float(out.mean()) - float(frame.mean())) < 30
    finally:
        sr.close()


@pytest.mark.skipif(not (MODELS / "RealESRGANv2-animevideo-xsx4.onnx").exists(),
                    reason="RealESRGAN 4x 模型未就绪")
def test_realesrgan_4x_model():
    from player.superres.real_esrgan import RealEsrganSuperRes
    sr = RealEsrganSuperRes(
        str(MODELS / "RealESRGANv2-animevideo-xsx4.onnx"), 4, use_cuda=False)
    try:
        frame = _test_frame(32, 40)   # 4x 模型小图即可
        out = sr.upscale(frame)
        assert out.shape == (128, 160, 3)
    finally:
        sr.close()


def test_superres_guard_switch():
    """守护包装：AI 超预算时切换 Lanczos。"""
    from player.core.player_engine import _SuperResGuard
    from player.superres.lanczos import LanczosSuperRes

    class SlowAI:
        name = "SlowAI"
        factor = 2

        def upscale(self, frame):
            import time
            time.sleep(0.05)   # 50ms/帧 > 预算
            return LanczosSuperRes(2).upscale(frame)

    guard = _SuperResGuard(SlowAI(), LanczosSuperRes(2), budget=0.02)
    frame = _test_frame(32, 40)
    # 滑动平均需要几次采样才判定（避免单帧抖动误切）
    for _ in range(4):
        out = guard.upscale(frame)
    assert out.shape == (64, 80, 3)
    # 预算 20ms < 实测 50ms → 连续几次后应已切换
    assert guard.switched is True
    out2 = guard.upscale(frame)   # 切换后走 Lanczos（不 sleep）
    assert out2.shape == (64, 80, 3)


def test_superres_guard_stays():
    """守护包装：AI 在预算内时不切换。"""
    from player.core.player_engine import _SuperResGuard
    from player.superres.lanczos import LanczosSuperRes

    class FastAI:
        name = "FastAI"
        factor = 2

        def upscale(self, frame):
            return LanczosSuperRes(2).upscale(frame)

    guard = _SuperResGuard(FastAI(), LanczosSuperRes(2), budget=1.0)
    frame = _test_frame(32, 40)
    guard.upscale(frame)
    guard.upscale(frame)
    assert guard.switched is False
