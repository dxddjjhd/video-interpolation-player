"""FSR 风格超分引擎：放大 + Unsharp 锐化（CPU 实时）。

模拟 AMD FSR 1.0 的"锐利放大"特性（FSR 同样包含 RCAS 锐化步骤），
面向 CPU 实时设计：
- CUBIC 放大（速度与质量均衡）
- Unsharp 锐化：out = up + k*(up - blur)，一步 cv2.addWeighted 完成，
  uint8 饱和截断天然防过冲/振铃
- 无边缘掩膜（掩膜版在弱 CPU 上大图开销过高），用较低强度
  k 控制噪声放大，视觉接近 FSR 锐化效果

720p→1440p 实测 ~20ms/帧（共享 vCPU），与 Lanczos 同级。
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import SuperRes

# 锐化强度上限（对应 FSR RCAS_LIMIT = 0.25 - 1/16，防过冲）
SHARP_LIMIT = 0.25 - 1.0 / 16.0


class FsrStyleSuperRes(SuperRes):
    name = "FSR"

    def __init__(self, factor: int = 2, sharpness: float = 0.18,
                 sigma: float = 1.2) -> None:
        assert factor in (2, 4)
        self.factor = factor
        self._k = min(max(sharpness, 0.0), SHARP_LIMIT)
        self._sigma = sigma

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        up = cv2.resize(frame, (w * self.factor, h * self.factor),
                        interpolation=cv2.INTER_CUBIC)
        if self._k <= 0:
            return up
        blur = cv2.GaussianBlur(up, (0, 0), self._sigma)
        # out = up + k*(up - blur) = (1+k)*up - k*blur（饱和截断防过冲）
        return cv2.addWeighted(up, 1.0 + self._k, blur, -self._k, 0)
