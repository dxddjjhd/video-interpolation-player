"""Lanczos 超分引擎：OpenCV INTER_LANCZOS4，CPU 实时兜底。"""

from __future__ import annotations

import cv2
import numpy as np

from .base import SuperRes


class LanczosSuperRes(SuperRes):
    name = "Lanczos"

    def __init__(self, factor: int = 2) -> None:
        assert factor in (2, 4)
        self.factor = factor

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        return cv2.resize(frame, (w * self.factor, h * self.factor),
                          interpolation=cv2.INTER_LANCZOS4)
