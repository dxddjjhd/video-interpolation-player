"""OpenCV 传统光流插帧引擎（CPU 兼容方案）。

算法：Farneback 稠密光流 + 前/后向 warp 加权融合。
对简单运动质量良好，快速运动/遮挡处可能有伪影，
作为无 GPU 时的降级引擎。
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from .base import Interpolator

log = logging.getLogger(__name__)


class OpticalFlowInterpolator(Interpolator):
    name = "OpenCV 光流"

    def __init__(self, pyr_scale: float = 0.5, levels: int = 3,
                 winsize: int = 21, iterations: int = 3) -> None:
        self._params = dict(
            pyr_scale=pyr_scale, levels=levels, winsize=winsize,
            iterations=iterations, poly_n=5, poly_sigma=1.1, flags=0,
        )
        self._gray0 = None
        self._gray1 = None
        self._cache = None   # (id0, id1, flow_f, flow_b)

    def _flows(self, f0: np.ndarray, f1: np.ndarray):
        """计算前向/后向光流（带相邻帧缓存，避免重复计算）。"""
        # 尺寸不同（拼接视频中途变分辨率）无法计算光流，直接抛错由上层跳过
        if f0.shape != f1.shape:
            raise ValueError(f"相邻帧尺寸不一致: {f0.shape} vs {f1.shape}")
        g0 = cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        # 缓存指纹含尺寸，防止碰撞复用旧尺寸的光流
        key = (f0.shape, g0.data.tobytes()[-4096:], g1.data.tobytes()[:4096])
        if self._cache is not None and self._cache[0] == key:
            return self._cache[2], self._cache[3]
        ff = cv2.calcOpticalFlowFarneback(g0, g1, None, **self._params)
        fb = cv2.calcOpticalFlowFarneback(g1, g0, None, **self._params)
        self._cache = (key, None, ff, fb)
        return ff, fb

    def interpolate(self, frame0, frame1, times: Sequence[float]) -> list[np.ndarray]:
        if not times:
            return []
        ff, fb = self._flows(frame0, frame1)
        h, w = frame0.shape[:2]
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        out: list[np.ndarray] = []
        for t in times:
            # 前向: f0 像素沿 t*flow 移动; 后向: f1 像素沿 (1-t)*flow 移动
            map_f = np.stack([xs + t * ff[..., 0], ys + t * ff[..., 1]], axis=-1)
            map_b = np.stack([xs - (1 - t) * fb[..., 0],
                              ys - (1 - t) * fb[..., 1]], axis=-1)
            warp0 = cv2.remap(frame0, map_f, None, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
            warp1 = cv2.remap(frame1, map_b, None, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
            mid = cv2.addWeighted(warp0, 1 - t, warp1, t, 0)
            out.append(mid)
        return out

    def close(self) -> None:
        self._cache = None
