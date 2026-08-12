"""插帧引擎统一接口。

约定：所有引擎输入输出均为 RGB uint8 ndarray (H, W, 3)，
times 为 (0,1) 区间内的中间帧时刻（相对帧间距归一化）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Interpolator(ABC):
    """帧间插值器抽象基类。"""

    #: 引擎显示名
    name: str = "base"

    @abstractmethod
    def interpolate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        times: Sequence[float],
    ) -> list[np.ndarray]:
        """对相邻帧 (frame0, frame1) 生成时刻为 times 的中间帧。

        返回与输入同尺寸的 RGB uint8 帧列表，顺序对应 times。
        """

    def warmup(self) -> None:
        """预热（如加载模型、跑一次空推理），可选覆写。"""

    def close(self) -> None:
        """释放资源，可选覆写。"""
