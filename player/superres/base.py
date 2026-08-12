"""超分辨率引擎统一接口。

约定：输入输出均为 RGB uint8 ndarray (H, W, 3)，
输出尺寸 = 输入 × factor（宽高各乘）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SuperRes(ABC):
    """超分辨率引擎抽象基类。"""

    #: 引擎显示名
    name: str = "base"
    #: 放大倍率
    factor: int = 1

    @abstractmethod
    def upscale(self, frame: np.ndarray) -> np.ndarray:
        """把一帧放大 factor 倍，返回 RGB uint8 帧。"""

    def warmup(self) -> None:
        """预热（加载模型/跑一次空推理），可选覆写。"""

    def close(self) -> None:
        """释放资源，可选覆写。"""
