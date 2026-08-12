"""超分引擎工厂：按选择创建引擎，AI 不可用时自动降级 Lanczos。"""

from __future__ import annotations

import logging

from ..utils.hardware import HardwareInfo
from .base import SuperRes
from .fsr_style import FsrStyleSuperRes
from .lanczos import LanczosSuperRes

log = logging.getLogger(__name__)

# kind 定义: off / lanczos2 / lanczos4 / fsr2 / fsr4 / ai2 / ai4
KINDS = ("off", "lanczos2", "lanczos4", "fsr2", "fsr4", "ai2", "ai4")


def create_superres(kind: str, hw: HardwareInfo,
                    model_ready: bool = False) -> SuperRes | None:
    """创建超分引擎；kind=off 返回 None。

    AI 模式：模型未就绪时抛 RuntimeError（由 UI 提示），
    创建失败自动降级为 Lanczos 并记录警告。
    """
    if kind not in KINDS:
        kind = "off"
    if kind == "off":
        return None

    factor = int(kind[-1])
    if kind.startswith("lanczos"):
        return LanczosSuperRes(factor)
    if kind.startswith("fsr"):
        # FSR 风格：Lanczos 放大 + 边缘自适应锐化（CPU 实时）
        return FsrStyleSuperRes(factor)

    # AI 模式
    if not model_ready:
        raise RuntimeError(
            "RealESRGAN 模型未就绪（模型下载中或失败），可稍后再试或使用 Lanczos")
    from .download_model import find_model
    from .real_esrgan import RealEsrganSuperRes
    model = find_model(factor)
    if model is None:
        raise RuntimeError(f"{factor}x 超分模型缺失")
    try:
        return RealEsrganSuperRes(str(model), factor, use_cuda=hw.cuda_available)
    except Exception as e:  # noqa: BLE001
        log.warning("RealESRGAN 加载失败(%s)，降级 Lanczos %dx", e, factor)
        return LanczosSuperRes(factor)
